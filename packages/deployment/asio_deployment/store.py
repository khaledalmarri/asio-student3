"""Persistence for deployments, serving pools and the shared `jobs` table.

`MemoryStore` is for tests and the stand-alone dev server. `PostgresStore` is the real one:
PostgreSQL is the source of truth (contract §1), jobs are claimed with `FOR UPDATE SKIP LOCKED`
and every write a worker makes is fenced by the job's `generation`, so a worker whose lease
expired cannot overwrite the result of the worker that took the job over.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .contracts import Deployment, Job, JobKind, JobState, ServingPool

M = TypeVar("M", bound=BaseModel)

SCHEMA_SQL = Path(__file__).with_name("sql") / "0001_s3_proposed.sql"


class DuplicateRequestRef(Exception):
    """A deployment with the same (owner_id, request_ref) already exists."""

    def __init__(self, existing: Deployment) -> None:
        super().__init__(existing.id)
        self.existing = existing


class Store(Protocol):
    # deployments + pool + first job are created in one transaction
    def create_deployment(self, d: Deployment, pool: ServingPool, job: Job | None) -> None: ...
    def get_deployment(self, deployment_id: str) -> Deployment | None: ...
    def find_deployment(self, owner_id: str, request_ref: str) -> Deployment | None: ...
    def save_deployment(self, d: Deployment) -> None: ...
    def list_deployments(self) -> list[Deployment]: ...

    def get_pool(self, pool_id: str) -> ServingPool | None: ...
    def cas_pool(self, pool: ServingPool, expected_generation: int) -> bool: ...
    def list_pools(self) -> list[ServingPool]: ...

    def enqueue_job(self, job: Job) -> Job: ...
    def claim_job(
        self, kinds: Iterable[JobKind], worker_id: str, lease_s: float, now: datetime
    ) -> Job | None: ...
    def extend_lease(self, job: Job, lease_s: float, now: datetime) -> bool: ...
    def finish_job(self, job: Job) -> bool: ...
    def get_job(self, job_id: str) -> Job | None: ...
    def list_jobs(self, target_ref: str | None = None) -> list[Job]: ...


# --------------------------------------------------------------------------- in-memory


class MemoryStore:
    """Same semantics as `PostgresStore`, guarded by one lock. Not for production."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._deployments: dict[str, Deployment] = {}
        self._pools: dict[str, ServingPool] = {}
        self._jobs: dict[str, Job] = {}

    @staticmethod
    def _c(m: M) -> M:
        return m.model_copy(deep=True)

    def create_deployment(self, d: Deployment, pool: ServingPool, job: Job | None) -> None:
        with self._lock:
            existing = self.find_deployment(d.owner_id, d.request_ref)
            if existing is not None:
                raise DuplicateRequestRef(existing)
            self._deployments[d.id] = self._c(d)
            self._pools[pool.id] = self._c(pool)
            if job is not None:
                self.enqueue_job(job)

    def get_deployment(self, deployment_id: str) -> Deployment | None:
        with self._lock:
            d = self._deployments.get(deployment_id)
            return self._c(d) if d else None

    def find_deployment(self, owner_id: str, request_ref: str) -> Deployment | None:
        with self._lock:
            for d in self._deployments.values():
                if d.owner_id == owner_id and d.request_ref == request_ref:
                    return self._c(d)
            return None

    def save_deployment(self, d: Deployment) -> None:
        with self._lock:
            self._deployments[d.id] = self._c(d)

    def list_deployments(self) -> list[Deployment]:
        with self._lock:
            return [self._c(d) for d in self._deployments.values()]

    def get_pool(self, pool_id: str) -> ServingPool | None:
        with self._lock:
            p = self._pools.get(pool_id)
            return self._c(p) if p else None

    def cas_pool(self, pool: ServingPool, expected_generation: int) -> bool:
        with self._lock:
            cur = self._pools.get(pool.id)
            if cur is None or cur.generation != expected_generation:
                return False
            self._pools[pool.id] = self._c(pool)
            return True

    def list_pools(self) -> list[ServingPool]:
        with self._lock:
            return [self._c(p) for p in self._pools.values()]

    def enqueue_job(self, job: Job) -> Job:
        with self._lock:
            for j in self._jobs.values():
                if j.owner_module == job.owner_module and j.request_ref == job.request_ref:
                    return self._c(j)
            self._jobs[job.id] = self._c(job)
            return self._c(job)

    def claim_job(
        self, kinds: Iterable[JobKind], worker_id: str, lease_s: float, now: datetime
    ) -> Job | None:
        ks = set(kinds)
        with self._lock:
            ready = [
                j
                for j in self._jobs.values()
                if j.kind in ks
                and (
                    (j.state == JobState.QUEUED and j.run_after <= now)
                    or (
                        j.state == JobState.RUNNING
                        and j.lease_until is not None
                        and j.lease_until < now
                    )
                )
            ]
            if not ready:
                return None
            j = min(ready, key=lambda x: (x.run_after, x.created_at))
            j.state = JobState.RUNNING
            j.attempt += 1
            j.generation += 1
            j.worker_id = worker_id
            j.lease_until = now + timedelta(seconds=lease_s)
            j.updated_at = now
            return self._c(j)

    def _owned(self, job: Job) -> Job | None:
        cur = self._jobs.get(job.id)
        if cur is None or cur.generation != job.generation or cur.worker_id != job.worker_id:
            return None
        return cur

    def extend_lease(self, job: Job, lease_s: float, now: datetime) -> bool:
        with self._lock:
            cur = self._owned(job)
            if cur is None or cur.state != JobState.RUNNING:
                return False
            cur.lease_until = now + timedelta(seconds=lease_s)
            cur.updated_at = now
            return True

    def finish_job(self, job: Job) -> bool:
        with self._lock:
            if self._owned(job) is None:
                return False
            self._jobs[job.id] = self._c(job)
            return True

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return self._c(j) if j else None

    def list_jobs(self, target_ref: str | None = None) -> list[Job]:
        with self._lock:
            return [
                self._c(j)
                for j in sorted(self._jobs.values(), key=lambda x: x.created_at)
                if target_ref is None or j.target_ref == target_ref
            ]


# --------------------------------------------------------------------------- PostgreSQL

_JSON_FIELDS = {"gpu_uuids", "config", "supported_model_ids", "payload", "result"}


def _params(m: BaseModel) -> dict[str, Any]:
    from psycopg.types.json import Jsonb

    out: dict[str, Any] = {}
    for k, v in m.model_dump(mode="python").items():
        if isinstance(v, Enum):
            v = v.value
        if k in _JSON_FIELDS and v is not None:
            v = Jsonb(v)
        out[k] = v
    return out


def _insert_sql(table: str, cols: list[str]) -> str:
    values = ", ".join(f"%({c})s" for c in cols)
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({values})"


class PostgresStore:
    """psycopg 3. One short connection per call (MVP); swap for a pool under load."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _conn(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.dsn, row_factory=dict_row)

    def apply_schema(self) -> None:
        """Dev/test only. In the team repo S1's migration creates these tables."""
        with self._conn() as c:
            c.execute(SCHEMA_SQL.read_text())

    # -- deployments
    def create_deployment(self, d: Deployment, pool: ServingPool, job: Job | None) -> None:
        import psycopg

        try:
            with self._conn() as c:
                dp = _params(d)
                c.execute(_insert_sql("deployments", list(dp)), dp)
                pp = _params(pool)
                c.execute(_insert_sql("serving_pools", list(pp)), pp)
                if job is not None:
                    jp = _params(job)
                    c.execute(_insert_sql("jobs", list(jp)), jp)
        except psycopg.errors.UniqueViolation:
            existing = self.find_deployment(d.owner_id, d.request_ref)
            if existing is None:
                raise
            raise DuplicateRequestRef(existing) from None

    def get_deployment(self, deployment_id: str) -> Deployment | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM deployments WHERE id=%s", (deployment_id,)).fetchone()
        return Deployment.model_validate(row) if row else None

    def find_deployment(self, owner_id: str, request_ref: str) -> Deployment | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM deployments WHERE owner_id=%s AND request_ref=%s",
                (owner_id, request_ref),
            ).fetchone()
        return Deployment.model_validate(row) if row else None

    def save_deployment(self, d: Deployment) -> None:
        p = _params(d)
        sets = ", ".join(f"{k}=%({k})s" for k in p if k != "id")
        with self._conn() as c:
            c.execute(f"UPDATE deployments SET {sets} WHERE id=%(id)s", p)

    def list_deployments(self) -> list[Deployment]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM deployments ORDER BY created_at").fetchall()
        return [Deployment.model_validate(r) for r in rows]

    # -- pools
    def get_pool(self, pool_id: str) -> ServingPool | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM serving_pools WHERE id=%s", (pool_id,)).fetchone()
        return ServingPool.model_validate(row) if row else None

    def cas_pool(self, pool: ServingPool, expected_generation: int) -> bool:
        p = _params(pool) | {"expected_generation": expected_generation}
        sets = ", ".join(f"{k}=%({k})s" for k in _params(pool) if k != "id")
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE serving_pools SET {sets} WHERE id=%(id)s "
                "AND generation=%(expected_generation)s",
                p,
            )
            return bool(cur.rowcount == 1)

    def list_pools(self) -> list[ServingPool]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM serving_pools ORDER BY id").fetchall()
        return [ServingPool.model_validate(r) for r in rows]

    # -- jobs
    def enqueue_job(self, job: Job) -> Job:
        p = _params(job)
        with self._conn() as c:
            c.execute(
                _insert_sql("jobs", list(p))
                + " ON CONFLICT (owner_module, request_ref) DO NOTHING",
                p,
            )
            row = c.execute(
                "SELECT * FROM jobs WHERE owner_module=%s AND request_ref=%s",
                (job.owner_module, job.request_ref),
            ).fetchone()
        return Job.model_validate(row)

    def claim_job(
        self, kinds: Iterable[JobKind], worker_id: str, lease_s: float, now: datetime
    ) -> Job | None:
        sql = """
        WITH c AS (
            SELECT id FROM jobs
            WHERE kind = ANY(%(kinds)s)
              AND ((state = 'QUEUED' AND run_after <= %(now)s)
                OR (state = 'RUNNING' AND lease_until < %(now)s))
            ORDER BY run_after, created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE jobs j SET state = 'RUNNING', attempt = j.attempt + 1,
            generation = j.generation + 1, worker_id = %(worker)s,
            lease_until = %(now)s + make_interval(secs => %(lease)s), updated_at = %(now)s
        FROM c WHERE j.id = c.id
        RETURNING j.*
        """
        with self._conn() as c:
            row = c.execute(
                sql,
                {
                    "kinds": [k.value for k in kinds],
                    "now": now,
                    "worker": worker_id,
                    "lease": lease_s,
                },
            ).fetchone()
        return Job.model_validate(row) if row else None

    def extend_lease(self, job: Job, lease_s: float, now: datetime) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET lease_until = %(now)s + make_interval(secs => %(lease)s), "
                "updated_at=%(now)s WHERE id=%(id)s AND generation=%(gen)s AND worker_id=%(w)s "
                "AND state='RUNNING'",
                {
                    "now": now,
                    "lease": lease_s,
                    "id": job.id,
                    "gen": job.generation,
                    "w": job.worker_id,
                },
            )
            return bool(cur.rowcount == 1)

    def finish_job(self, job: Job) -> bool:
        p = _params(job)
        keep = [
            "state",
            "attempt",
            "lease_until",
            "run_after",
            "result",
            "last_error",
            "shutdown_verified_at",
            "updated_at",
        ]
        sets = ", ".join(f"{k}=%({k})s" for k in keep)
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE jobs SET {sets} WHERE id=%(id)s AND generation=%(generation)s "
                "AND worker_id=%(worker_id)s",
                p,
            )
            return bool(cur.rowcount == 1)

    def get_job(self, job_id: str) -> Job | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
        return Job.model_validate(row) if row else None

    def list_jobs(self, target_ref: str | None = None) -> list[Job]:
        with self._conn() as c:
            if target_ref is None:
                rows = c.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM jobs WHERE target_ref=%s ORDER BY created_at", (target_ref,)
                ).fetchall()
        return [Job.model_validate(r) for r in rows]
