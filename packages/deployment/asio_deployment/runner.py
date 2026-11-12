"""Shared durable job runner (S3 P0, used by S3 deployments and S5 workspaces).

Semantics, in one place:
- **Claim**: one worker per job via `FOR UPDATE SKIP LOCKED`; a claim sets a lease and bumps
  `generation`, which fences every later write (a worker that lost its lease cannot finish the job).
- **Lease**: handlers call `ctx.heartbeat()` during long work; an expired lease makes the job
  claimable again, and the next worker continues from the persisted state (handlers are idempotent).
- **One job per target at a time**: if another job for the same `target_ref` holds a live lease,
  the newer job is requeued without spending an attempt (the older one runs first).
- **Retry**: `RetryLater` (or any unexpected error) requeues with exponential backoff until
  `max_attempts`; `JobFailed` ends the job with a result; success records the handler's result.
- **Verification**: a handler that stops something sets `ctx.shutdown_verified_at` only after it has
  proof; S1 releases GPUs only for jobs that carry it.

S5 registers its handlers with `runner.register(JobKind.START_WORKSPACE, handler)` and enqueues
jobs with `owner_module="workspaces"`; nothing else in this module is S3-specific.
"""

from __future__ import annotations

import logging
import random
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from .contracts import Job, JobKind, JobState
from .store import Store

log = logging.getLogger("asio_deployment.runner")

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(UTC)


class RetryLater(Exception):
    """Transient: requeue with backoff (spends an attempt)."""


class JobFailed(Exception):
    """Permanent: finish the job as FAILED with this result."""

    def __init__(self, message: str, result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.result = result or {}


class LeaseLost(Exception):
    """Another worker owns the job now; stop without writing anything."""


@dataclass
class JobContext:
    job: Job
    store: Store
    clock: Clock
    lease_s: float
    shutdown_verified_at: datetime | None = None
    notes: list[str] = field(default_factory=list)

    def heartbeat(self) -> None:
        if not self.store.extend_lease(self.job, self.lease_s, self.clock()):
            raise LeaseLost(self.job.id)


Handler = Callable[[JobContext], dict[str, Any]]


def default_backoff(attempt: int) -> float:
    return float(min(60, 2**attempt)) + random.uniform(0, 1)


class JobRunner:
    def __init__(
        self,
        store: Store,
        worker_id: str,
        *,
        lease_s: float = 60.0,
        clock: Clock = utcnow,
        backoff: Callable[[int], float] = default_backoff,
        busy_requeue_s: float = 2.0,
    ) -> None:
        self.store = store
        self.worker_id = worker_id
        self.lease_s = lease_s
        self.clock = clock
        self.backoff = backoff
        self.busy_requeue_s = busy_requeue_s
        self.handlers: dict[JobKind, Handler] = {}

    def register(self, kind: JobKind, handler: Handler) -> None:
        self.handlers[kind] = handler

    def _blocked_by_older(self, job: Job, now: datetime) -> Job | None:
        for other in self.store.list_jobs(target_ref=job.target_ref):
            if (
                other.id != job.id
                and other.state == JobState.RUNNING
                and other.lease_until is not None
                and other.lease_until > now
                and (other.created_at, other.id) < (job.created_at, job.id)
            ):
                return other
        return None

    def _finish(self, job: Job, **updates: Any) -> None:
        done = job.model_copy(update={**updates, "updated_at": self.clock(), "lease_until": None})
        if not self.store.finish_job(done):
            log.warning("job %s: lost ownership before finishing (generation fenced)", job.id)

    def run_once(self) -> Job | None:
        """Claim and run at most one job. Returns the claimed job (as claimed) or None."""
        if not self.handlers:
            return None
        now = self.clock()
        job = self.store.claim_job(self.handlers.keys(), self.worker_id, self.lease_s, now)
        if job is None:
            return None

        if job.attempt > job.max_attempts:
            self._finish(
                job,
                state=JobState.FAILED,
                last_error=job.last_error or "max attempts exceeded (lease expired)",
                result={**(job.result or {}), "max_attempts_exceeded": True},
            )
            return job

        older = self._blocked_by_older(job, now)
        if older is not None:
            self._finish(
                job,
                state=JobState.QUEUED,
                attempt=job.attempt - 1,
                run_after=now + timedelta(seconds=self.busy_requeue_s),
                last_error=f"waiting for job {older.id} on the same target",
            )
            return job

        ctx = JobContext(job=job, store=self.store, clock=self.clock, lease_s=self.lease_s)
        try:
            result = self.handlers[job.kind](ctx)
        except LeaseLost:
            log.warning("job %s: lease lost during execution", job.id)
            return job
        except JobFailed as e:
            self._finish(
                job,
                state=JobState.FAILED,
                last_error=str(e),
                result=e.result,
                shutdown_verified_at=ctx.shutdown_verified_at,
            )
            return job
        except Exception as e:  # RetryLater and anything unexpected
            msg = str(e) if isinstance(e, RetryLater) else f"{e.__class__.__name__}: {e}"
            if not isinstance(e, RetryLater):
                log.error("job %s crashed:\n%s", job.id, traceback.format_exc())
            if job.attempt >= job.max_attempts:
                self._finish(
                    job,
                    state=JobState.FAILED,
                    last_error=msg,
                    result={"retries_exhausted": True},
                    shutdown_verified_at=ctx.shutdown_verified_at,
                )
            else:
                self._finish(
                    job,
                    state=JobState.QUEUED,
                    last_error=msg,
                    run_after=self.clock() + timedelta(seconds=self.backoff(job.attempt)),
                )
            return job

        self._finish(
            job,
            state=JobState.SUCCEEDED,
            result=result,
            last_error=None,
            shutdown_verified_at=ctx.shutdown_verified_at,
        )
        return job
