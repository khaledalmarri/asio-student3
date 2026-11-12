"""Deployment Controller (S3 P0).

Two halves:
- **API side** (`create`, `request_drain`, `request_stop`, `get`): validates, persists the
  deployment + a STARTING pool record + a DEPLOY job in one transaction, and returns. It never
  touches Docker.
- **Worker side** (`handle_deploy`, `handle_stop`, `enqueue_expiring`): runs inside the
  privileged worker through the shared job runner.

ServingPool readiness has exactly one writer, `_publish`, which enforces the allowed transitions
and fences every write on the pool's `generation`. READY is only published by `handle_deploy`
after the entry point answers `/health` *and* the served model is listed by `/v1/models`.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from .contracts import (
    TERMINAL_STATES,
    AllocationKind,
    AllocationState,
    Backend,
    Deployment,
    DeploymentConfig,
    DesiredState,
    Job,
    JobKind,
    JobState,
    ObservedState,
    Readiness,
    ResourceClass,
    ServingPool,
)
from .errors import ApiError
from .runner import Clock, JobContext, JobFailed, RetryLater, utcnow
from .runtime import ProbeResult, Runtime, RuntimeError_, in_flight, preflight, probe
from .s1 import S1Client, S1Error
from .store import DuplicateRequestRef, Store
from .templates import TemplateError, build_launch_spec, gpu_memory_utilization

OWNER_MODULE = "deployment"

# Allowed readiness transitions. Same-state writes are refused too: every publish is an event.
_TRANSITIONS: dict[Readiness, set[Readiness]] = {
    Readiness.STARTING: {
        Readiness.READY,
        Readiness.UNHEALTHY,
        Readiness.DRAINING,
        Readiness.STOPPED,
    },
    Readiness.READY: {Readiness.DRAINING, Readiness.UNHEALTHY},
    Readiness.UNHEALTHY: {Readiness.DRAINING, Readiness.STOPPED},
    Readiness.DRAINING: {Readiness.STOPPED, Readiness.UNHEALTHY},
    Readiness.STOPPED: set(),
}


class CreateDeploymentRequest(BaseModel):
    """`POST /api/v1/deployments` body (contract §5.4 example)."""

    request_ref: str = Field(min_length=1, max_length=200)
    resource_class: ResourceClass
    allocation_id: str | None = None
    model_id: str = Field(min_length=1)
    backend: Backend
    parallelism_plan_id: str | None = None
    desired_state: DesiredState = DesiredState.RUNNING
    config: DeploymentConfig


class StaleGeneration(Exception):
    pass


class InvalidTransition(Exception):
    pass


@dataclass
class ControllerSettings:
    advertise_host: str = field(
        default_factory=lambda: os.environ.get("ASIO_S3_ADVERTISE_HOST", "127.0.0.1")
    )
    start_timeout_s: float = 900.0
    probe_interval_s: float = 5.0
    default_drain_timeout_s: float = 30.0
    max_drain_timeout_s: float = 600.0
    # Start stopping a serving deployment this long before its allocation ends (plus drain time).
    expiry_margin_s: float = 60.0
    # Pre-launch GPU check: free memory needed = vLLM's share of the GPU + this margin.
    gpu_free_margin_mib: int = 1024
    # True = refuse a GPU that has any process on it. Off by default: on a shared node (gpu-02)
    # other tenants' processes are normal, and S1's allocation is what makes the GPU ours.
    require_exclusive_gpus: bool = field(
        default_factory=lambda: os.environ.get("ASIO_S3_REQUIRE_EXCLUSIVE_GPUS") == "1"
    )


class DeploymentController:
    def __init__(
        self,
        store: Store,
        s1: S1Client | None = None,
        runtime: Runtime | None = None,
        settings: ControllerSettings | None = None,
        clock: Clock = utcnow,
        sleep: Callable[[float], None] = time.sleep,
        prober: Callable[[str, str], ProbeResult] = probe,
        in_flight_fn: Callable[[str], int | None] = in_flight,
    ) -> None:
        self.store = store
        self.s1 = s1
        self.runtime = runtime
        self.settings = settings or ControllerSettings()
        self.clock = clock
        self.sleep = sleep
        self.prober = prober
        self.in_flight = in_flight_fn

    # ================================================================== API side (no Docker)

    def create(self, caller_id: str, req: CreateDeploymentRequest) -> tuple[Deployment, Job, bool]:
        """Returns (deployment, deploy job, created). Replays with the same payload are no-ops."""
        existing = self.store.find_deployment(caller_id, req.request_ref)
        if existing is not None:
            return self._replay(existing, req)

        self._validate(req)
        now = self.clock()
        dep_id = f"dep_{uuid.uuid4().hex[:16]}"
        pool_id = f"pool_{dep_id[4:]}"
        endpoint = req.config.endpoint_url or (
            f"http://{self.settings.advertise_host}:{req.config.port}"
        )
        d = Deployment(
            id=dep_id,
            owner_id=caller_id,
            request_ref=req.request_ref,
            resource_class=req.resource_class,
            allocation_id=req.allocation_id,
            parallelism_plan_id=req.parallelism_plan_id,
            model_id=req.model_id,
            backend=req.backend,
            desired_state=DesiredState.RUNNING,
            observed_state=ObservedState.PENDING,
            pool_id=pool_id,
            generation=1,
            config=req.config,
            created_at=now,
            updated_at=now,
        )
        pool = ServingPool(
            id=pool_id,
            deployment_id=dep_id,
            allocation_id=req.allocation_id,
            resource_class=req.resource_class,
            backend=req.backend,
            endpoint_url=endpoint,
            supported_model_ids=[req.model_id],
            readiness=Readiness.STARTING,
            rated_capacity_rps=req.config.rated_capacity_rps,
            generation=1,
            updated_at=now,
        )
        job = self._new_job(JobKind.DEPLOY, f"deploy:{dep_id}", d, now)
        try:
            self.store.create_deployment(d, pool, job)
        except DuplicateRequestRef as e:  # lost a race with an identical request
            return self._replay(e.existing, req)
        return d, job, True

    def _replay(
        self, existing: Deployment, req: CreateDeploymentRequest
    ) -> tuple[Deployment, Job, bool]:
        same = (
            existing.resource_class == req.resource_class
            and existing.allocation_id == req.allocation_id
            and existing.model_id == req.model_id
            and existing.backend == req.backend
            and existing.parallelism_plan_id == req.parallelism_plan_id
            and existing.config == req.config
        )
        if not same:
            raise ApiError(
                409,
                "IDEMPOTENCY_CONFLICT",
                "request_ref was already used with a different payload",
                {"request_ref": req.request_ref, "deployment_id": existing.id},
            )
        jobs = [j for j in self.store.list_jobs(existing.id) if j.kind == JobKind.DEPLOY]
        return existing, jobs[0], False

    def _validate(self, req: CreateDeploymentRequest) -> None:
        if req.desired_state != DesiredState.RUNNING:
            raise ApiError(422, "INVALID_DESIRED_STATE", "a new deployment must be RUNNING")
        if req.resource_class == ResourceClass.GPU_SERVING:
            if not req.allocation_id:
                raise ApiError(
                    422, "ALLOCATION_REQUIRED", "GPU_SERVING deployments need an S1 allocation_id"
                )
            if req.backend != Backend.VLLM:
                raise ApiError(
                    422,
                    "BACKEND_NOT_SUPPORTED",
                    f"GPU_SERVING backend {req.backend} is not supported yet "
                    "(Dynamo is the P1 CT-18 trial)",
                    {"backend": req.backend},
                )
            if req.config.endpoint_url:
                raise ApiError(
                    422, "INVALID_CONFIG", "endpoint_url is set by S3 for GPU_SERVING deployments"
                )
            if self.s1 is not None:  # fail fast; the worker re-checks at launch time
                self._check_allocation(req.allocation_id, permanent_as=ApiError)
        else:
            if not req.config.endpoint_url:
                raise ApiError(
                    422,
                    "ENDPOINT_REQUIRED",
                    "ROUTING_ONLY entries register an existing entry point URL",
                )

    def _check_allocation(self, allocation_id: str, permanent_as: type[Exception]) -> Any:
        assert self.s1 is not None
        try:
            a = self.s1.get_allocation(allocation_id)
        except S1Error as e:
            if permanent_as is ApiError:
                status = 422 if e.permanent else 503
                raise ApiError(
                    status, "ALLOCATION_UNAVAILABLE", str(e), {"allocation_id": allocation_id}
                ) from e
            if e.permanent:
                raise JobFailed(str(e), {"allocation_id": allocation_id}) from e
            raise RetryLater(str(e)) from e
        problem = None
        if a.kind != AllocationKind.SERVING:
            problem = f"allocation kind is {a.kind}, not SERVING"
        elif a.state not in (AllocationState.COMMITTED, AllocationState.ACTIVE):
            problem = f"allocation state is {a.state}"
        elif not a.gpu_uuids:
            problem = "allocation has no GPU UUIDs"
        elif a.end_at <= self.clock():
            problem = "allocation window has ended"
        if problem:
            if permanent_as is ApiError:
                raise ApiError(422, "ALLOCATION_INVALID", problem, {"allocation_id": allocation_id})
            raise JobFailed(problem, {"allocation_id": allocation_id})
        return a

    def _new_job(
        self,
        kind: JobKind,
        request_ref: str,
        d: Deployment,
        now: datetime,
        payload: dict[str, Any] | None = None,
    ) -> Job:
        return Job(
            id=f"job_{uuid.uuid4().hex[:16]}",
            kind=kind,
            owner_module=OWNER_MODULE,
            request_ref=request_ref,
            allocation_id=d.allocation_id,
            target_ref=d.id,
            run_after=now,
            payload=payload or {},
            created_at=now,
            updated_at=now,
        )

    def get(self, deployment_id: str) -> Deployment:
        d = self.store.get_deployment(deployment_id)
        if d is None:
            raise ApiError(
                404,
                "DEPLOYMENT_NOT_FOUND",
                f"deployment {deployment_id!r} not found",
                {"deployment_id": deployment_id},
            )
        return d

    def request_drain(
        self,
        deployment_id: str,
        reason: str = "OPERATOR_DRAIN",
        drain_timeout_s: float | None = None,
    ) -> tuple[Deployment, Job]:
        t = self.settings.default_drain_timeout_s if drain_timeout_s is None else drain_timeout_s
        return self._request_stop(deployment_id, reason, min(t, self.settings.max_drain_timeout_s))

    def request_stop(
        self, deployment_id: str, reason: str = "OPERATOR_STOP"
    ) -> tuple[Deployment, Job]:
        return self._request_stop(deployment_id, reason, 0.0)

    def _request_stop(
        self, deployment_id: str, reason: str, drain_timeout_s: float
    ) -> tuple[Deployment, Job]:
        d = self.get(deployment_id)
        jobs = [j for j in self.store.list_jobs(d.id) if j.kind == JobKind.STOP_DEPLOY]
        live = [j for j in jobs if j.state in (JobState.QUEUED, JobState.RUNNING)]
        if live:
            return d, live[0]
        if d.observed_state == ObservedState.STOPPED:
            if jobs:
                return d, jobs[-1]
        now = self.clock()
        if d.desired_state != DesiredState.STOPPED:
            d.desired_state = DesiredState.STOPPED
            d.updated_at = now
            self.store.save_deployment(d)
        # A failed (unverified) stop can be retried: each retry gets a new request_ref.
        job = self._new_job(
            JobKind.STOP_DEPLOY,
            f"stop:{d.id}:{len(jobs)}",
            d,
            now,
            {"reason": reason, "drain_timeout_s": drain_timeout_s},
        )
        return d, self.store.enqueue_job(job)

    # ================================================================== readiness (single writer)

    def publish_pool_state(
        self, deployment_id: str, state: Readiness, generation: int, **fields: Any
    ) -> ServingPool:
        """Contract §5.5 interface. `generation` is the new generation and must be current + 1."""
        d = self.get(deployment_id)
        pool = self.store.get_pool(d.pool_id)
        if pool is None:
            raise StaleGeneration(f"pool {d.pool_id} missing")
        if generation != pool.generation + 1:
            raise StaleGeneration(f"pool {pool.id} is at generation {pool.generation}")
        if state not in _TRANSITIONS[pool.readiness]:
            raise InvalidTransition(f"{pool.readiness} -> {state} is not allowed")
        now = self.clock()
        new = pool.model_copy(
            update={"readiness": state, "generation": generation, "updated_at": now, **fields}
        )
        if not self.store.cas_pool(new, expected_generation=pool.generation):
            raise StaleGeneration(f"pool {pool.id} changed concurrently")
        d.generation = generation
        d.updated_at = now
        self.store.save_deployment(d)
        return new

    def _publish(self, d: Deployment, state: Readiness, **fields: Any) -> ServingPool:
        pool = self.store.get_pool(d.pool_id)
        assert pool is not None
        if pool.readiness == state:
            return pool
        p = self.publish_pool_state(d.id, state, pool.generation + 1, **fields)
        d.generation = p.generation
        return p

    def _observe(self, d: Deployment, state: ObservedState, error: str | None = None) -> None:
        fresh = self.store.get_deployment(d.id)
        if fresh is not None:  # keep fields written by publish_pool_state / API (desired_state)
            d.generation = fresh.generation
            d.desired_state = fresh.desired_state
        d.observed_state = state
        if error is not None:
            d.last_error = error
        d.updated_at = self.clock()
        self.store.save_deployment(d)

    # ================================================================== worker side

    def handle_deploy(self, ctx: JobContext) -> dict[str, Any]:
        d = self.get(ctx.job.target_ref)
        if d.desired_state == DesiredState.STOPPED or d.observed_state in TERMINAL_STATES:
            return {"skipped": True, "reason": "stop requested before launch"}
        if d.resource_class == ResourceClass.ROUTING_ONLY:
            return self._deploy_routing_only(ctx, d)

        assert self.s1 is not None and self.runtime is not None, "worker needs S1 + runtime"
        alloc = self._check_allocation(d.allocation_id or "", permanent_as=JobFailed)
        d.gpu_uuids = list(alloc.gpu_uuids)
        try:
            spec = build_launch_spec(d)
        except TemplateError as e:
            self._publish(d, Readiness.UNHEALTHY)
            self._observe(d, ObservedState.FAILED, f"template: {e}")
            raise JobFailed(str(e), {"launched": False}) from e
        d.container_name = spec.name

        if not self.runtime.exists(spec.name):  # a retry after lease loss reuses the container
            gpu_check = self._preflight_gpus(ctx, d)
            self._observe(d, ObservedState.STARTING)
            try:
                self.runtime.start(spec)
            except Exception as e:
                return self._fail_launch(ctx, d, f"start failed: {e}")
        else:
            gpu_check = {"skipped": "container already running from an earlier attempt"}
            self._observe(d, ObservedState.STARTING)

        pool = self.store.get_pool(d.pool_id)
        assert pool is not None
        deadline = self.clock() + timedelta(seconds=self.settings.start_timeout_s)
        last = ProbeResult(False, "not probed")
        while self.clock() < deadline:
            ctx.heartbeat()
            if self.get(d.id).desired_state == DesiredState.STOPPED:
                return {
                    "aborted": True,
                    "reason": "stop requested during startup",
                    "last_probe": last.detail,
                }
            if not self.runtime.is_running(spec.name):
                return self._fail_launch(
                    ctx,
                    d,
                    "container exited during startup: " + self.runtime.logs(spec.name)[-1500:],
                )
            last = self.prober(pool.endpoint_url, d.config.served_model_name)
            if last.ok:
                break
            self.sleep(self.settings.probe_interval_s)
        else:
            return self._fail_launch(
                ctx, d, f"not healthy within {self.settings.start_timeout_s:.0f}s: {last.detail}"
            )

        try:
            self.s1.activate_allocation(alloc.id, idempotency_key=f"activate:{d.id}")
        except S1Error as e:
            if e.permanent:
                return self._fail_launch(ctx, d, f"S1 refused activation: {e}")
            raise RetryLater(f"S1 activate: {e}") from e
        now = self.clock()
        self._publish(d, Readiness.READY, health_checked_at=now)
        self._observe(d, ObservedState.READY)
        return {
            "ready": True,
            "container": spec.name,
            "gpu_uuids": d.gpu_uuids,
            "endpoint_url": pool.endpoint_url,
            "probe": last.detail,
            "gpu_check": gpu_check,
            "evidence_label": "REAL_GPU",
        }

    def _preflight_gpus(self, ctx: JobContext, d: Deployment) -> dict[str, Any]:
        """Check the granted GPUs on the real node right before `docker run`.

        Busy GPU: retried with backoff (it may be our own previous container still freeing
        memory); after the job's last attempt the deployment FAILS with `gpu_busy` and nothing was
        launched, so there is nothing to clean up. S3 does not release the allocation in that case:
        the GPU is busy with something S3 did not start, so S1 decides (reassign or quarantine).
        A GPU UUID that is not on this node fails at once.
        """
        assert self.runtime is not None
        util = gpu_memory_utilization(d.config.vllm_args)
        try:
            status = self.runtime.gpu_status(d.gpu_uuids)
        except RuntimeError_ as e:
            if ctx.job.attempt < ctx.job.max_attempts:
                raise RetryLater(f"GPU check unavailable: {e}") from e
            return self._fail_preflight(d, [f"GPU check unavailable: {e}"], {}, gpu_busy=False)
        totals = [st.total_mib for st in status.values()] or [0]
        need = int(util * max(totals)) + self.settings.gpu_free_margin_mib
        res = preflight(
            status, d.gpu_uuids, need, require_exclusive=self.settings.require_exclusive_gpus
        )
        if res.ok:
            return res.evidence
        if not res.permanent and ctx.job.attempt < ctx.job.max_attempts:
            raise RetryLater("GPU busy: " + "; ".join(res.problems))
        return self._fail_preflight(d, res.problems, res.evidence, gpu_busy=not res.permanent)

    def _fail_preflight(
        self, d: Deployment, problems: list[str], evidence: dict[str, Any], *, gpu_busy: bool
    ) -> dict[str, Any]:
        msg = "GPU not usable: " + "; ".join(problems)
        self._publish(d, Readiness.UNHEALTHY)
        self._observe(d, ObservedState.FAILED, msg)
        raise JobFailed(
            msg,
            {
                "ready": False,
                "launched": False,
                "gpu_busy": gpu_busy,
                "gpu_check": evidence,
                "problems": problems,
                "s1_action": "reassign or quarantine; S3 launched nothing and released nothing",
            },
        )

    def _deploy_routing_only(self, ctx: JobContext, d: Deployment) -> dict[str, Any]:
        pool = self.store.get_pool(d.pool_id)
        assert pool is not None
        res = self.prober(pool.endpoint_url, d.config.served_model_name)
        if not res.ok:
            raise RetryLater(f"routing-only upstream not ready: {res.detail}")
        self._publish(d, Readiness.READY, health_checked_at=self.clock())
        self._observe(d, ObservedState.READY)
        return {"ready": True, "routing_only": True, "probe": res.detail}

    def _fail_launch(self, ctx: JobContext, d: Deployment, error: str) -> dict[str, Any]:
        """CT-05: a failed model start is never READY, and the GPU stays protected until the
        cleanup is verified (then S1 may release it); unverified cleanup means quarantine."""
        assert self.runtime is not None and self.s1 is not None
        self._publish(d, Readiness.UNHEALTHY)
        self._observe(d, ObservedState.FAILED, error)
        verified, evidence = self._stop_and_verify(d)
        result: dict[str, Any] = {
            "ready": False,
            "error": error[:2000],
            "shutdown_verified": verified,
            "shutdown_evidence": evidence,
        }
        if verified:
            ctx.shutdown_verified_at = self.clock()
            self._release(d, evidence, result)
        else:
            result["quarantine_required"] = True
        raise JobFailed(error[:500], result)

    def _stop_and_verify(self, d: Deployment) -> tuple[bool, dict[str, Any]]:
        assert self.runtime is not None
        name = d.container_name
        if not name:
            return True, {"container": None, "note": "no container was ever started"}
        pids = self.runtime.host_pids(name) if self.runtime.exists(name) else []
        self.runtime.stop(name)
        return self.runtime.verify_stopped(name, pids, d.gpu_uuids)

    def _release(self, d: Deployment, evidence: dict[str, Any], result: dict[str, Any]) -> None:
        assert self.s1 is not None
        if not d.allocation_id:
            return
        try:
            self.s1.release_allocation(d.allocation_id, f"release:{d.id}", evidence)
            result["allocation_released"] = True
        except S1Error as e:  # proof is persisted on the job; S1's reconciler can still release
            result["allocation_released"] = False
            result["release_error"] = str(e)

    def handle_stop(self, ctx: JobContext) -> dict[str, Any]:
        d = self.get(ctx.job.target_ref)
        if d.observed_state == ObservedState.STOPPED:
            return {"already_stopped": True}
        reason = str(ctx.job.payload.get("reason", "STOP"))
        drain_s = float(ctx.job.payload.get("drain_timeout_s", 0.0))
        pool = self.store.get_pool(d.pool_id)
        assert pool is not None

        # 1. Persist DRAINING before anything else: S1/S2 stop selecting the pool.
        if pool.readiness not in (Readiness.DRAINING, Readiness.STOPPED):
            pool = self._publish(d, Readiness.DRAINING)
            self._observe(d, ObservedState.DRAINING)

        # 2. Cache invalidation barrier: no stop until S1 acknowledges.
        if self.s1 is not None:
            try:
                ack = self.s1.invalidate_pool(pool.id, pool.generation, reason)
            except S1Error as e:
                raise RetryLater(f"S1 invalidation failed: {e}") from e
            if not ack:
                raise RetryLater("S1 did not acknowledge pool invalidation; holding DRAINING")

        # 3. Let in-flight requests finish, up to the drain timeout.
        drained, remaining = self._wait_for_drain(ctx, pool.endpoint_url, drain_s)

        if d.resource_class == ResourceClass.ROUTING_ONLY:
            self._publish(d, Readiness.STOPPED)
            self._observe(d, ObservedState.STOPPED)
            return {"stopped": True, "routing_only": True, "reason": reason}

        # 4. Stop and verify.
        self._observe(d, ObservedState.STOPPING)
        verified, evidence = self._stop_and_verify(d)
        result: dict[str, Any] = {
            "reason": reason,
            "drained": drained,
            "in_flight_at_stop": remaining,
            "shutdown_verified": verified,
            "shutdown_evidence": evidence,
        }
        if not verified:
            self._observe(d, ObservedState.FAILED, "shutdown not verified: GPU must be quarantined")
            result["quarantine_required"] = True
            raise JobFailed("shutdown not verified", result)

        # 5. Persist proof, then STOPPED, then ask S1 to release.
        ctx.shutdown_verified_at = self.clock()
        self._publish(d, Readiness.STOPPED)
        self._observe(d, ObservedState.STOPPED)
        self._release(d, evidence, result)
        return result

    def _wait_for_drain(
        self, ctx: JobContext, endpoint: str, timeout_s: float
    ) -> tuple[bool, int | None]:
        """Wait until vLLM reports no running/waiting requests. If the count is unknown (no
        metrics), wait the whole timeout: unknown is not the same as idle."""
        deadline = self.clock() + timedelta(seconds=timeout_s)
        remaining = self.in_flight(endpoint)
        while remaining != 0 and self.clock() < deadline:
            ctx.heartbeat()
            self.sleep(min(2.0, self.settings.probe_interval_s))
            remaining = self.in_flight(endpoint)
        return (remaining == 0), remaining

    def enqueue_expiring(self) -> list[str]:
        """P0 serving-lease expiry, S3 side: start a drain early enough to finish before the
        allocation ends. S1's reconciler may also enqueue STOP_DEPLOY; both paths are idempotent."""
        if self.s1 is None:
            return []
        now = self.clock()
        started: list[str] = []
        for d in self.store.list_deployments():
            if (
                d.resource_class != ResourceClass.GPU_SERVING
                or d.desired_state != DesiredState.RUNNING
                or d.observed_state in TERMINAL_STATES
                or not d.allocation_id
            ):
                continue
            try:
                a = self.s1.get_allocation(d.allocation_id)
            except S1Error:
                continue
            lead = timedelta(
                seconds=self.settings.default_drain_timeout_s + self.settings.expiry_margin_s
            )
            if a.end_at - lead <= now or a.state not in (
                AllocationState.COMMITTED,
                AllocationState.ACTIVE,
            ):
                self.request_drain(d.id, reason="SERVING_LEASE_EXPIRY")
                started.append(d.id)
        return started
