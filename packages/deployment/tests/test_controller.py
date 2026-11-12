from __future__ import annotations

from datetime import timedelta

import pytest

from asio_deployment.contracts import (
    AllocationKind,
    AllocationState,
    Backend,
    DeploymentConfig,
    JobKind,
    JobState,
    ObservedState,
    Readiness,
    ResourceClass,
)
from asio_deployment.controller import InvalidTransition, StaleGeneration
from asio_deployment.errors import ApiError

from .conftest import UUIDS, Env, make_request


def _pool(env: Env, dep_id: str):
    d = env.store.get_deployment(dep_id)
    assert d is not None
    p = env.store.get_pool(d.pool_id)
    assert p is not None
    return d, p


def _jobs(env: Env, dep_id: str, kind: JobKind):
    return [j for j in env.store.list_jobs(dep_id) if j.kind == kind]


# ------------------------------------------------------------------ create / idempotency


def test_create_persists_starting_pool_and_deploy_job_without_touching_docker(env: Env) -> None:
    d, job, created = env.controller.create("admin-1", make_request())
    assert created and job.kind == JobKind.DEPLOY and job.state == JobState.QUEUED
    d, p = _pool(env, d.id)
    assert d.observed_state == ObservedState.PENDING
    assert p.readiness == Readiness.STARTING
    assert p.endpoint_url == "http://10.0.0.2:8001"
    assert p.rated_capacity_rps == 12.5
    assert env.runtime.started == []


def test_same_request_ref_and_payload_is_a_replay(env: Env) -> None:
    d1, j1, c1 = env.controller.create("admin-1", make_request())
    d2, j2, c2 = env.controller.create("admin-1", make_request())
    assert (d1.id, j1.id, c1, c2) == (d2.id, j2.id, True, False)
    assert len(env.store.list_deployments()) == 1


def test_request_ref_is_caller_scoped(env: Env) -> None:
    d1, _, _ = env.controller.create("admin-1", make_request())
    d2, _, _ = env.controller.create("admin-2", make_request())
    assert d1.id != d2.id


def test_same_request_ref_different_payload_is_409(env: Env) -> None:
    env.controller.create("admin-1", make_request())
    with pytest.raises(ApiError) as e:
        env.controller.create("admin-1", make_request(model_id="other-model"))
    assert (e.value.status_code, e.value.code) == (409, "IDEMPOTENCY_CONFLICT")


@pytest.mark.parametrize(
    "over,code",
    [
        ({"allocation_id": None}, "ALLOCATION_REQUIRED"),
        ({"backend": Backend.DYNAMO}, "BACKEND_NOT_SUPPORTED"),
        ({"allocation_id": "missing"}, "ALLOCATION_UNAVAILABLE"),
    ],
)
def test_create_rejects_invalid_gpu_serving(env: Env, over: dict, code: str) -> None:
    with pytest.raises(ApiError) as e:
        env.controller.create("admin-1", make_request(**over))
    assert e.value.code == code


def test_create_rejects_non_serving_allocation(env: Env) -> None:
    env.s1.allocations["alloc-1"].kind = AllocationKind.EXPERIMENT
    with pytest.raises(ApiError) as e:
        env.controller.create("admin-1", make_request())
    assert (e.value.status_code, e.value.code) == (422, "ALLOCATION_INVALID")


# ------------------------------------------------------------------ deploy


def test_deploy_reaches_ready_only_after_model_probe(env: Env) -> None:
    d, _, _ = env.controller.create("admin-1", make_request())
    env.drain_jobs()
    d, p = _pool(env, d.id)
    assert p.readiness == Readiness.READY and d.observed_state == ObservedState.READY
    assert env.prober.calls >= 2 and p.health_checked_at is not None
    assert env.published == [Readiness.READY]
    assert d.gpu_uuids == UUIDS
    spec = env.runtime.started[0]
    assert spec.gpu_uuids == tuple(UUIDS) and spec.name == f"asio-s3-{d.id}"
    assert env.s1.names().count("activate") == 1
    assert env.s1.allocations["alloc-1"].state == AllocationState.ACTIVE
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert job.state == JobState.SUCCEEDED and job.result and job.result["ready"] is True


def test_ct05_crash_during_startup_never_ready_and_releases_after_verified_cleanup(
    env: Env,
) -> None:
    env.runtime.crash_after_start = True
    d, _, _ = env.controller.create("admin-1", make_request())
    env.drain_jobs()
    d, p = _pool(env, d.id)
    assert Readiness.READY not in env.published
    assert p.readiness == Readiness.UNHEALTHY and d.observed_state == ObservedState.FAILED
    assert "CUDA out of memory" in (d.last_error or "")
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert job.state == JobState.FAILED and job.shutdown_verified_at is not None
    assert job.result and job.result["shutdown_verified"] is True
    assert "activate" not in env.s1.names()
    assert env.s1.names()[-1] == "release"


def test_ct05_unverified_cleanup_keeps_gpu_and_asks_for_quarantine(env: Env) -> None:
    env.runtime.crash_after_start = True
    env.runtime.leave_pids = True
    d, _, _ = env.controller.create("admin-1", make_request())
    env.drain_jobs()
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert job.state == JobState.FAILED and job.shutdown_verified_at is None
    assert job.result and job.result["quarantine_required"] is True
    assert "release" not in env.s1.names()
    assert Readiness.READY not in env.published


def test_model_never_healthy_times_out_as_failed(env: Env) -> None:
    env.prober.ready_after = None
    d, _, _ = env.controller.create("admin-1", make_request())
    env.drain_jobs()
    d, p = _pool(env, d.id)
    assert p.readiness == Readiness.UNHEALTHY and d.observed_state == ObservedState.FAILED
    assert "not healthy within" in (d.last_error or "")
    assert Readiness.READY not in env.published


def test_layout_that_does_not_match_granted_gpus_fails_without_launch(env: Env) -> None:
    cfg = make_request().config.model_copy(update={"vllm_args": ["--tensor-parallel-size", "4"]})
    d, _, _ = env.controller.create("admin-1", make_request(config=cfg))
    env.drain_jobs()
    d, p = _pool(env, d.id)
    assert env.runtime.started == [] and p.readiness == Readiness.UNHEALTHY
    assert "needs 4 GPU(s)" in (d.last_error or "")


def test_allocation_expired_before_launch_fails_permanently(env: Env) -> None:
    d, _, _ = env.controller.create("admin-1", make_request())
    env.clock.advance(3 * 3600)
    env.drain_jobs()
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert job.state == JobState.FAILED and "ended" in (job.last_error or "")
    assert env.runtime.started == []


# ------------------------------------------------------------------ stop / drain


def _ready(env: Env) -> str:
    d, _, _ = env.controller.create("admin-1", make_request())
    env.drain_jobs()
    return d.id


def test_stop_order_draining_then_invalidate_then_stop_then_release(env: Env) -> None:
    dep = _ready(env)
    env.s1.calls.clear()
    env.controller.request_stop(dep)
    env.drain_jobs()
    d, p = _pool(env, dep)
    assert env.published[-2:] == [Readiness.DRAINING, Readiness.STOPPED]
    assert env.s1.names() == ["invalidate", "release"]
    inv = env.s1.calls[0][1]
    assert inv[0] == p.id and inv[1] == p.generation - 1  # generation of the DRAINING write
    assert p.readiness == Readiness.STOPPED and d.observed_state == ObservedState.STOPPED
    assert not env.runtime.exists(f"asio-s3-{dep}")
    job = _jobs(env, dep, JobKind.STOP_DEPLOY)[0]
    assert job.state == JobState.SUCCEEDED and job.shutdown_verified_at is not None
    assert job.result and job.result["shutdown_evidence"]["container_removed"] is True


def test_no_invalidation_ack_holds_draining_and_does_not_stop(env: Env) -> None:
    dep = _ready(env)
    env.s1.ack_invalidation = False
    env.controller.request_stop(dep)
    env.runner.run_once()
    d, p = _pool(env, dep)
    job = _jobs(env, dep, JobKind.STOP_DEPLOY)[0]
    assert p.readiness == Readiness.DRAINING and job.state == JobState.QUEUED
    assert "acknowledge" in (job.last_error or "")
    assert env.runtime.is_running(f"asio-s3-{dep}")
    env.s1.ack_invalidation = True
    env.clock.advance(5)
    env.drain_jobs()
    assert _pool(env, dep)[1].readiness == Readiness.STOPPED


def test_drain_waits_for_in_flight_requests(env: Env) -> None:
    dep = _ready(env)
    env.inflight.values = [3, 2, 1, 0]
    t0 = env.clock.now
    env.controller.request_drain(dep, drain_timeout_s=60)
    env.drain_jobs()
    job = _jobs(env, dep, JobKind.STOP_DEPLOY)[0]
    assert job.result and job.result["drained"] is True and job.result["in_flight_at_stop"] == 0
    assert env.clock.now - t0 >= timedelta(seconds=4)


def test_drain_with_unknown_in_flight_waits_full_timeout(env: Env) -> None:
    dep = _ready(env)
    env.inflight.values = [None]
    t0 = env.clock.now
    env.controller.request_drain(dep, drain_timeout_s=20)
    env.drain_jobs()
    assert env.clock.now - t0 >= timedelta(seconds=20)
    assert _pool(env, dep)[1].readiness == Readiness.STOPPED


def test_stop_without_verified_shutdown_does_not_release(env: Env) -> None:
    dep = _ready(env)
    env.runtime.leave_pids = True
    env.controller.request_stop(dep)
    env.drain_jobs()
    d, p = _pool(env, dep)
    job = _jobs(env, dep, JobKind.STOP_DEPLOY)[0]
    assert job.state == JobState.FAILED and job.result and job.result["quarantine_required"]
    assert p.readiness == Readiness.DRAINING and d.observed_state == ObservedState.FAILED
    assert "release" not in env.s1.names()


def test_stop_is_idempotent(env: Env) -> None:
    dep = _ready(env)
    _, j1 = env.controller.request_stop(dep)
    _, j2 = env.controller.request_stop(dep)
    assert j1.id == j2.id
    env.drain_jobs()
    _, j3 = env.controller.request_stop(dep)
    assert j3.id == j1.id and env.s1.names().count("release") == 1


def test_stop_requested_during_startup_aborts_launch_and_cleans_up(env: Env) -> None:
    env.prober.ready_after = None
    d, _, _ = env.controller.create("admin-1", make_request())
    orig = env.prober.__call__

    def probe_then_stop(endpoint: str, model: str):
        if env.prober.calls == 1:
            env.controller.request_stop(d.id)
        return orig(endpoint, model)

    env.controller.prober = probe_then_stop
    env.drain_jobs()
    deploy = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert deploy.result and deploy.result.get("aborted") is True
    assert Readiness.READY not in env.published
    d2, p = _pool(env, d.id)
    assert p.readiness == Readiness.STOPPED and d2.observed_state == ObservedState.STOPPED


def test_serving_lease_expiry_drains_and_stops_before_end(env: Env) -> None:
    dep = _ready(env)
    assert env.controller.enqueue_expiring() == []
    env.clock.advance(2 * 3600 - 80)  # inside drain (30 s) + margin (60 s) of end_at
    assert env.controller.enqueue_expiring() == [dep]
    assert env.controller.enqueue_expiring() == []  # desired_state is STOPPED now
    env.drain_jobs()
    job = _jobs(env, dep, JobKind.STOP_DEPLOY)[0]
    assert job.payload["reason"] == "SERVING_LEASE_EXPIRY"
    assert job.state == JobState.SUCCEEDED
    assert env.clock.now < env.s1.allocations["alloc-1"].end_at
    assert env.s1.allocations["alloc-1"].state == AllocationState.RELEASED


# ------------------------------------------------------------------ single writer


def test_publish_rejects_stale_generation_and_invalid_transition(env: Env) -> None:
    dep = _ready(env)
    _, p = _pool(env, dep)
    with pytest.raises(StaleGeneration):
        env.controller.publish_pool_state(dep, Readiness.DRAINING, p.generation)
    env.controller.publish_pool_state(dep, Readiness.DRAINING, p.generation + 1)
    with pytest.raises(InvalidTransition):
        env.controller.publish_pool_state(dep, Readiness.READY, p.generation + 2)


def test_routing_only_entry_point_needs_no_allocation_or_container(env: Env) -> None:
    req = make_request(
        "switchyard-1",
        resource_class=ResourceClass.ROUTING_ONLY,
        allocation_id=None,
        backend=Backend.SWITCHYARD,
        config=DeploymentConfig(
            served_model_name="qwen-demo", port=9000, endpoint_url="http://10.0.0.9:9000"
        ),
    )
    d, _, _ = env.controller.create("admin-1", req)
    env.drain_jobs()  # first probe fails -> retried with backoff, still STARTING
    assert _pool(env, d.id)[1].readiness == Readiness.STARTING
    env.clock.advance(2)
    env.drain_jobs()
    d, p = _pool(env, d.id)
    assert p.readiness == Readiness.READY and p.allocation_id is None
    assert env.runtime.started == []
    env.controller.request_stop(d.id)
    env.drain_jobs()
    assert _pool(env, d.id)[1].readiness == Readiness.STOPPED


# ------------------------------------------------------------------ pre-launch GPU check


def test_busy_gpu_is_retried_then_fails_without_launching_or_releasing(env: Env) -> None:
    env.runtime.gpu_free_mib[UUIDS[1]] = 3000  # someone outside ASIO is using it
    d, _, _ = env.controller.create("admin-1", make_request())
    env.runner.run_once()
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert job.state == JobState.QUEUED and "GPU busy" in (job.last_error or "")
    assert env.store.get_deployment(d.id).observed_state == ObservedState.PENDING  # type: ignore[union-attr]
    for _ in range(job.max_attempts):
        env.clock.advance(2)
        env.runner.run_once()
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    d2, p = _pool(env, d.id)
    assert job.state == JobState.FAILED and job.result and job.result["gpu_busy"] is True
    assert job.result["launched"] is False and UUIDS[1] in job.result["problems"][0]
    assert env.runtime.started == [] and Readiness.READY not in env.published
    assert p.readiness == Readiness.UNHEALTHY and d2.observed_state == ObservedState.FAILED
    assert "activate" not in env.s1.names() and "release" not in env.s1.names()


def test_gpu_that_frees_up_on_retry_launches(env: Env) -> None:
    env.runtime.gpu_free_mib[UUIDS[0]] = 1000
    d, _, _ = env.controller.create("admin-1", make_request())
    env.runner.run_once()
    assert env.runtime.started == []
    env.runtime.gpu_free_mib[UUIDS[0]] = 80000
    env.clock.advance(2)
    env.drain_jobs()
    d2, p = _pool(env, d.id)
    assert p.readiness == Readiness.READY
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert job.result and job.result["gpu_check"]["gpus"][UUIDS[0]]["free_mib"] == 80000


def test_gpu_not_on_this_node_fails_at_once(env: Env) -> None:
    env.runtime.missing_gpus.add(UUIDS[0])
    d, _, _ = env.controller.create("admin-1", make_request())
    env.runner.run_once()
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert job.state == JobState.FAILED and job.attempt == 1
    assert job.result and job.result["gpu_busy"] is False
    assert "not on this node" in job.result["problems"][0]


def test_need_follows_vllm_memory_share(env: Env) -> None:
    # 0.3 x 81920 MiB + 1024 margin = 25600 MiB needed; 30000 free is enough, 20000 is not.
    cfg = make_request().config.model_copy(
        update={"vllm_args": ["--tensor-parallel-size", "2", "--gpu-memory-utilization", "0.3"]}
    )
    env.runtime.gpu_free_mib.update({UUIDS[0]: 30000, UUIDS[1]: 30000})
    env.controller.create("admin-1", make_request(config=cfg))
    env.drain_jobs()
    assert len(env.runtime.started) == 1
    env.runtime.gpu_free_mib.update({UUIDS[0]: 20000})
    env.controller.create("admin-1", make_request("second", config=cfg))
    env.runner.run_once()
    assert len(env.runtime.started) == 1


def test_exclusive_mode_refuses_a_gpu_with_any_process(env: Env) -> None:
    env.controller.settings.require_exclusive_gpus = True
    env.runtime.gpu_pids[UUIDS[0]] = [4242]
    d, _, _ = env.controller.create("admin-1", make_request())
    env.runner.run_once()
    job = _jobs(env, d.id, JobKind.DEPLOY)[0]
    assert job.state == JobState.QUEUED and "4242" in (job.last_error or "")
    assert env.runtime.started == []
