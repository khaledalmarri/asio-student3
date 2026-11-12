from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from asio_deployment.contracts import (
    Allocation,
    AllocationKind,
    AllocationState,
    Backend,
    DeploymentConfig,
    JobKind,
    Readiness,
    ResourceClass,
)
from asio_deployment.controller import (
    ControllerSettings,
    CreateDeploymentRequest,
    DeploymentController,
)
from asio_deployment.runner import JobRunner
from asio_deployment.runtime import FakeRuntime, ProbeResult
from asio_deployment.s1 import FakeS1
from asio_deployment.store import MemoryStore, PostgresStore, Store

PG_DSN = os.environ.get("ASIO_TEST_PG_DSN")
T0 = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
UUIDS = ["GPU-11111111-aaaa", "GPU-22222222-bbbb"]


class FakeClock:
    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += timedelta(seconds=s)

    def advance(self, s: float) -> None:
        self.now += timedelta(seconds=s)


@dataclass
class Prober:
    """Healthy after `ready_after` probes; never healthy if None."""

    ready_after: int | None = 2
    calls: int = 0

    def __call__(self, endpoint: str, model: str) -> ProbeResult:
        self.calls += 1
        if self.ready_after is not None and self.calls >= self.ready_after:
            return ProbeResult(True, "health ok; model listed")
        return ProbeResult(False, "/health -> connection refused")


@dataclass
class InFlight:
    values: list[int | None] = field(default_factory=lambda: [0])

    def __call__(self, endpoint: str) -> int | None:
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def _pg_store() -> Store:
    s = PostgresStore(PG_DSN or "")
    s.apply_schema()
    with s._conn() as c:
        c.execute("TRUNCATE jobs, deployments, serving_pools")
    return s


@pytest.fixture(params=["memory", "postgres"])
def store(request: pytest.FixtureRequest) -> Iterator[Store]:
    if request.param == "postgres":
        if not PG_DSN:
            pytest.skip("ASIO_TEST_PG_DSN not set")
        yield _pg_store()
    else:
        yield MemoryStore()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def s1(clock: FakeClock) -> FakeS1:
    f = FakeS1()
    f.allocations["alloc-1"] = Allocation(
        id="alloc-1",
        kind=AllocationKind.SERVING,
        owner_id="s3",
        gpu_uuids=list(UUIDS),
        node_id="gpu-02",
        start_at=clock.now,
        end_at=clock.now + timedelta(hours=2),
        state=AllocationState.COMMITTED,
    )
    return f


@dataclass
class Env:
    store: Store
    clock: FakeClock
    s1: FakeS1
    runtime: FakeRuntime
    prober: Prober
    inflight: InFlight
    controller: DeploymentController
    runner: JobRunner
    published: list[Readiness]

    def drain_jobs(self, max_steps: int = 50) -> None:
        for _ in range(max_steps):
            if self.runner.run_once() is None:
                return
        raise AssertionError("jobs did not settle")


@pytest.fixture
def env(store: Store, clock: FakeClock, s1: FakeS1) -> Env:
    runtime = FakeRuntime()
    prober = Prober()
    inflight = InFlight()
    ctl = DeploymentController(
        store,
        s1=s1,
        runtime=runtime,
        settings=ControllerSettings(
            advertise_host="10.0.0.2",
            start_timeout_s=60,
            probe_interval_s=5,
            default_drain_timeout_s=30,
        ),
        clock=clock,
        sleep=clock.sleep,
        prober=prober,
        in_flight_fn=inflight,
    )
    published: list[Readiness] = []
    orig = ctl.publish_pool_state

    def recording(deployment_id: str, state: Readiness, generation: int, **kw: object):
        p = orig(deployment_id, state, generation, **kw)
        published.append(state)
        return p

    ctl.publish_pool_state = recording  # type: ignore[method-assign]
    runner = JobRunner(store, "worker-a", lease_s=60, clock=clock, backoff=lambda a: 1.0)
    runner.register(JobKind.DEPLOY, ctl.handle_deploy)
    runner.register(JobKind.STOP_DEPLOY, ctl.handle_stop)
    return Env(store, clock, s1, runtime, prober, inflight, ctl, runner, published)


def make_request(ref: str = "s3-qwen-deployment-001", **over: object) -> CreateDeploymentRequest:
    body: dict[str, object] = dict(
        request_ref=ref,
        resource_class=ResourceClass.GPU_SERVING,
        allocation_id="alloc-1",
        model_id="qwen-demo",
        backend=Backend.VLLM,
        config=DeploymentConfig(
            served_model_name="qwen-demo",
            port=8001,
            model_path="/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/x",
            vllm_args=["--tensor-parallel-size", "2", "--max-num-seqs", "64"],
            rated_capacity_rps=12.5,
        ),
    )
    body.update(over)
    return CreateDeploymentRequest.model_validate(body)
