from __future__ import annotations

import threading
import uuid
from collections import Counter
from datetime import timedelta

import pytest

from asio_deployment.contracts import Job, JobKind, JobState
from asio_deployment.runner import JobContext, JobFailed, JobRunner, RetryLater
from asio_deployment.store import MemoryStore, Store

from .conftest import PG_DSN, T0, FakeClock, _pg_store


def _job(target: str = "ws-1", kind: JobKind = JobKind.START_WORKSPACE, now=T0, **kw) -> Job:
    return Job(
        id=f"job_{uuid.uuid4().hex[:10]}",
        kind=kind,
        owner_module="workspaces",
        request_ref=kw.pop("request_ref", uuid.uuid4().hex),
        target_ref=target,
        run_after=now,
        created_at=now,
        updated_at=now,
        **kw,
    )


def test_enqueue_is_idempotent_per_module_and_request_ref(store: Store) -> None:
    a = store.enqueue_job(_job(request_ref="r1"))
    b = store.enqueue_job(_job(request_ref="r1"))
    assert a.id == b.id and len(store.list_jobs()) == 1


def test_s5_style_handler_success_and_result(store: Store, clock: FakeClock) -> None:
    store.enqueue_job(_job())
    r = JobRunner(store, "w", clock=clock)

    def start_workspace(ctx: JobContext) -> dict:
        ctx.heartbeat()
        return {"url": "https://ws.example/abc"}

    r.register(JobKind.START_WORKSPACE, start_workspace)
    job = r.run_once()
    assert job is not None
    done = store.get_job(job.id)
    assert (
        done
        and done.state == JobState.SUCCEEDED
        and done.result == {"url": "https://ws.example/abc"}
    )
    assert done.lease_until is None and r.run_once() is None


def test_retry_with_backoff_then_fail_after_max_attempts(store: Store, clock: FakeClock) -> None:
    store.enqueue_job(_job(max_attempts=3))
    r = JobRunner(store, "w", clock=clock, backoff=lambda a: 10.0)
    r.register(
        JobKind.START_WORKSPACE, lambda ctx: (_ for _ in ()).throw(RetryLater("docker busy"))
    )
    r.run_once()
    j = store.list_jobs()[0]
    assert j.state == JobState.QUEUED and j.attempt == 1 and j.last_error == "docker busy"
    assert r.run_once() is None  # backoff not elapsed
    for _ in range(2):
        clock.advance(11)
        r.run_once()
    j = store.list_jobs()[0]
    assert j.state == JobState.FAILED and j.attempt == 3


def test_job_failed_records_result_and_shutdown_proof(store: Store, clock: FakeClock) -> None:
    store.enqueue_job(_job(kind=JobKind.STOP_WORKSPACE))
    r = JobRunner(store, "w", clock=clock)

    def stop(ctx: JobContext) -> dict:
        ctx.shutdown_verified_at = clock()
        raise JobFailed("files not preserved", {"shutdown_verified": True})

    r.register(JobKind.STOP_WORKSPACE, stop)
    r.run_once()
    j = store.list_jobs()[0]
    assert j.state == JobState.FAILED and j.result == {"shutdown_verified": True}
    assert j.shutdown_verified_at == clock()


def test_expired_lease_is_taken_over_and_old_worker_is_fenced(
    store: Store, clock: FakeClock
) -> None:
    store.enqueue_job(_job())
    a = store.claim_job([JobKind.START_WORKSPACE], "worker-a", 30, clock())
    assert (
        a is not None
        and store.claim_job([JobKind.START_WORKSPACE], "worker-b", 30, clock()) is None
    )
    clock.advance(31)
    b = store.claim_job([JobKind.START_WORKSPACE], "worker-b", 30, clock())
    assert b is not None and b.id == a.id and b.generation == a.generation + 1 and b.attempt == 2
    assert not store.extend_lease(a, 30, clock())
    assert not store.finish_job(a.model_copy(update={"state": JobState.SUCCEEDED}))
    assert store.finish_job(b.model_copy(update={"state": JobState.SUCCEEDED}))
    assert store.get_job(a.id).state == JobState.SUCCEEDED  # type: ignore[union-attr]


def test_second_job_on_same_target_waits_without_spending_an_attempt(
    store: Store, clock: FakeClock
) -> None:
    first = store.enqueue_job(_job(target="dep-1"))
    clock.advance(1)
    store.enqueue_job(_job(target="dep-1", now=clock()))
    held = store.claim_job([JobKind.START_WORKSPACE], "other-worker", 60, clock())
    assert held is not None and held.id == first.id
    r = JobRunner(store, "w", clock=clock)
    ran: list[str] = []
    r.register(JobKind.START_WORKSPACE, lambda ctx: ran.append(ctx.job.id) or {})
    r.run_once()
    second = [j for j in store.list_jobs() if j.id != first.id][0]
    assert ran == [] and second.state == JobState.QUEUED and second.attempt == 0
    assert "waiting for job" in (second.last_error or "")


def test_lease_expired_too_often_ends_failed(store: Store, clock: FakeClock) -> None:
    store.enqueue_job(_job(max_attempts=1))
    store.claim_job([JobKind.START_WORKSPACE], "dead-worker", 10, clock())
    clock.advance(11)
    r = JobRunner(store, "w", clock=clock)
    r.register(JobKind.START_WORKSPACE, lambda ctx: {})
    r.run_once()
    j = store.list_jobs()[0]
    assert j.state == JobState.FAILED and j.result and j.result["max_attempts_exceeded"]


def test_unexpected_exception_is_retried(store: Store, clock: FakeClock) -> None:
    store.enqueue_job(_job())
    r = JobRunner(store, "w", clock=clock, backoff=lambda a: 0.0)
    calls = Counter[str]()

    def flaky(ctx: JobContext) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return {"ok": True}

    r.register(JobKind.START_WORKSPACE, flaky)
    r.run_once()
    r.run_once()
    assert store.list_jobs()[0].state == JobState.SUCCEEDED


@pytest.mark.skipif(not PG_DSN, reason="ASIO_TEST_PG_DSN not set")
def test_postgres_skip_locked_runs_each_job_exactly_once_under_concurrency() -> None:
    store = _pg_store()
    for i in range(40):
        store.enqueue_job(_job(target=f"ws-{i}"))
    seen: Counter[str] = Counter()
    lock = threading.Lock()

    def handler(ctx: JobContext) -> dict:
        with lock:
            seen[ctx.job.id] += 1
        return {}

    def work(wid: str) -> None:
        r = JobRunner(store, wid)
        r.register(JobKind.START_WORKSPACE, handler)
        while r.run_once() is not None:
            pass

    threads = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 40 and set(seen.values()) == {1}
    assert {j.state for j in store.list_jobs()} == {JobState.SUCCEEDED}


def test_memory_store_claims_in_run_after_order(clock: FakeClock) -> None:
    s = MemoryStore()
    late = s.enqueue_job(_job(now=clock() + timedelta(seconds=5)))
    early = s.enqueue_job(_job())
    assert s.claim_job([JobKind.START_WORKSPACE], "w", 10, clock()).id == early.id  # type: ignore[union-attr]
    clock.advance(6)
    assert s.claim_job([JobKind.START_WORKSPACE], "w", 10, clock()).id == late.id  # type: ignore[union-attr]
