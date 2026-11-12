"""Day-2 Docker/GPU smoke test: the real worker path on one GPU, with S1 MOCKED.

    cd packages/deployment && uv sync
    uv run python scripts/smoke_gpu.py --gpu-uuid auto

What it does (evidence: runtime REAL_GPU, S1 MOCKED, store as configured):
  create deployment -> DEPLOY job -> docker run pinned to the GPU UUID -> /health + /v1/models
  -> READY -> one chat completion -> STOP job (DRAINING -> invalidate -> stop -> verify) -> STOPPED

It only creates and removes its own `asio-s3-*` container and never touches other containers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from asio_deployment.contracts import (
    Allocation,
    AllocationKind,
    AllocationState,
    Backend,
    DeploymentConfig,
    JobKind,
    ResourceClass,
)
from asio_deployment.controller import (
    ControllerSettings,
    CreateDeploymentRequest,
    DeploymentController,
)
from asio_deployment.runner import JobRunner
from asio_deployment.runtime import DockerRuntime
from asio_deployment.s1 import FakeS1
from asio_deployment.store import MemoryStore


def pick_gpu(min_free_mib: int) -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = [(int(f), u) for u, f in (x.split(", ") for x in out.strip().splitlines())]
    free, uuid = max(rows)
    if free < min_free_mib:
        raise SystemExit(f"no GPU has {min_free_mib} MiB free (best: {free} MiB on {uuid})")
    return uuid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-uuid", default="auto", help="GPU UUID, or 'auto' = most free memory")
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--served-name", default="qwen-demo-small")
    ap.add_argument("--port", type=int, default=8411)
    ap.add_argument("--mem-util", type=float, default=0.25)
    ap.add_argument("--out", default="s3-smoke-evidence.jsonl")
    a = ap.parse_args()

    gpu = pick_gpu(int(a.mem_util * 81920) + 2048) if a.gpu_uuid == "auto" else a.gpu_uuid
    now = datetime.now(UTC)
    s1 = FakeS1()
    s1.allocations["smoke-alloc"] = Allocation(
        id="smoke-alloc",
        kind=AllocationKind.SERVING,
        owner_id="s3-smoke",
        gpu_uuids=[gpu],
        node_id="gpu-02",
        start_at=now,
        end_at=now + timedelta(hours=1),
        state=AllocationState.COMMITTED,
    )
    store = MemoryStore()
    ctl = DeploymentController(
        store,
        s1=s1,
        runtime=DockerRuntime(),
        settings=ControllerSettings(advertise_host="127.0.0.1", start_timeout_s=600),
    )
    runner = JobRunner(store, "smoke-worker", lease_s=120)
    runner.register(JobKind.DEPLOY, ctl.handle_deploy)
    runner.register(JobKind.STOP_DEPLOY, ctl.handle_stop)

    req = CreateDeploymentRequest(
        request_ref=f"smoke-{int(time.time())}",
        resource_class=ResourceClass.GPU_SERVING,
        allocation_id="smoke-alloc",
        model_id=a.served_name,
        backend=Backend.VLLM,
        config=DeploymentConfig(
            served_model_name=a.served_name,
            port=a.port,
            hf_model=a.hf_model,
            vllm_args=[
                "--gpu-memory-utilization",
                str(a.mem_util),
                "--max-model-len",
                "4096",
                "--max-num-seqs",
                "64",
            ],
        ),
    )
    t0 = time.time()
    d, _, _ = ctl.create("s3-smoke", req)
    print(f"deployment {d.id} on {gpu}; launching ...", flush=True)
    while runner.run_once() is not None:
        pass
    d = ctl.get(d.id)
    pool = store.get_pool(d.pool_id)
    deploy_job = [j for j in store.list_jobs(d.id) if j.kind == JobKind.DEPLOY][0]
    record: dict = {
        "time": now.isoformat(timespec="seconds"),
        "gpu_uuid": gpu,
        "hf_model": a.hf_model,
        "evidence": {"runtime": "REAL_GPU", "s1": "MOCKED"},
        "deploy": {
            "state": deploy_job.state,
            "observed": d.observed_state,
            "readiness": pool.readiness if pool else None,
            "seconds": round(time.time() - t0),
            "result": deploy_job.result,
            "error": deploy_job.last_error,
        },
    }
    print(
        f"deploy -> {d.observed_state} / pool {record['deploy']['readiness']} "
        f"in {record['deploy']['seconds']} s",
        flush=True,
    )

    if pool and pool.readiness == "READY":
        r = httpx.post(
            f"{pool.endpoint_url}/v1/chat/completions",
            timeout=60,
            json={
                "model": a.served_name,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Say hello in five words."}],
            },
        )
        record["chat"] = {
            "status": r.status_code,
            "text": r.json()["choices"][0]["message"]["content"]
            if r.status_code == 200
            else r.text[:300],
        }
        print(f"chat completion -> {r.status_code}: {record['chat']['text']!r}", flush=True)

    ctl.request_drain(d.id, reason="SMOKE_TEST_DONE", drain_timeout_s=10)
    while runner.run_once() is not None:
        pass
    stop_job = [j for j in store.list_jobs(d.id) if j.kind == JobKind.STOP_DEPLOY][0]
    d = ctl.get(d.id)
    record["stop"] = {
        "state": stop_job.state,
        "observed": d.observed_state,
        "shutdown_verified_at": stop_job.shutdown_verified_at.isoformat()
        if stop_job.shutdown_verified_at
        else None,
        "result": stop_job.result,
    }
    record["s1_calls"] = s1.names()
    print(
        f"stop -> {d.observed_state}; verified={bool(stop_job.shutdown_verified_at)}; "
        f"S1 calls {s1.names()}",
        flush=True,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(f"evidence appended to {a.out}")
    ok = (
        record["deploy"]["readiness"] == "READY"
        and record.get("chat", {}).get("status") == 200
        and stop_job.shutdown_verified_at is not None
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
