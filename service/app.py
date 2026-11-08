"""Step 7: Student 3's API service (`/parallelism/*`), following the PROPOSED contract in docs/api-contract.md.

Run on the node (from the repo root, inside the venv):
    uv pip install fastapi uvicorn
    ASIO_TOKEN=<secret> uvicorn service.app:app --host 127.0.0.1 --port 8300

Endpoints
    GET  /health
    POST /parallelism/plan                      plan only (no side effects)
    POST /parallelism/benchmark                 start a benchmark job against a running endpoint -> job id
    GET  /parallelism/benchmark/{job_id}        job status and result rows
    POST /parallelism/deploy                    plan + deployment spec; "launch": true starts it (background job)
    GET  /parallelism/deployments               deployment log
    DELETE /parallelism/deployments/{name}      stop a deployment started by this service
    GET  /parallelism/capabilities              capability matrix (declared vs validated)
    GET  /parallelism/topology                  live GPU topology and free GPUs
    GET  /metrics                               Prometheus text format (for Student 6)

Anything that starts processes or containers needs the X-ASIO-Token header when ASIO_TOKEN is set:
on a node where every user is root, an open endpoint that launches containers would be an easy way to
take someone else's GPUs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from planner import capabilities, deploy, topology
from planner.plan import PlanRequest, plan

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "results" / "jobs"
app = FastAPI(title="ASIO Student 3 - Parallelism & Serving Configuration Optimizer", version="0.1")

_jobs: dict[str, dict] = {}
_bench_lock = threading.Lock()
_metrics = {"plans_total": 0, "plan_seconds_sum": 0.0, "benchmarks_total": 0, "benchmark_running": 0,
            "deployments_launched_total": 0, "deployments_failed_total": 0}
_owned: set[str] = set()


def _auth(token: str | None) -> None:
    want = os.environ.get("ASIO_TOKEN")
    if want and token != want:
        raise HTTPException(401, "missing or wrong X-ASIO-Token")


# ---------------- request models (field names = proposed contract) ----------------
class Workload(BaseModel):
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    max_model_len: int = 4096


class SLA(BaseModel):
    tpot_p95_ms: float = Field(gt=0)


class Allocation(BaseModel):
    gpu_ids: list[int] = Field(min_length=1)
    gpu_mem_util: float = Field(0.9, gt=0, le=0.95)


class Runtime(BaseModel):
    engine: str = "vllm"
    version: str = "0.31"


class PlanBody(BaseModel):
    model: str
    workload: Workload
    sla: SLA
    allocation: Allocation
    target_concurrency: int | None = Field(None, gt=0)
    runtime: Runtime = Runtime()


class BenchBody(BaseModel):
    url: str = "http://localhost:8401"
    model: str
    label: str
    prompt_tokens: int = 512
    output_tokens: int = 128
    concurrency: list[int] = [1, 8, 32, 64, 128]
    procs: int = 4


class DeployBody(PlanBody):
    name: str = "student3-deploy"
    port: int = 8401
    profile: str = deploy.DEFAULT_PROFILE
    model_path: str | None = None
    launch: bool = False


def _to_request(b: PlanBody) -> PlanRequest:
    if b.runtime.engine != "vllm":
        raise HTTPException(422, f"engine {b.runtime.engine!r} not supported (only vllm)")
    return PlanRequest(model=b.model, prompt_tokens=b.workload.prompt_tokens, output_tokens=b.workload.output_tokens,
                       tpot_p95_ms=b.sla.tpot_p95_ms, gpus=b.allocation.gpu_ids,
                       target_concurrency=b.target_concurrency, max_model_len=b.workload.max_model_len,
                       gpu_mem_util=b.allocation.gpu_mem_util, vllm_version=b.runtime.version)


def _plan(b: PlanBody) -> dict:
    t0 = time.perf_counter()
    try:
        out = plan(_to_request(b))
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(422, f"cannot plan: {e}")
    _metrics["plans_total"] += 1
    _metrics["plan_seconds_sum"] += time.perf_counter() - t0
    return out


# ---------------- endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parallelism/plan")
def post_plan(body: PlanBody):
    return _plan(body)


@app.post("/parallelism/benchmark", status_code=202)
def post_benchmark(body: BenchBody, x_asio_token: str | None = Header(None)):
    _auth(x_asio_token)
    if not _bench_lock.acquire(blocking=False):
        raise HTTPException(409, "a benchmark is already running; results would interfere")
    job = uuid.uuid4().hex[:12]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    out = JOBS_DIR / f"{job}.jsonl"
    cmd = [sys.executable, str(ROOT / "bench.py"), "--url", body.url, "--model", body.model, "--label", body.label,
           "--prompt-tokens", str(body.prompt_tokens), "--output-tokens", str(body.output_tokens),
           "--procs", str(body.procs), "--out", str(out), "--concurrency", *map(str, body.concurrency)]
    _jobs[job] = {"id": job, "kind": "benchmark", "status": "running", "request": body.model_dump(), "rows": []}
    _metrics["benchmarks_total"] += 1
    _metrics["benchmark_running"] = 1

    def run():
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
            ok = p.returncode == 0 and all(r["failed"] == 0 for r in rows)
            _jobs[job].update(status="done" if ok else "failed", rows=rows, log=(p.stdout + p.stderr)[-4000:])
        finally:
            _metrics["benchmark_running"] = 0
            _bench_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job, "status": "running"}


@app.get("/parallelism/benchmark/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


@app.post("/parallelism/deploy")
def post_deploy(body: DeployBody, x_asio_token: str | None = Header(None)):
    out = _plan(body)
    if out["status"] != "ok":
        raise HTTPException(422, {"message": "no feasible plan", "explanation": out["explanation"]})
    if body.profile not in deploy.PROFILES:
        raise HTTPException(422, f"unknown image profile {body.profile!r}")
    prof = deploy.PROFILES[body.profile]
    try:
        path = body.model_path or deploy.resolve_snapshot(body.model)
    except FileNotFoundError as e:
        raise HTTPException(422, str(e))
    sp = deploy.spec(out, body.model, path, body.name, body.port, prof)
    if not body.launch:
        return {"status": "planned", "spec": sp}
    _auth(x_asio_token)
    try:
        problems = deploy.check_gpus_free(sp["gpu_ids"], body.allocation.gpu_mem_util)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise HTTPException(503, f"cannot check GPUs (nvidia-smi): {e}")
    if problems:
        raise HTTPException(409, {"message": "planned GPUs lack free memory", "problems": problems})
    job = uuid.uuid4().hex[:12]
    _jobs[job] = {"id": job, "kind": "deploy", "status": "starting", "spec": sp}

    def run():
        res = deploy.launch(sp["command"], body.name, body.port)
        _jobs[job].update(status=res["status"], result=res)
        key = "deployments_launched_total" if res["status"] == "healthy" else "deployments_failed_total"
        _metrics[key] += 1
        if res["status"] == "healthy":
            _owned.add(body.name)
        deploy.record({"name": body.name, "model": body.model, "layout": sp["layout"], "gpu_ids": sp["gpu_ids"],
                       "max_num_seqs": sp["max_concurrency_per_engine"], "result": res["status"],
                       "startup_s": res.get("seconds"), "via": "api"})

    threading.Thread(target=run, daemon=True).start()
    return {"status": "starting", "job_id": job, "spec": sp}


@app.get("/parallelism/deployments")
def get_deployments():
    p = ROOT / "results" / "deployments.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


@app.delete("/parallelism/deployments/{name}")
def delete_deployment(name: str, x_asio_token: str | None = Header(None)):
    _auth(x_asio_token)
    if name not in _owned:
        raise HTTPException(403, "this service only stops deployments it started")
    deploy.stop(name)
    _owned.discard(name)
    return {"status": "stopped", "name": name}


@app.get("/parallelism/capabilities")
def get_capabilities():
    return capabilities.build(str(ROOT / "results" / "results.jsonl"))


@app.get("/parallelism/topology")
def get_topology(min_free_mib: int = 0):
    try:
        t = topology.discover()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise HTTPException(503, f"nvidia-smi unavailable: {e}")
    if min_free_mib:
        t["free_gpus"] = topology.free_gpus(t, min_free_mib)
    return t


@app.get("/metrics")
def metrics():
    from fastapi.responses import PlainTextResponse
    lines = []
    for k, v in _metrics.items():
        kind = "gauge" if k == "benchmark_running" else "counter"
        lines += [f"# TYPE asio_parallelism_{k} {kind}", f"asio_parallelism_{k} {v}"]
    return PlainTextResponse("\n".join(lines) + "\n")
