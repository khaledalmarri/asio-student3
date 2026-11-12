# asio-deployment (S3 — Deployment Controller)

ASIO V3.1, contract draft 0.4. S3 P0: the Deployment Controller, the **shared durable job runner**
(used by S3 deployments and S5 workspaces), GPU-pinned vLLM templates, and the **single writer of
`ServingPool.readiness`**.

| Contract item | Where |
|---|---|
| `POST /api/v1/deployments` (caller-scoped `request_ref`), `GET /{id}`, `POST /{id}/drain`, `/stop` | `api.py` |
| `Deployment`, `Job`, `ServingPool` (write side) | `contracts.py` (provisional until `packages/common`) |
| Durable runner: claim (`FOR UPDATE SKIP LOCKED`), lease, retry, fencing, verification | `runner.py`, `store.py` |
| `publish_pool_state(deployment_id, state, generation)` — the only readiness writer | `controller.py` |
| Privileged worker (only process with the Docker socket) | `worker.py` |
| GPU-pinned vLLM template, image allowlist, GPUs by **UUID** | `templates.py` |
| Verified shutdown (container gone + none of its PIDs on the granted GPU UUIDs) | `runtime.py` |
| Proposed `jobs` / `deployments` / `serving_pools` DDL for S1's migration | `sql/0001_s3_proposed.sql` |

## Lifecycles

**Deploy** (DEPLOY job): S1 allocation must be `SERVING`, `COMMITTED|ACTIVE`, inside its window →
`docker run --gpus "device=<granted UUIDs>"` → probe `/health` **and** `/v1/models` lists the served
model → S1 `activate` → publish `READY`. Any failure (start error, crash, timeout) publishes
`UNHEALTHY` — never `READY` (CT-05) — then stops and verifies; S1 `release` is called only with
verified shutdown evidence, otherwise the job result says `quarantine_required`.

**Drain / stop / P0 serving-lease expiry** (STOP_DEPLOY job): persist `DRAINING` → S1
`invalidate_pool(pool_id, minimum_generation, reason)` and **wait for the acknowledgment** (no ack =
job retried, pool held `DRAINING`, container untouched) → wait for vLLM's running+waiting requests to
reach 0 (unknown counts wait the full drain timeout) → stop → verify → persist
`shutdown_verified_at` → publish `STOPPED` → S1 `release`. The worker starts this drain for any
serving deployment whose allocation ends within `drain timeout + margin`; S1's reconciler may also
enqueue a STOP_DEPLOY — both paths are idempotent.

**Routing-only** (Switchyard / Dynamo frontend entry point): no allocation, no container; the entry
point is probed before `READY`. Entry points, not Dynamo workers, are registry records.

## Shared job runner for S5

```python
from asio_deployment.contracts import JobKind
from asio_deployment.runner import JobContext, JobFailed, JobRunner, RetryLater


def start_workspace(ctx: JobContext) -> dict:
    ...  # ctx.job.target_ref = workspace id, ctx.job.allocation_id = S1 grant
    ctx.heartbeat()  # during long work; raises LeaseLost if another worker took over
    return {"url": "..."}  # stored as job.result


def stop_workspace(ctx: JobContext) -> dict:
    ...  # stop + verify
    ctx.shutdown_verified_at = ctx.clock()  # only with proof; S1 releases only then
    return {"shutdown_evidence": {...}}


runner.register(JobKind.START_WORKSPACE, start_workspace)
runner.register(JobKind.STOP_WORKSPACE, stop_workspace)
```

Enqueue with `owner_module="workspaces"` and a unique `request_ref` (enqueue is idempotent on
`(owner_module, request_ref)`). `RetryLater` = transient (backoff, spends an attempt), `JobFailed` =
permanent with a result, anything else = retried. Only one job per `target_ref` runs at a time.

## Run

```bash
cd packages/deployment
uv sync
uv run pytest -q                       # in-memory store; add ASIO_TEST_PG_DSN to also run on PostgreSQL
ASIO_AUTH_MODE=development uv run uvicorn asio_deployment.app:app --port 8003
```

Day-2 GPU smoke test on the node (real Docker + GPU, S1 mocked, Qwen2.5-0.5B on the GPU with the
most free memory; it only ever creates/removes its own `asio-s3-*` container):

```bash
uv run python scripts/smoke_gpu.py --gpu-uuid auto
```

## Evidence (as of this PR)

| Check | Label | Result |
|---|---|---|
| `pytest` on the in-memory store and on PostgreSQL 16 (incl. 6 workers × 40 jobs, each run exactly once) | MOCKED runtime / S1 | 87 passed |
| `ruff check`, `ruff format --check`, `mypy asio_deployment` | offline | clean |
| `scripts/smoke_gpu.py` on gpu-02 | REAL_GPU runtime, MOCKED S1 | to be run (Day 2) |

## Open questions

**S1**
1. `jobs` columns beyond §2.2 (`max_attempts`, `run_after`, `worker_id`, `payload`, `last_error`) and
   `UNIQUE (owner_module, request_ref)` — OK for your migration?
2. `invalidate_pool` transport: in-process call (monolith) or HTTP? `HttpS1Client` assumes
   `POST /internal/v1/serving-pools/{id}/invalidate` → `{"acknowledged": true}`.
3. `release` body: S3 sends `{"reason": "S3_VERIFIED_SHUTDOWN", "shutdown_evidence": {...}}` with
   `Idempotency-Key: release:<deployment_id>`. Acceptable?
4. Quarantine: on unverified shutdown S3 ends the job `FAILED` with `result.quarantine_required=true`
   and no `shutdown_verified_at`, and does not release. Does your reconciler quarantine from that, or
   should S3 call an endpoint?
5. Expired ACTIVE serving lease: will your reconciler insert `STOP_DEPLOY` jobs directly (which
   `owner_module`)? S3 also starts the drain early on its own.
6. Auth: S3 reads `request.state.caller` (`id`, `role`); only `admin`/`system` may deploy.
7. Which node and which 4 GPU UUIDs (Day-1 decision, S1/S3)?

**S2** — served model names: `qwen-demo` (Qwen2.5-7B-Instruct) and `qwen-demo-small`
(Qwen2.5-0.5B-Instruct), matching your demo profiles. `rated_capacity_rps` comes from the planner
(SIMULATED) until S3's P1 benchmarks replace it.

**S4** — placement-plan fields S3 should merge into `ParallelismPlan`, and the EP launch args.

**S6** — job results carry `evidence_label`, GPU UUIDs, container and shutdown evidence: is that the
evidence record you want?

## Not in this PR

Dynamo GPU_SERVING backend (CT-18, P1), serving-lease renewal via `/allocations/{id}/extend`
(CT-16, P1), benchmark jobs (CT-07, P1), Prometheus counters (with S6).
