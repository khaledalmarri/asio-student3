# feat(s3): Deployment Controller, shared job runner and /api/v1 deployment routes (contract 0.4)

## Summary

S3 (Parallelism & Deployment) implementation of contract draft 0.4, P0 first. Draft — some S3
assumptions need S1 confirmation (questions in `packages/deployment/README.md`).

All changes are inside `packages/deployment/` and `packages/parallelism/`. The API process never
touches Docker; only the separate worker does. S3 never allocates GPUs: it launches on S1-granted
GPU UUIDs only.

## What's included

- **Routes (§5.4):** `POST /api/v1/deployments` (202; caller-scoped `request_ref`; identical replay
  returns the same deployment, a different payload is `409 IDEMPOTENCY_CONFLICT`),
  `GET /api/v1/deployments/{id}`, `POST /{id}/drain`, `POST /{id}/stop`. Errors use the §3.1
  envelope with `X-Request-ID`.
- **Shared durable job runner (§2.2):** PostgreSQL claim with `FOR UPDATE SKIP LOCKED`, leases with
  heartbeat, `generation` fencing, retry with backoff, one job per target at a time, and
  `shutdown_verified_at` set only with proof. S5 registers `START_WORKSPACE` / `STOP_WORKSPACE`
  handlers on the same runner.
- **ServingPool single writer (§4.6):** `publish_pool_state(deployment_id, state, generation)` with
  allowed transitions and compare-and-set on `generation`. `READY` only after `/health` **and**
  `/v1/models` lists the served model.
- **Failure (CT-05):** a failed start is never `READY`. The GPU is released only after verified
  cleanup; otherwise the job reports `quarantine_required`.
- **Drain / stop / P0 serving-lease expiry:** `DRAINING` → S1 invalidation + acknowledgment → wait
  for in-flight requests → stop → verify → `STOPPED` → S1 release. The worker starts the drain before
  the allocation ends.
- **GPU-pinned vLLM template:** `--gpus "device=<UUIDs>"`, image allowlist, TP×PP×DP must equal the
  granted GPU count.
- **Planner (P1):** `POST /api/v1/parallelism/plan` returns a `ParallelismPlan` with
  `gpu_constraints` for S1 and a `deployment_template` (vLLM args, `--max-num-seqs`,
  `rated_capacity_rps` for S2's static ranking).
- **Proposed DDL** for S1's migration: `packages/deployment/asio_deployment/sql/0001_s3_proposed.sql`.

## Evidence

| Check | Label | Result |
|---|---|---|
| deployment `pytest` (in-memory + PostgreSQL 16, incl. 6 concurrent workers × 40 jobs, each run exactly once) | MOCKED runtime / S1 | 87 passed |
| parallelism `pytest` | SIMULATED | 6 passed |
| `ruff check`, `ruff format --check`, `mypy` (both packages) | offline | clean |
| `scripts/smoke_gpu.py` on gpu-02 (docker → READY → chat completion → verified stop) | REAL_GPU runtime, MOCKED S1 | see PR comment |

## How to run

```bash
cd packages/deployment && uv sync && uv run pytest -q
cd packages/parallelism && uv sync && uv run pytest -q
```

## Questions

- **S1:** jobs columns, the invalidation transport, the release body, quarantine signalling,
  reconciler STOP_DEPLOY jobs, caller context, and which 4 GPU UUIDs (details in the deployment
  README).
- **S2:** served model names `qwen-demo` / `qwen-demo-small`, matching your demo profiles.
- **S4:** the placement-plan schema.
- **S5:** the runner handler interface.
- **S6:** the evidence fields.

## PR checklist (contract §11)

- [x] Matches ASIO V3.1 approved ownership (S3 deploys and writes readiness; no GPU grants, no dispatch)
- [x] Shared schemas follow the contract (provisional copies pending `packages/common`)
- [x] Unit and contract tests pass
- [x] Authorization, idempotency and failure states handled (auth context from S1)
- [x] Privileged worker separated: allowed operations = `docker run/inspect/top/logs/stop/rm` on `asio-s3-*` containers + `nvidia-smi` queries
- [ ] Request IDs / metrics: `X-Request-ID` done; Prometheus counters to follow with S6
- [x] Real GPU vs simulated evidence clearly identified
- [x] README updated

## Rollback

Additive only: the PR creates `packages/deployment/` and `packages/parallelism/` content and touches
nothing else.
