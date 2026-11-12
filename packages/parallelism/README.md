# asio-parallelism (S3 — parallelism planner, P1)

`POST /api/v1/parallelism/plan` returns a `ParallelismPlan` (contract §4.4): the TP/PP/DP layout,
the `gpu_constraints` to request from S1 (count, GPU model, min free memory — never GPU ids: S1
chooses the UUIDs), and a `deployment_template` that drops straight into a deployment's `config`
(`vllm_args` incl. `--max-num-seqs`, `image_profile`, `rated_capacity_rps`).

The planner in `core/` is the calibrated model measured on gpu-02 (8× A100-SXM4-80GB, vLLM 0.31):
memory model within 2% on pre-registered tests, held-out TPOT error 6.4%, and an end-to-end run that
met a 15 ms TPOT p95 SLA in 3/3 trials. Predictions are labelled `SIMULATED`; only benchmarks on S1
`EXPERIMENT` allocations become `REAL_GPU`.

| Route | Status |
|---|---|
| `POST /api/v1/parallelism/plan` | works (422 `GPU_CONFIGURATION_INFEASIBLE` / `MODEL_UNKNOWN`) |
| `GET /api/v1/parallelism/plans/{id}` | works (in memory; S3 assumption so deployments can cite a plan id) |
| `POST /api/v1/parallelism/benchmarks`, `GET .../{id}` | 501 stub — P1, Days 4–7 |

`moe_placement_plan_id` is recorded with a warning; expert parallelism is not in the performance
model yet (CT-15 with S4).

```bash
cd packages/parallelism
uv sync && uv run pytest -q
uv run uvicorn asio_parallelism.api:app --port 8004
```
