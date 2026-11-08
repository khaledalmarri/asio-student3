# Step 3: benchmark comparison (2026-11-08, gpu-02)

> **Superseded at high concurrency.** These runs used one client process, which capped throughput at ~23k tok/s and
> inflated TPOT p95 at 128–256 (see `step3b-collapse.md`, Experiment E). Low-concurrency rows (≤ 32) stand.

Qwen/Qwen2.5-0.5B-Instruct, BF16, vLLM 0.31.0+cu126.abujafar1, `--gpu-memory-utilization 0.25 --max-model-len 4096`,
GPUs 0 (TP1) and 0,1 (others), both shared with idle tenants. Workload: 512-token random-ID prompts, 128 output tokens
(`ignore_eos`), 4× concurrency requests per level (min 16). All 24 levels: 0 failed requests.
Raw rows: `results.jsonl`. Produced with `run_config.sh` + `bench.py`.

## Output throughput (tok/s)

| Concurrency | qwen0.5b-tp1 | qwen0.5b-tp2 | qwen0.5b-pp2 | qwen0.5b-dp2 |
|---|---|---|---|---|
| 1 | 486 | 498 | 345 | 498 |
| 8 | 3,132 | 3,131 | 1,509 | 3,280 |
| 32 | 8,374 | 8,439 | 5,225 | 10,709 |
| 64 | 12,936 | 13,244 | 9,472 | 16,779 |
| 128 | 15,364 | 17,162 | 15,579 | 20,842 |
| 256 | 12,272 | 13,780 | 17,127 | 23,419 |

## TPOT p95 (ms)

| Concurrency | qwen0.5b-tp1 | qwen0.5b-tp2 | qwen0.5b-pp2 | qwen0.5b-dp2 |
|---|---|---|---|---|
| 1 | 2.0 | 1.99 | 3.0 | 1.93 |
| 8 | 2.28 | 2.44 | 5.12 | 2.37 |
| 32 | 3.19 | 3.4 | 6.48 | 2.74 |
| 64 | 4.27 | 4.15 | 6.6 | 3.43 |
| 128 | 6.97 | 6.37 | 7.27 | 4.82 |
| 256 | 17.21 | 16.33 | 14.94 | 6.25 |

## TTFT p95 (ms)

| Concurrency | qwen0.5b-tp1 | qwen0.5b-tp2 | qwen0.5b-pp2 | qwen0.5b-dp2 |
|---|---|---|---|---|
| 1 | 18.8 | 14.92 | 21.13 | 18.04 |
| 8 | 51.3 | 54.69 | 48.89 | 33.02 |
| 32 | 194.37 | 160.05 | 185.84 | 71.66 |
| 64 | 235.25 | 233.64 | 133.84 | 139.13 |
| 128 | 463.56 | 417.17 | 248.46 | 242.26 |
| 256 | 1117.05 | 878.41 | 627.03 | 322.52 |

## SLA view: TPOT p95 ≤ 5 ms

| Config | GPUs | Max concurrency with TPOT p95 ≤ 5 ms | SLA-compliant tok/s | per GPU | Peak tok/s (at conc) |
|---|---|---|---|---|---|
| qwen0.5b-tp1 | 1 | 64 (4.27 ms) | 12,936 | 12,936 | 15,364 (128) |
| qwen0.5b-tp2 | 2 | 64 (4.15 ms) | 13,244 | 6,622 | 17,162 (128) |
| qwen0.5b-pp2 | 2 | 1 (3.0 ms) | 345 | 172 | 17,127 (256) |
| qwen0.5b-dp2 | 2 | 128 (4.82 ms) | 20,842 | 10,421 | 23,419 (256) |

## Findings

1. DP=2 is best under load (highest SLA-compliant throughput); at ≤ 8 concurrent requests it equals TP=1, so the
   second GPU only pays off above ~64–128 requests per copy.
2. TP=1 is the most efficient per GPU. On a shared node, "use fewer GPUs" is a valid planner answer.
3. TP=2 adds ~12% peak throughput for 2× GPUs: not worth it for a 0.5B model (per-step overhead dominates).
4. PP=2 halves throughput at low concurrency (TPOT ≥ 5 ms from 8 requests) and fails a 5 ms TPOT SLA at any
   useful load, but it is the only single-copy config that does not lose throughput at 256.
5. TP=1 and TP=2 lose throughput above 128 concurrent requests (TPOT p95 jumps to 16–17 ms). The client is not
   the cause (same client got 23k tok/s from DP=2); CUDA graphs cover batch 512. Under investigation (Step 3b).
6. DP=2 reaches 76% of 2× TP=1 peak (23,419 vs 30,728): suspected single API server bottleneck (Step 3b, D).

These rules are for a small, overhead-bound model. Llama-3.1-8B on the same node (LoadCast Exp 9, vLLM 0.8.5)
gained 1.53× per-token speed from TP=2, so the crossover depends on model size (Step 4).
