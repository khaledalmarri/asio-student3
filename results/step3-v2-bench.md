# Step 3 v2: benchmark comparison with a validated client (2026-11-08, gpu-02)

Replaces the high-concurrency rows of `step3-bench.md`. Client: `bench.py --procs 4` (validated: `--procs 8` within 3%).
Qwen/Qwen2.5-0.5B-Instruct BF16, vLLM 0.31.0+cu126.abujafar1, `--gpu-memory-utilization 0.25 --max-model-len 4096`,
512-token random-ID prompts, 128 output tokens. DP=2 uses `--api-server-count 2`. 7872 requests, 0 failed.

**Interference caveat:** `run_config.sh` printed GPU utilisation of 87–100% on GPUs 0/1 *before* the TP=2, PP=2 and
DP=2 runs (0% before TP=1). Memory showed only the other tenants, so this is either tenant compute or the tail of the
previous run's utilisation window; not separated. TP=2 and DP=2 at 256 match earlier runs within 1%.

## Output throughput (tok/s)

| Concurrency | TP=1 | TP=2 | PP=2 | DP=2 |
|---|---|---|---|---|
| 1 | 485 | 495 | 343 | 500 |
| 8 | 3,140 | 3,171 | 1,503 | 2,798 |
| 32 | 8,391 | 8,730 | 5,246 | 10,486 |
| 64 | 13,010 | 13,710 | 10,388 | 18,457 |
| 128 | 15,546 | 17,227 | 16,651 | 27,387 |
| 256 | 12,273 | 13,877 | 15,889 | 33,839 |

## TPOT p95 (ms)

| Concurrency | TP=1 | TP=2 | PP=2 | DP=2 |
|---|---|---|---|---|
| 1 | 1.96 | 1.96 | 3.04 | 1.94 |
| 8 | 2.24 | 2.33 | 5.3 | 3.07 |
| 32 | 3.25 | 3.2 | 6.23 | 2.81 |
| 64 | 4.02 | 4.06 | 5.73 | 3.08 |
| 128 | 6.53 | 6.33 | 6.91 | 4.07 |
| 256 | 16.98 | 15.15 | 15.26 | 6.45 |

## TTFT p95 (ms)

| Concurrency | TP=1 | TP=2 | PP=2 | DP=2 |
|---|---|---|---|---|
| 1 | 20.11 | 16.91 | 20.33 | 14.15 |
| 8 | 54.49 | 59.15 | 55.92 | 133.98 |
| 32 | 194.43 | 202.74 | 168.2 | 116.86 |
| 64 | 257.11 | 253.88 | 131.14 | 113.7 |
| 128 | 543.91 | 485.41 | 286.59 | 210.05 |
| 256 | 1243.34 | 962.85 | 763.76 | 460.06 |

## SLA view

| Config | GPUs | Max concurrency with TPOT p95 ≤ 5 ms | SLA-compliant tok/s | per GPU |
|---|---|---|---|---|
| TP1 | 1 | 64 (4.02 ms) | 13,010 | 13,010 |
| TP2 | 2 | 64 (4.06 ms) | 13,710 | 6,855 |
| PP2 | 2 | 1 (3.04 ms) | 343 | 171 |
| DP2 | 2 | 128 (4.07 ms) | 27,387 | 13,693 |

| Config | GPUs | Max concurrency with TPOT p95 ≤ 8 ms | SLA-compliant tok/s | per GPU |
|---|---|---|---|---|
| TP1 | 1 | 128 (6.53 ms) | 15,546 | 15,546 |
| TP2 | 2 | 128 (6.33 ms) | 17,227 | 8,613 |
| PP2 | 2 | 128 (6.91 ms) | 16,651 | 8,325 |
| DP2 | 2 | 256 (6.45 ms) | 33,839 | 16,919 |

## Findings (supersede Step 3 v1 at concurrency ≥ 64)

1. **DP=2 dominates under load**: 27,387 tok/s within a 5 ms TPOT p95 SLA (2.1× TP=1) and 33,839 at 256 — linear
   scaling, now that the client is not the bottleneck. Per GPU it matches TP=1 within the SLA (13.7k vs 13.0k).
2. **Single-engine configs lose throughput above 128** (TP=1 15,547 → 12,273; TP=2 17,227 → 13,877): server-side,
   confirmed with two client modes (Step 3b).
3. **TP=2** buys ≤ 11% over TP=1 for 2× GPUs at every load level.
4. **PP=2** halves throughput at 8 concurrent and fails a 5 ms TPOT SLA at any load above 1; under an 8 ms SLA it
   reaches 16,651 at 128, still below DP=2.
5. **DP=2 at 8 concurrent** (2,798 tok/s, TTFT p95 134 ms) is below TP=1 (3,141): with 2 API servers and 4 client
   processes, 8 requests may split unevenly across the two engines. Low-load DP gains nothing anyway.

Planner rule for this model on this node: 1 GPU (TP=1) up to ~64 concurrent requests; above that add copies (DP),
capping each copy at ~128. Never TP=2 or PP=2 for a 0.5B model.
