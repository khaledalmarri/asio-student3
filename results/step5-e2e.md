# Step 5: end-to-end tests — deploy the planner's choice, then check its promise

Procedure: `python -m planner.deploy ... --launch` (plan → container → healthy), then
`bench.py --concurrency <planned max> --procs 4`, then remove. Prediction = the planner's own output, recorded
before the benchmark.

## Test 1 (2026-11-08) — FAILED the SLA

Request: Qwen2.5-7B, 512 in / 128 out, TPOT p95 ≤ 15 ms, GPUs 0,1,4,5, mem-util 0.3. Headroom 90%.

| | Planner promised | Measured | |
|---|---|---|---|
| Layout | TP=4, `--max-num-seqs 106` | deployed as planned, healthy after 101 s | ✓ Step 6 deployer works |
| TPOT p95 @106 | 13.45 ms (≤ 15) | **15.53 ms** | ✗ SLA missed by 3.5% |
| TPOT p50 @106 | 12.50 ms | 13.71 ms | model −9% |
| Throughput | 7,200 tok/s | **5,947 tok/s** | model +21% |
| Failed requests | — | 0 / 424 | ✓ |

Why: the performance model under-predicts TPOT for TP=4 at high concurrency (−16% at 128 in its own fit set), and
p95/p50 was 1.133 here vs the 1.076 assumed. Measured/predicted TPOT p95 = 1.155.

Fix: planning headroom 0.90 → **0.85** (1/1.155 = 0.866, rounded down). Root-cause refinement left for later:
an MFU that falls with TP (smaller matrix multiplies per GPU) would address the TP=4 bias directly.

## Test 2 (2026-11-08) — PASSED, with a thin margin

Planner now chooses TP=4 with `--max-num-seqs 97`, predicted TPOT p95 12.69 ms, 6,981 tok/s.
Applying test 1's measured ratios: expected TPOT p95 ≈ 12.69 × 1.155 ≈ **14.7 ms** (passes) and throughput
≈ 6,981 × 0.826 ≈ **5,770 tok/s**. Pass = TPOT p95 ≤ 15 ms with 0 failed requests.

| | Planner (raw) | Corrected expectation | Measured | |
|---|---|---|---|---|
| Layout | TP=4, `--max-num-seqs 97` | — | deployed as planned, healthy after 101 s | ✓ |
| TPOT p95 @97 | 12.69 ms | 14.7 ms | **14.62 ms** | ✓ SLA met (0.38 ms / 2.5% to spare) |
| TPOT p50 @97 | — | — | 12.62 ms | |
| Throughput | 6,981 tok/s | 5,770 tok/s | **5,751 tok/s** | raw model still +21% |
| Failed requests | — | — | 0 / 388 | ✓ |

The fix worked, but the margin (2.5%) is inside the ~3% run-to-run noise measured in Step 3, so a repeat could
miss. The raw performance model is still biased for TP=4 at high load (TPOT −13%, throughput +21%); the headroom
hides that rather than fixing it. Next: 2–3 repeats of this test, and an MFU-vs-TP term in `planner/perf.py`.
