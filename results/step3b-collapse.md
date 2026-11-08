# Step 3b: why single-copy configs lose throughput above 128 concurrent (2026-11-08)

Same setup as Step 3 (Qwen2.5-0.5B, 512-token prompts, 128 output tokens, GPU 0 / GPUs 0,1). One variable per run.
Baselines from Step 3: TP=1 15,364 tok/s at 128, 12,272 at 256; DP=2 20,842 at 128, 23,419 at 256.

| Exp | Change | tok/s @128 | tok/s @256 | TTFT p95 @256 | TPOT p95 @256 | Verdict |
|---|---|---|---|---|---|---|
| A | TP=1 `--max-num-seqs 512` | 14,725 (−4%) | 12,538 (+2%) | 1,078 ms | 17.2 ms | No effect: sequence limit is not the cause |
| B | TP=1 `--max-num-batched-tokens 16384` | 15,774 (+3%) | 13,039 (+6%) | **576 ms** (was 1,117) | 18.1 ms | Not the cause; trades TTFT for TPOT |
| C | TP=1 at 256, `top` per process | — | 12,202 | 1,117 ms | 17.5 ms | Inconclusive: EngineCore up to 499%, API server up to 437% CPU (process totals, not per thread) |
| D | DP=2 `--api-server-count 2` | **22,972 (+10%)** | 23,313 (0%) | 389 ms | 6.6 ms | Second API server helps at 128, not at 256 |

Also ruled out earlier: CUDA graphs (largest capture size 512 ≥ 256; mode FULL_AND_PIECEWISE).

## What we learned

1. The prefill token budget (`--max-num-batched-tokens`) is a serving-profile knob: 8× larger budget halved TTFT p95
   at 256 (1,117 → 576 ms) and raised TPOT p95 (17.2 → 18.1 ms). The planner should set it from the SLA
   (TTFT-sensitive vs TPOT-sensitive workloads).
2. DP=2 has now plateaued at ~23,300–23,400 tok/s in three runs (one API server @256, two @128, two @256).
   The same Python client process parses every streamed chunk, so the client itself may cap at ~23k tok/s
   (Experiment E: `bench.py --procs 4`).
3. The TP=1 drop at 256 is still unexplained. Next diagnostics: thread-level CPU (`top -H`) and vLLM's own
   `/metrics` (running/waiting requests, server-side TPOT) during the 256 level.

## Planner rule (holds regardless of the cause)

Cap concurrency at ~128 requests per Qwen-0.5B copy on this node; above that, add copies (DP) instead of
admitting more requests to one engine.
