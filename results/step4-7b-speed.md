# Qwen2.5-7B speed sweep: where TP starts paying off (2026-11-08, gpu-02)

vLLM 0.31.0+cu126.abujafar1, `--gpu-memory-utilization 0.3 --max-model-len 4096`, 512-token prompts, 128 output
tokens, `bench.py --procs 4`. TP1 on GPU 0; TP2 on 0,1; TP4 on 0,1,4,5. 1,440 requests, 0 failed.

## Output throughput (tok/s) and TPOT p95 (ms)

| Concurrency | TP=1 tok/s | TP=2 tok/s | TP=4 tok/s | TP=1 TPOT p95 | TP=2 TPOT p95 | TP=4 TPOT p95 |
|---|---|---|---|---|---|---|
| 1 | 96 | 155 | 239 | 10.12 | 6.29 | 4.05 |
| 8 | 635 | 918 | 1,460 | 12.19 | 8.39 | 5.25 |
| 32 | 1,686 | 2,694 | 3,897 | 17.76 | 11.05 | 7.58 |
| 64 | 2,214 | 3,686 | 5,507 | 27.04 | 16.17 | 10.78 |
| 128 | 2,625 | 4,302 | 6,529 | 45.59 | 27.6 | 18.0 |

## Single-request decode vs the LoadCast roofline (weights ÷ 77% of 2.04 TB/s + ~1.4 ms + all-reduce)

| | Predicted TPOT | Measured | Error | Speed-up vs TP=1 |
|---|---|---|---|---|
| TP=1 | ~11.1 ms | 10.12 | −9% | 1.0× |
| TP=2 | ~6.6 ms | 6.29 | −5% | 1.61× |
| TP=4 | ~4.3 ms | 4.05 | −6% | 2.50× |

Contrast: Qwen2.5-0.5B gained 1.01× from TP=2 (per-step overhead dominates a 0.93 GiB weight read).

## SLA view: the best layout depends on the SLA

| TPOT p95 SLA | TP=1 (and DP copies of it) | TP=2 | TP=4 | Most tok/s per GPU |
|---|---|---|---|---|
| ≤ 15 ms | 8 conc → 635 tok/s → **635/GPU** | 32 → 2,694 → **1,347/GPU** | 64 → 5,507 → **1,377/GPU** | **TP=4 ≈ TP=2 (2.2× TP=1)** |
| ≤ 30 ms | 64 → 2,214 → **2,214/GPU** | 128 → 4,302 → **2,151/GPU** | 128 → 6,529 → **1,632/GPU** | **TP=1 (DP copies)** |

## Findings

1. For a 7B, TP pays off: 1.61× (TP=2) and 2.50× (TP=4) faster tokens. For a 0.5B it did not. The crossover
   follows from weight-read time vs fixed per-step overhead, which the roofline predicts within 5–9%.
2. **Strict SLA → TP; loose SLA → DP copies of TP=1.** Under 15 ms TPOT, TP=2/TP=4 deliver 2.2× more tokens per
   GPU than TP=1, because TP=1 cannot hold the SLA above 8 concurrent requests. Under 30 ms, TP=1 copies are most
   efficient per GPU.
3. Under load this workload is prompt-bound: at 128 concurrent, TP=2 also processes ~17k prompt tok/s
   (≈2.6 × 10¹⁴ FLOP/s, ~40% of two A100s' peak), which is why TPOT grows 4× from 1 to 128 concurrent.
   The performance model needs a prefill-compute term; this is also the case for prefill/decode disaggregation.
