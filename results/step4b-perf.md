# Step 4b: performance model (2026-11-08)

`planner/perf.py`: steady-state model of one vLLM replica (TP; DP = independent replicas). Formula in the module
docstring. Calibration for vLLM 0.31 on gpu-02 fitted by `python -m planner.fit_perf`.

**Fit set:** Qwen2.5-0.5B TP1, Qwen2.5-7B TP1, Qwen2.5-7B TP4 (TPOT p50 at concurrency 1–128).
**Test set (not used in the fit):** Qwen2.5-0.5B TP2, Qwen2.5-7B TP2, Qwen2.5-0.5B DP2.

| | Mean abs. error TPOT p50 | Worst TPOT | Mean abs. error throughput | Worst throughput |
|---|---|---|---|---|
| Fit | 5.5% | 16% | 9.4% | 22% |
| **Test** | **6.4%** | **14%** | **10.9%** | **37%** (0.5B TP2 at 128, over-predicted) |

## Decision check (`python -m planner.check_decisions`)

Same procedure on measured and predicted values: largest benchmarked concurrency whose TPOT p95 meets the SLA,
then rank layouts by tokens/s per GPU.

| Case | Measured ranking | Model ranking | Verdict |
|---|---|---|---|
| 7B, TPOT p95 ≤ 15 ms | TP4 > TP2 > TP1 | TP4 > TP2 > TP1 | same |
| 7B, TPOT p95 ≤ 30 ms | TP1 (2,214/GPU) > TP2 (2,151) > TP4 | TP2 > TP1 > TP4 | tie: model pick is within 3% of measured best |
| 0.5B, TPOT p95 ≤ 5 ms | DP2 (13,693/GPU) > TP1 (13,010) > TP2 | TP1 > DP2 > TP2 | tie: within 5% |

## Caveats

- The constants are not physical measurements: `eta` = 0.96 of peak bandwidth is implausibly high; it trades off
  against `fixed_ms`. Only their combination is validated.
- Throughput uses a fitted duty factor (0.85) because the steady-state model has no TTFT/queueing.
- TTFT under load is not modelled. SLAs on TTFT need measured data or a queueing model.
- The "tie within 5%" rule is generous given the ~3% run-to-run noise; near-ties need a benchmark to decide.
- Not covered: PP, MoE, quantized weights, other prompt/output ratios (all data is 512 in / 128 out), models
  larger than 7B on vLLM 0.31, concurrency above 128 per engine.

## Full fit output

```
fitted: {'eta': 0.961, 'fixed_ms': 1.4, 'allreduce_us': 8.42, 'per_request_us': 9.81, 'mfu': 0.762, 'duty': 0.849} 
(stored calibration: {'eta': 0.961, 'fixed_ms': 1.4, 'allreduce_us': 8.42, 'per_request_us': 9.81, 'mfu': 0.762, 'duty': 0.849, 'p95_factor': 1.076} )

config             conc  TPOT p50   pred    err     tok/s    pred    err  set
qwen0.5b-tp1-v2       1      1.92   1.92    -0%       485     443    -9%  fit
qwen0.5b-tp1-v2       8      2.17   2.13    -2%     3,140   3,192    +2%  fit
qwen0.5b-tp1-v2      32      3.02   2.85    -6%     8,391   9,536   +14%  fit
qwen0.5b-tp1-v2      64      3.49   3.81    +9%    13,010  14,260   +10%  fit
qwen0.5b-tp1-v2     128      5.53   5.73    +4%    15,546  18,955   +22%  fit
qwen7b-tp1-v2         1     10.12   9.20    -9%        96      92    -4%  fit
qwen7b-tp1-v2         8     11.11  11.18    +1%       635     608    -4%  fit
qwen7b-tp1-v2        32     15.90  17.97   +13%     1,686   1,512   -10%  fit
qwen7b-tp1-v2        64     25.61  27.02    +6%     2,214   2,011    -9%  fit
qwen7b-tp1-v2       128     44.34  45.13    +2%     2,625   2,408    -8%  fit
qwen7b-tp4-v2         1      4.04   4.30    +6%       239     197   -17%  fit
qwen7b-tp4-v2         8      4.78   4.85    +1%     1,460   1,401    -4%  fit
qwen7b-tp4-v2        32      6.63   6.72    +1%     3,897   4,042    +4%  fit
qwen7b-tp4-v2        64      9.85   9.22    -6%     5,507   5,893    +7%  fit
qwen7b-tp4-v2       128     17.00  14.22   -16%     6,529   7,644   +17%  fit
  fit: mean |error| TPOT 5.5%, throughput 9.4%, worst TPOT 16%

qwen0.5b-tp2-v2       1      1.91   2.07    +8%       495     411   -17%  test
qwen0.5b-tp2-v2       8      2.21   2.21    -0%     3,171   3,077    -3%  test
qwen0.5b-tp2-v2      32      2.91   2.69    -8%     8,730  10,116   +16%  test
qwen0.5b-tp2-v2      64      3.40   3.32    -2%    13,710  16,350   +19%  test
qwen0.5b-tp2-v2     128      5.12   4.60   -10%    17,227  23,632   +37%  test
qwen0.5b-dp2-v2       1      1.90   1.92    +1%       500     443   -11%  test
qwen0.5b-dp2-v2       8      2.30   2.01   -13%     2,798   3,383   +21%  test
qwen0.5b-dp2-v2      32      2.58   2.37    -8%    10,486  11,472    +9%  test
qwen0.5b-dp2-v2      64      2.79   2.85    +2%    18,457  19,072    +3%  test
qwen0.5b-dp2-v2     128      3.56   3.81    +7%    27,387  28,520    +4%  test
qwen0.5b-dp2-v2     256      5.42   5.73    +6%    33,839  37,910   +12%  test
qwen7b-tp2-v2         1      6.29   5.78    -8%       155     147    -5%  test
qwen7b-tp2-v2         8      7.89   6.80   -14%       918     999    +9%  test
qwen7b-tp2-v2        32      9.82  10.31    +5%     2,694   2,634    -2%  test
qwen7b-tp2-v2        64     14.89  15.00    +1%     3,686   3,623    -2%  test
qwen7b-tp2-v2       128     26.71  24.36    -9%     4,302   4,460    +4%  test
  test: mean |error| TPOT 6.4%, throughput 10.9%, worst TPOT 14%

```
