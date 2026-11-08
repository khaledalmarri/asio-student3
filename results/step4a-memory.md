# Step 4a: memory model validation log

`python -m planner.check_memory` prints the current table. History of tests, in order:

| # | Prediction (made before the run) | Measured | Error | Outcome |
|---|---|---|---|---|
| 1 | Qwen2.5-7B TP1, mem 0.3: 132,734 tokens (overhead slope fitted on Llama-8B, vLLM 0.8.5) | 144,832 | −8.4% | **Failed** (±5% bar). Weights right (14.18 vs 14.29 GiB); overhead 2.73 predicted vs 1.98 actual |
| 2 | Qwen2.5-7B TP2, mem 0.3: ~538,000 tokens (refit on vLLM 0.31 TP1 points only, TP term +0.55) | 564,176 | −4.6% | Passed, barely. TP=2 overhead for the 7B (1.77 GiB) is *lower* than its TP=1 overhead (1.98) |

Changes after test 2 (chosen after seeing it, so test 2 no longer counts as held-out):
- Overhead is calibrated **per vLLM version** (`CALIBRATION["0.31"]`, `CALIBRATION["0.8.5"]`).
- Activation overhead is divided by TP: `base + per_hidden × hidden / TP + tp_extra`.

Pending held-out tests on vLLM 0.31 (predictions recorded before running):

| # | Config | Predicted KV tokens |
|---|---|---|
| 3 | Qwen2.5-7B PP2, mem 0.3, GPUs 0,1 | 578,471 → measured **568,048 (+1.8%), passed**; weights 7.15 GiB/stage vs 7.09 predicted |
| 4 | Qwen2.5-7B TP4, mem 0.3, GPUs 0,1,4,5 | 1,397,907 → measured **1,390,896 (+0.5%), passed**; weights 3.63 GiB/GPU vs 3.55 predicted. vLLM chose the CUSTOM all-reduce for 4 GPUs across both NUMA nodes (it requires full NVLink connectivity above 2 GPUs) |

Known small bias: weights are under-predicted by ~0.8% (vLLM reports 14.29 vs 14.18 GiB for the 7B,
0.93 vs 0.92 for the 0.5B); parameter counting ignores biases, norms and runtime buffers.

## Status

Two pre-registered tests on vLLM 0.31 passed (PP=2 +1.8%, TP=4 +0.5%). The model is usable for feasibility
filtering on this node and image. Untested: TP=8, TP×PP combinations, models with hidden size > 3584 on vLLM 0.31,
MoE models, quantized weights. The planner should keep a safety margin (e.g. require 10% spare KV) until these are covered.
