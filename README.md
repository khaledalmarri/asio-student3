# ASIO — Student 3: Parallelism & Serving Configuration Optimizer

Given a model, a workload and a GPU allocation, choose a feasible parallelism
configuration (TP / PP / DP / CP, aggregated or prefill/decode-disaggregated),
explain the choice, and deploy it on real GPUs.

Owned APIs (to agree with the team): `/parallelism/plan`, `/parallelism/benchmark`.

## Build steps

| Step | What | Status |
|---|---|---|
| 0 | Node inventory (gpu-02) | done |
| 1 | vLLM serving one model on one GPU | done |
| 2 | Manual TP=2, PP=2, DP=2 on the same model | done (single request + 64-way load) |
| 3 | Benchmark script `bench.py` (TTFT, TPOT, throughput, errors) | in progress |
| 4 | Memory feasibility + performance model | |
| 5 | Planner: enumerate → filter → score → explain | |
| 6 | Deployment templates + deployer | |
| 7 | FastAPI service `/parallelism/*` | |
| 8 | Prefill/decode disaggregation with Dynamo | |

## Node facts (gpu-02, measured 2026-11-08)

| Fact | Value | Planner impact |
|---|---|---|
| GPUs | 8× A100-SXM4-80GB | 80 GB per GPU; no FP8 compute; FlashAttention 2 only (FA3 needs Hopper) |
| Interconnect | 12 NVLink links × 25 GB/s per GPU, NVSwitch (Fabric Manager active), P2P over NVLink OK on all pairs | Any GPUs can form a TP group; placement is decided by free memory |
| PCIe / NUMA | GPU pairs (0-1, 2-3, 4-5, 6-7) under PCIe switches; GPUs 0-3 on NUMA 0, 4-7 on NUMA 1 | Pin CPU threads to the GPU's NUMA node |
| Driver / CUDA | 560.35.05 / 12.6 | Official vLLM 0.31.0 wheel is CUDA 13 (will not run); `+cu129` wheel exists, untested |
| Shared | All 8 GPUs have other tenants; 3 at ~99% compute | Pick GPUs by free memory; benchmarks need a reservation |
| Host | Ubuntu 20.04, Python 3.8 (3.12 via uv), Docker + nvidia-ctk, no Slurm, 1 TB RAM, 256 threads | Launch processes/containers directly |

Always `export CUDA_DEVICE_ORDER=PCI_BUS_ID` so CUDA indices match `nvidia-smi`.

## Serving image used so far

`fbdaa5259f0a` (vLLM `0.31.0+cu126.abujafar1`, built by another node user). It needs:
mount the HF cache at a neutral path (container runs as uid 1000 and cannot read `/root`),
pass the model as a local snapshot path, and `--attention-backend FLASH_ATTN
--attention-config '{"flash_attn_version":2}'` (no FlashInfer in the image).
See `results/step1-2-manual.md` for the exact command.

## Benchmark

```bash
python bench.py --model Qwen/Qwen2.5-0.5B-Instruct --label qwen0.5b-dp2 --concurrency 1 8 32 64 128 256
```

Random token-ID prompts of exact length (no prefix-cache effects), streaming
TTFT/TPOT p50/p95, throughput, ok/failed counts. Appends one JSON line per
concurrency level to `results.jsonl`. Standard library only.

## Prior work

Calibration ideas and A100 measurements reused from the author's LoadCast project
(AI in Production, MBZUAI, Fall 2026): github.com/khaledalmarri/LoadCast,
Experiments 6, 9, 10, 11 (vLLM 0.8.5 on this same node).
