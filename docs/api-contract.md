# `/parallelism/plan` — PROPOSED contract (not yet agreed by the team)

Owner: Student 3. Callers: Student 1 (platform core / deployer), Student 2 (admission & SLA), Student 4 (MoE).
Status: draft for team review. Field names may change; the planner logic (`planner/plan.py`) does not depend on them.

## Request — `POST /parallelism/plan`

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "workload": {"prompt_tokens": 512, "output_tokens": 128, "max_model_len": 4096},
  "sla": {"tpot_p95_ms": 15},
  "allocation": {"gpu_ids": [0, 1, 4, 5], "gpu_mem_util": 0.9},
  "runtime": {"engine": "vllm", "version": "0.31"}
}
```

| Field | From | Meaning |
|---|---|---|
| `model` | caller | Known model name or a path to the model folder on the node |
| `workload.prompt_tokens`, `output_tokens` | Student 2 | Typical request shape (p50 is fine); drives prompt interference |
| `workload.max_model_len` | caller | Longest request the deployment must accept; memory feasibility checks it |
| `sla.tpot_p95_ms` | Student 2 | Per-token latency limit; the planner plans to 85% of it |
| `target_concurrency` | Student 2 (optional) | Requests in flight to sustain. Given → fewest GPUs that meet it. Omitted → maximise throughput on the allocation |
| `allocation.gpu_ids` | Student 1 | GPUs this deployment may use (identified by index; UUIDs preferred later) |
| `allocation.gpu_mem_util` | Student 1 | Fraction of each GPU's memory the deployment may reserve. Student 1 must guarantee that much is free |

Not yet supported (planned): `sla.ttft_p95_ms` (no TTFT model), MoE expert-parallel input from Student 4, disaggregated prefill/decode.

## Response (real planner output for the request above, without `target_concurrency`, trimmed)

```json
{
  "status": "ok",
  "model": "Qwen2.5-7B-Instruct",
  "objective": "maximise SLA-compliant throughput on the 4 GPUs given",
  "chosen": {
    "name": "TP=4 PP=1 DP=1",
    "tp": 4,
    "pp": 1,
    "dp": 1,
    "gpus": 4,
    "gpu_ids": [
      0,
      1,
      4,
      5
    ],
    "gpu_groups": [
      [
        0,
        1,
        4,
        5
      ]
    ],
    "concurrency_per_replica": 97,
    "concurrency_total": 97,
    "tpot_p95_ms": 12.69,
    "throughput_tok_s": 6981.0,
    "per_gpu_tok_s": 1745.0,
    "kv_tokens_per_replica": 4993024,
    "vllm_args": [
      "--tensor-parallel-size",
      "4",
      "--pipeline-parallel-size",
      "1",
      "--gpu-memory-utilization",
      "0.9",
      "--max-model-len",
      "4096",
      "--max-num-seqs",
      "97"
    ],
    "notes": []
  },
  "alternatives": [
    {
      "name": "TP=2 PP=1 DP=2",
      "throughput_tok_s": 6055.0
    }
  ],
  "near_ties": [],
  "rejected": [
    {
      "name": "TP=1 PP=2 DP=1",
      "reason": "SLA: even one request takes 13.3 ms/token (p95) > 12.8 ms (85% of the 15 ms SLA)"
    }
  ],
  "warnings": [],
  "explanation": [
    "Objective: maximise SLA-compliant throughput on the 4 GPUs g...",
    "Chosen TP=4 PP=1 DP=1 on 4 GPU(s): up to 97 requests per eng...",
    "Runner-up TP=2 PP=1 DP=2: 6,055 tok/s (1,514 per GPU), 13% l..."
  ],
  "calibration": {
    "vllm_version": "0.31",
    "gpu": "A100-SXM4-80GB",
    "node": "gpu-02"
  }
}
```

| Field | For whom | Meaning |
|---|---|---|
| `chosen.vllm_args` | Student 1 | Arguments to append to `vllm serve <model>`. `--max-num-seqs` enforces the SLA concurrency per engine |
| `chosen.gpu_groups` | Student 1 | GPU ids per replica (one inner list per DP replica) |
| `chosen.concurrency_per_replica` | Student 2 | Admission limit per engine that keeps the SLA |
| `near_ties` | everyone | Layouts within 5% of the chosen one: run `/parallelism/benchmark` before committing |
| `warnings` | everyone | e.g. model outside the calibrated range → predictions extrapolated |
| `status` | everyone | `ok` or `infeasible` (then `explanation` says why every layout failed) |

## Questions for the team

1. Does Student 1 identify GPUs by index or UUID? (Indices can change between reboots; UUIDs do not.)
2. Does Student 2 express load as concurrency or as requests/s? (The planner can convert with Little's law if given output length and TPOT.)
3. Who owns `gpu_mem_util` — the allocator (what is free) or the planner (what is needed)?
4. Should the planner be allowed to return fewer GPUs than allocated (it does today when `target_concurrency` is set)?
