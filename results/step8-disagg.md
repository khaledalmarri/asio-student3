# Step 8: prefill/decode disaggregation — predictions and experiment plan

**Status: prediction only.** `planner/disagg.py` estimates the gain; nothing has been run on GPUs yet.

## Why (measured)

On Qwen2.5-7B TP=2, 32 concurrent requests, TPOT was **23.9 ms with 2048-token prompts vs 7.3 ms with 128-token
prompts** (`step4b-perf.md`). New prompts are read in the same steps as everyone else's tokens. Separate prefill
GPUs remove that interference from the decode GPUs.

## Predictions (recorded before any run; 7B, TPOT p95 ≤ 15 ms, 4 GPUs, mem-util 0.3)

`python -m planner.disagg --model Qwen/Qwen2.5-7B-Instruct --prompt P --output O --tpot-sla 15 --gpus 4 --mem-util 0.3`

| Workload | Aggregated (planner) | Disaggregated (best split) | Predicted gain | Recommendation |
|---|---|---|---|---|
| RAG 2048 / 128 | TP=4: 2,042 tok/s | prefill 3×TP1 + decode 1×TP1: 2,927 tok/s (prefill-bound) | **1.43×** | disaggregate |
| 512 / 128 | TP=4: 7,200 tok/s | prefill 3×TP1 + decode 1×TP1: 8,536 tok/s (decode-bound) | 1.19× | disaggregate (marginal) |
| Chat 128 / 512 | TP=2×DP=2: 24,840 tok/s | prefill 1×TP1 + decode 3×TP1: 27,473 tok/s | 1.11× | aggregated (below the 15% bar) |

Assumptions that the experiment must check:
- Decode-only TPOT = the validated perf model minus its prefill term.
- Prefill capacity uses the MFU fitted indirectly from the interference term (0.76); never measured directly.
- KV transfer at an **assumed** 50 GB/s effective (NIXL over NVLink; link peak 300 GB/s): ~2.3 ms per 2048-token request.
- Decode workers keep the same duty factor (0.85) as aggregated engines.
- TTFT is not compared (no TTFT model).

## Experiment plan on gpu-02

**Safety first.** Another user's Dynamo stack is running (`abu-jafar-mova-dynamo`, `-ingress`, `-moe`). Dynamo
workers find each other through a discovery service (etcd/NATS or similar). If our workers register in *their*
discovery namespace, their frontend could route their traffic to our GPUs, or ours to theirs. Do the discovery
below, and either agree with them or run a fully separate discovery service and namespace.

### 1. Discovery (read-only)

```bash
docker inspect fa1949a76b78 --format 'image={{.Config.Image}} cmd={{json .Config.Cmd}} entry={{json .Config.Entrypoint}}'
docker inspect f24b4c0b6bd6 --format 'image={{.Config.Image}} cmd={{json .Config.Cmd}}'
docker exec fa1949a76b78 sh -c 'env | grep -i -E "etcd|nats|dyn_|namespace" ; which etcd nats-server'
docker exec fa1949a76b78 python3 -m dynamo.vllm --help 2>&1 | grep -i -E "prefill|decode|disagg|connector|kv-transfer|namespace" | head -30
docker exec fa1949a76b78 python3 -m dynamo.frontend --help 2>&1 | head -40
```

### 2. Run (after discovery; template to fill from step 1)

One frontend, one prefill worker (GPU 0, TP=1), one decode worker (GPU 1, TP=1), own namespace and ports.
Then the same `bench.py` RAG test against the frontend, compared with an aggregated TP=1×DP=2 on the same 2 GPUs:

```bash
python bench.py --url http://localhost:<frontend-port> --model Qwen/Qwen2.5-7B-Instruct --label qwen7b-disagg-1p1d \
  --prompt-tokens 2048 --output-tokens 128 --concurrency 1 8 32 --procs 4 --out results/results.jsonl
```

Recorded prediction for that 2-GPU RAG run (TPOT p95 ≤ 15 ms, mem-util 0.3):
aggregated TP=2 **887 tok/s** vs prefill 1×TP1 + decode 1×TP1 **976 tok/s** → **1.10×, a near-tie**.
On 2 GPUs the single prefill GPU is the bottleneck; the predicted 1.43× needs 4 GPUs (3 prefill + 1 decode).
A 2-GPU run therefore tests the model (does decode-only TPOT drop as predicted, ~13.5 ms at its concurrency limit?),
not the headline gain.
