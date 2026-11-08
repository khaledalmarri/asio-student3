# Steps 1–2: manual parallelism comparison (2026-11-08, gpu-02)

Model: Qwen/Qwen2.5-0.5B-Instruct (BF16; 24 layers, hidden 896, 14 heads, 2 KV heads, head_dim 64, vocab 151,936, tied embeddings).
Server: vLLM 0.31.0+cu126.abujafar1 (image `fbdaa5259f0a`), `--gpu-memory-utilization 0.25 --max-model-len 4096`,
FLASH_ATTN (FA2). GPUs 0 and 1 (both shared with idle tenants).

```bash
docker run -d --name student3-<cfg> --gpus '"device=0,1"' --ipc=host -p 8401:8000 \
  -v /root/.cache/huggingface:/hf:ro \
  -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  --entrypoint vllm fbdaa5259f0a \
  serve /hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
  --attention-backend FLASH_ATTN --attention-config '{"flash_attn_version":2}' \
  --gpu-memory-utilization 0.25 --max-model-len 4096 \
  [--tensor-parallel-size 2 | --pipeline-parallel-size 2 | --data-parallel-size 2]
```

## Memory (from vLLM logs)

| Config | Weights per GPU | KV cache (tokens) | KV bytes/token/GPU | Overhead per GPU (derived) |
|---|---|---|---|---|
| TP=1 (1 GPU) | 0.93 GiB | 1,570,816 | 12 KiB | 1.09 GiB |
| TP=2 | 0.46 GiB | 3,128,480 | 6 KiB | 1.64 GiB |
| PP=2 | 0.59 GiB | 3,249,840 | 6 KiB | 0.81 GiB |
| DP=2 | 0.93 GiB ×2 | 1,570,816 ×2 | 12 KiB | 1.09 GiB |

Overhead = 0.25 × 80 GiB − weights − KV tokens × KV bytes/token.
Predictions made beforehand: TP1 weights 0.93 (exact), KV ~1.53M (−2.7%); TP2 weights 0.47, KV 3.0–3.1M;
PP2 weights 0.59 (exact — embedding 0.25 GiB cannot be split across stages).
TP all-reduce backend chosen by vLLM: `CUSTOM`.

## Speed

Single request: prompt "Hello", 512 output tokens, `ignore_eos`, median of runs 2–3.
Load: 64 concurrent requests × 256 output tokens via curl + xargs (coarse; includes curl start-up).

| Config | ms/token, single request | tok/s at 64 concurrent | 64-way success |
|---|---|---|---|
| TP=1 | 1.95 | — | — |
| TP=2 | 1.93 | — | — |
| PP=2 | 2.62 | ~10,500–10,850 (2 runs) | 64/64 |
| DP=2 | 1.98 | ~20,300 | 64/64 |

## Findings (to re-verify with bench.py)

1. Small model is per-step-overhead bound: weight read ≈ 0.64 ms of 1.95 ms per token.
2. TP=2 gives no latency gain for a 0.5B model; custom all-reduce costs ≈ 6–7 µs each (48 per token).
3. PP=2 adds ≈ 0.67 ms per token for a single request; lowest memory overhead, most KV.
4. DP=2 keeps TP=1 latency and roughly doubles PP=2 throughput at 64-way load.
