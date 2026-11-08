#!/usr/bin/env bash
# Start one vLLM configuration, wait until healthy, benchmark it, save logs, remove it.
#
# Usage:   ./run_config.sh <label> <gpus> [extra vllm flags...]
# Example: ./run_config.sh qwen0.5b-tp2 0,1 --tensor-parallel-size 2
#
# Override defaults with env vars: MODEL_ID, MODEL_PATH, IMAGE, PORT, MEM, MAXLEN, CONC
set -euo pipefail

LABEL=$1; GPUS=$2; shift 2
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}
IMAGE=${IMAGE:-fbdaa5259f0a}
PORT=${PORT:-8401}
MEM=${MEM:-0.25}
MAXLEN=${MAXLEN:-4096}
CONC=${CONC:-"1 8 32 64 128 256"}
NAME=student3-bench
cd "$(dirname "$0")"
mkdir -p results/logs

# Always remove the container on exit (success, failure or Ctrl+C) so GPUs are freed.
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[$LABEL] GPUs $GPUS, flags: $*"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$GPUS" \
  | sed 's/^/  before: GPU /'

docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" --ipc=host -p "$PORT:8000" \
  -v /root/.cache/huggingface:/hf:ro \
  -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  --entrypoint vllm "$IMAGE" \
  serve "$MODEL_PATH" --served-model-name "$MODEL_ID" \
  --attention-backend FLASH_ATTN --attention-config '{"flash_attn_version":2}' \
  --gpu-memory-utilization "$MEM" --max-model-len "$MAXLEN" "$@" >/dev/null

# Wait for /health, but stop early if the container dies.
for i in $(seq 1 120); do
  if curl -sf "localhost:$PORT/health" >/dev/null; then break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "[$LABEL] server crashed during start-up:"; docker logs --tail 30 "$NAME" 2>&1
    docker logs "$NAME" > "results/logs/$LABEL.log" 2>&1 || true
    exit 1
  fi
  sleep 5
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "[$LABEL] not healthy after 10 min"; exit 1; }
echo "[$LABEL] healthy after ~$((i * 5)) s"

docker logs "$NAME" 2>&1 | grep -E "Model loading took|KV cache size|all-reduce backends" | sed 's/^.*INFO/  /'

python bench.py --url "http://localhost:$PORT" --model "$MODEL_ID" --label "$LABEL" \
  --concurrency $CONC --out results/results.jsonl

docker logs "$NAME" > "results/logs/$LABEL.log" 2>&1
echo "[$LABEL] done; log saved to results/logs/$LABEL.log"
