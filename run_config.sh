#!/usr/bin/env bash
# Start one vLLM configuration, wait until healthy, benchmark it, save logs, remove it.
#
# Usage:   ./run_config.sh <label> <gpus> [extra vllm flags...]
# Example: ./run_config.sh qwen0.5b-tp2 0,1 --tensor-parallel-size 2
#
# Override defaults with env vars: MODEL_ID, MODEL_PATH, IMAGE, PORT, MEM, MAXLEN, CONC, PROCS
# KEEP=1 starts the server and leaves it running (no benchmark); remove it with: docker rm -f student3-bench
set -euo pipefail

LABEL=$1; GPUS=$2; shift 2
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}
IMAGE=${IMAGE:-fbdaa5259f0a}
PORT=${PORT:-8401}
MEM=${MEM:-0.25}
MAXLEN=${MAXLEN:-4096}
CONC=${CONC:-"1 8 32 64 128 256"}
PROCS=${PROCS:-4}
KEEP=${KEEP:-0}
NAME=student3-bench
cd "$(dirname "$0")"
mkdir -p results/logs

# Remove the container on exit (success, failure or Ctrl+C) so GPUs are freed, unless KEEP=1.
cleanup() { if [ "$KEEP" != "1" ] || [ "${HEALTHY:-0}" != "1" ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT
docker rm -f "$NAME" >/dev/null 2>&1 || true

HOST_PATH=/root/.cache/huggingface${MODEL_PATH#/hf}
if [ ! -f "$HOST_PATH/config.json" ]; then
  echo "[$LABEL] model not found on host: $HOST_PATH (download it first, or check MODEL_PATH)"; exit 1
fi
echo "[$LABEL] model $MODEL_ID, mem-util $MEM, GPUs $GPUS, flags: $*"
sleep 5   # let the previous run's utilisation window expire, so this reading reflects other tenants
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$GPUS" \
  | sed 's/^/  before: GPU /'

# Inside the container the selected GPUs are renumbered 0..N-1. Set CUDA_VISIBLE_DEVICES explicitly so an
# image default (this image ships one) cannot hide some of them.
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
INNER=$(seq -s, 0 $((NGPU - 1)))
docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" -e CUDA_VISIBLE_DEVICES="$INNER" --ipc=host -p "$PORT:8000" \
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
HEALTHY=1

docker logs "$NAME" 2>&1 | grep -E "Model loading took|KV cache size|all-reduce backends" | sed 's/^.*INFO/  /'

if [ "$KEEP" = "1" ]; then
  echo "[$LABEL] KEEP=1: server left running on port $PORT. Remove with: docker rm -f $NAME"
  exit 0
fi

python bench.py --url "http://localhost:$PORT" --model "$MODEL_ID" --label "$LABEL" \
  --concurrency $CONC --procs "$PROCS" --out results/results.jsonl

docker logs "$NAME" > "results/logs/$LABEL.log" 2>&1
echo "[$LABEL] done; log saved to results/logs/$LABEL.log"
