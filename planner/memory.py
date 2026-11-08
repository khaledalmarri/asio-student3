"""Step 4a: per-GPU memory model for vLLM serving under TP x PP.

budget          = gpu_memory_utilization x GPU memory              (vLLM reserves all of it up front)
weights / GPU   = largest pipeline stage's weights / TP
                  (TP splits everything incl. the vocabulary; PP cannot split the embedding:
                   first stage holds the input embedding, last stage the output head - a copy when tied)
KV bytes/token  = 2 (K,V) x layers in the stage x ceil(kv_heads / TP) x head_dim x bytes
overhead / GPU  = activations + CUDA graphs + communication buffers (calibrated per vLLM version, see CALIBRATION)
KV tokens       = (budget - weights - overhead) / KV bytes per token, minimum over stages

DP replicas do not change any per-GPU number; each replica has its own KV cache.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .model_spec import ModelSpec

GiB = 1024 ** 3

# Overhead per GPU (GiB) = base + per_hidden x hidden / TP  [x pp_factor when PP > 1]  + tp_extra when TP > 1
# Activation memory shrinks with TP; communication buffers appear when TP > 1. Calibrated per vLLM version,
# because overhead changes between versions (Llama-8B: 3.04 GiB on 0.8.5; a 7B uses 1.98 GiB on 0.31).
CALIBRATION = {
    # Fit (2026-11-08, gpu-02): base/per_hidden from Qwen2.5-0.5B TP1 (1.09) and Qwen2.5-7B TP1 (1.98);
    # tp_extra from Qwen2.5-0.5B TP2 (1.64); pp_factor from Qwen2.5-0.5B PP2 (0.81).
    # Not used in the fit: Qwen2.5-7B TP2.
    "0.31": dict(base=0.795, per_hidden=0.000329, tp_extra=0.70, pp_factor=0.74),
    # Fit from LoadCast Exp 9/11 (vLLM 0.8.5): Llama-3.1-8B TP1 (3.04) and TP2 (4.49); not used: Llama-3.1-70B TP4.
    "0.8.5": dict(base=0.544, per_hidden=0.000609, tp_extra=2.70, pp_factor=0.74),
}
DEFAULT_VERSION = "0.31"


@dataclass
class MemoryEstimate:
    tp: int
    pp: int
    budget_gib: float
    weights_gib: float          # largest stage, per GPU
    overhead_gib: float
    kv_gib: float               # per GPU, worst stage
    kv_bytes_per_token: int     # per GPU, worst stage
    kv_tokens: int              # per replica
    fits: bool
    reason: str

    def max_concurrency(self, tokens_per_request: int) -> float:
        return self.kv_tokens / tokens_per_request


def stage_layers(layers: int, pp: int) -> list[int]:
    """vLLM-style even split: earlier stages get the remainder."""
    base, rem = divmod(layers, pp)
    return [base + (1 if i < rem else 0) for i in range(pp)]


def overhead_gib(m: ModelSpec, tp: int, pp: int, vllm_version: str = DEFAULT_VERSION) -> float:
    c = CALIBRATION[vllm_version]
    o = c["base"] + c["per_hidden"] * m.hidden / tp
    if pp > 1:
        o *= c["pp_factor"]
    if tp > 1:
        o += c["tp_extra"]
    return o


def estimate(m: ModelSpec, tp: int = 1, pp: int = 1, gpu_mem_gib: float = 80.0,
             mem_util: float = 0.9, max_model_len: int | None = None,
             vllm_version: str = DEFAULT_VERSION) -> MemoryEstimate:
    budget = mem_util * gpu_mem_gib
    b = m.dtype_bytes
    embed = m.embed_params * b
    layer = m.layer_params * b
    kv_heads_per_gpu = math.ceil(m.kv_heads / tp)       # KV heads are replicated when TP > kv_heads
    over = overhead_gib(m, tp, pp, vllm_version)

    worst_kv_tokens, worst = None, None
    for i, n in enumerate(stage_layers(m.layers, pp)):
        w = n * layer
        if i == 0:
            w += embed                                  # input embedding
        if i == pp - 1 and (pp > 1 or not m.tied_embeddings):
            w += embed                                  # output head (separate, or a copy of tied weights)
        w_gpu = w / tp / GiB
        kv_bpt = 2 * n * kv_heads_per_gpu * m.head_dim * b
        kv_gib = budget - w_gpu - over
        tokens = int(kv_gib * GiB // kv_bpt) if kv_gib > 0 else 0
        if worst_kv_tokens is None or tokens < worst_kv_tokens:
            worst_kv_tokens, worst = tokens, (w_gpu, kv_gib, kv_bpt)

    w_gpu, kv_gib, kv_bpt = worst
    need = max_model_len or 1
    fits = kv_gib > 0 and worst_kv_tokens >= need
    if kv_gib <= 0:
        reason = f"weights {w_gpu:.1f} + overhead {over:.1f} GiB exceed the {budget:.1f} GiB budget per GPU"
    elif not fits:
        reason = f"KV cache holds {worst_kv_tokens:,} tokens < max_model_len {need:,}"
    else:
        reason = "fits"
    return MemoryEstimate(tp, pp, budget, w_gpu, over, kv_gib, kv_bpt, worst_kv_tokens, fits, reason)
