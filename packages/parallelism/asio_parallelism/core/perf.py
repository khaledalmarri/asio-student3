"""Step 4b: steady-state performance model for one vLLM replica (TP only; DP = independent replicas).

Every decode step, all b running requests advance one token. A step costs:

  decode    = weights_per_gpu / (bandwidth x eta)            read every weight once
            + fixed                                          scheduler, sampling, launches (per step)
            + 2 x layers x allreduce x log2(TP)              two all-reduces per layer (tree/ring depth)
            + b x context x kv_bytes_per_token / (bandwidth x eta)   read each request's KV cache
            + b x per_request                                per-sequence CPU/GPU bookkeeping
  prefill   = ((b - 1) / b) x b x (prompt / output) x 2 x params / (TP x peak x mfu)
              In steady state every finished request is replaced by a new one, so each step also
              reads b x prompt/output prompt tokens. A request does not wait for its own prompt
              (that is TTFT), hence the (b - 1) / b factor: at b = 1 there is no interference.

  TPOT(b) = decode + prefill
  throughput = active requests / TPOT x duty      duty < 1: the share of time a request is generating
              rather than waiting for its first token (fitted; the steady-state model has no TTFT)

Not modelled: TTFT under load (needs a queueing model), pipeline parallelism, the single-engine
throughput drop above ~128 concurrent requests seen on gpu-02 (the planner caps concurrency there).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .model_spec import ModelSpec


@dataclass(frozen=True)
class GPU:
    name: str
    mem_bw: float        # bytes / s
    peak_flops: float    # dense BF16 FLOP / s


A100_SXM_80GB = GPU("A100-SXM4-80GB", mem_bw=2.039e12, peak_flops=312e12)

# Fitted by planner/fit_perf.py on gpu-02 data (vLLM 0.31, 2026-11-08). See results/step4b-perf.md.
CALIBRATION = {
    "0.31": dict(eta=0.961, fixed_ms=1.4, allreduce_us=8.42, per_request_us=9.81, mfu=0.762, duty=0.849,
                 p95_factor=1.076),   # TPOT p95 / p50, median on the fit set
}
DEFAULT_VERSION = "0.31"


@dataclass(frozen=True)
class Workload:
    prompt_tokens: int
    output_tokens: int


def step_breakdown(m: ModelSpec, tp: int, b: float, w: Workload, gpu: GPU = A100_SXM_80GB,
                   params: dict | None = None) -> dict:
    """Per-step time components in milliseconds for one replica with b running requests."""
    c = params or CALIBRATION[DEFAULT_VERSION]
    bw = gpu.mem_bw * c["eta"]
    weights = m.total_params * m.dtype_bytes / tp
    kv_bpt = 2 * m.layers * math.ceil(m.kv_heads / tp) * m.head_dim * m.dtype_bytes
    context = w.prompt_tokens + w.output_tokens / 2            # average KV length while decoding
    parts = {
        "weights": weights / bw * 1e3,
        "fixed": c["fixed_ms"],
        "allreduce": 2 * m.layers * c["allreduce_us"] / 1e3 * math.log2(tp),
        "kv_read": b * context * kv_bpt / bw * 1e3,
        "per_request": b * c["per_request_us"] / 1e3,
    }
    prompt_per_step = max(b - 1, 0) * w.prompt_tokens / w.output_tokens
    parts["prefill"] = prompt_per_step * 2 * m.total_params / (tp * gpu.peak_flops * c["mfu"]) * 1e3
    return parts


def tpot_ms(m: ModelSpec, tp: int, b: float, w: Workload, **kw) -> float:
    return sum(step_breakdown(m, tp, b, w, **kw).values())


def replica_load(concurrency: int, dp: int) -> tuple[float, int]:
    """Requests per busy replica and number of busy replicas when the router spreads load evenly."""
    busy = min(dp, max(concurrency, 1))
    return concurrency / busy, busy


def throughput(m: ModelSpec, tp: int, concurrency: int, w: Workload, dp: int = 1,
               params: dict | None = None, **kw) -> float:
    """Output tokens/s for `concurrency` requests spread over dp replicas of a TP=tp engine."""
    c = params or CALIBRATION[DEFAULT_VERSION]
    b, busy = replica_load(concurrency, dp)
    return busy * b / tpot_ms(m, tp, b, w, params=c, **kw) * 1e3 * c["duty"]


def tpot_p95_ms(m: ModelSpec, tp: int, b: float, w: Workload, params: dict | None = None, **kw) -> float:
    c = params or CALIBRATION[DEFAULT_VERSION]
    return tpot_ms(m, tp, b, w, params=c, **kw) * c["p95_factor"]


def max_concurrency_for_sla(m: ModelSpec, tp: int, w: Workload, tpot_p95_sla_ms: float,
                            cap: int = 128, levels: list[int] | None = None, **kw) -> int:
    """Largest per-replica concurrency whose predicted TPOT p95 meets the SLA (0 if even b=1 fails).
    cap=128: above that one engine loses throughput on gpu-02 (Step 3b), so the planner never goes there."""
    best = 0
    for b in (levels or range(1, cap + 1)):
        if b > cap:
            break
        if tpot_p95_ms(m, tp, b, w, **kw) <= tpot_p95_sla_ms:
            best = b
        else:
            break
    return best
