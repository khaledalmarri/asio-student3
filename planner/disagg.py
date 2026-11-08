"""Step 8: should prefill and decode run on separate GPUs? (prediction only - not yet validated on GPUs)

Aggregated (today): every engine reads new prompts and generates tokens in the same steps, so long prompts
slow everyone's tokens (the "prefill" term in planner/perf.py; measured 3.3x on 7B TP2 RAG vs chat).

Disaggregated (Dynamo): prefill workers only read prompts and hand the KV cache to decode workers over
NVLink (NIXL); decode workers only generate. Then:
  decode TPOT(b)      = perf model WITHOUT the prefill term
  decode capacity     = decode workers x b / TPOT x duty
  prefill capacity    = prefill workers x TP x peak x MFU / (2 x params) prompt tokens/s
  throughput          = min(decode capacity, prefill capacity x output / prompt)
  KV transfer         = prompt x KV bytes/token per request; adds to TTFT and must fit the link bandwidth

Run:  python -m planner.disagg --model Qwen/Qwen2.5-7B-Instruct --prompt 2048 --output 128 --tpot-sla 15 --gpus 4
"""
from __future__ import annotations

import argparse
import math

from .memory import estimate as mem_estimate
from .model_spec import resolve
from .perf import A100_SXM_80GB, CALIBRATION, DEFAULT_VERSION, Workload, step_breakdown
from .plan import ENGINE_CAP, KV_SPARE, SLA_HEADROOM, PlanRequest, plan

KV_LINK_GBPS = 50.0   # [assumption] effective NIXL GPU-to-GPU KV transfer rate; NVLink peak is 300 GB/s


def decode_tpot_p95(m, tp: int, b: float, w: Workload) -> float:
    c = CALIBRATION[DEFAULT_VERSION]
    parts = step_breakdown(m, tp, b, w)
    return (sum(parts.values()) - parts["prefill"]) * c["p95_factor"]


def prefill_tokens_per_s(m, tp: int) -> float:
    c = CALIBRATION[DEFAULT_VERSION]
    return tp * A100_SXM_80GB.peak_flops * c["mfu"] / (2 * m.total_params)


def best_split(model: str, prompt: int, output: int, tpot_sla: float, n_gpus: int,
               mem_util: float = 0.9, max_model_len: int = 4096) -> dict | None:
    m = resolve(model)
    w = Workload(prompt, output)
    c = CALIBRATION[DEFAULT_VERSION]
    limit = tpot_sla * SLA_HEADROOM
    kv_bpt = 2 * m.layers * m.kv_heads * m.head_dim * m.dtype_bytes          # whole request, all TP ranks
    best = None
    for tp_p in (1, 2, 4, 8):
        for tp_d in (1, 2, 4, 8):
            for n_p in range(1, n_gpus):
                for n_d in range(1, n_gpus):
                    if n_p * tp_p + n_d * tp_d > n_gpus:
                        continue
                    if m.heads % tp_p or m.heads % tp_d:
                        continue
                    mem = mem_estimate(m, tp=tp_d, mem_util=mem_util, max_model_len=max_model_len)
                    if not mem.fits or not mem_estimate(m, tp=tp_p, mem_util=mem_util, max_model_len=max_model_len).fits:
                        continue
                    kv_cap = int(mem.kv_tokens * (1 - KV_SPARE) // (prompt + output))
                    b = 0
                    for x in range(1, min(ENGINE_CAP, kv_cap) + 1):
                        if decode_tpot_p95(m, tp_d, x, w) <= limit:
                            b = x
                        else:
                            break
                    if b == 0:
                        continue
                    tpot = decode_tpot_p95(m, tp_d, b, w) / c["p95_factor"]
                    decode_cap = n_d * b / tpot * 1e3 * c["duty"]
                    prefill_cap = n_p * prefill_tokens_per_s(m, tp_p) * output / prompt
                    x_out = min(decode_cap, prefill_cap)
                    kv_gbps = x_out * prompt / output * kv_bpt / 1e9
                    if kv_gbps > KV_LINK_GBPS * n_p:
                        continue
                    gpus = n_p * tp_p + n_d * tp_d
                    cand = {"prefill": f"{n_p}x TP={tp_p}", "decode": f"{n_d}x TP={tp_d}", "gpus": gpus,
                            "decode_concurrency_per_worker": b, "throughput_tok_s": round(x_out),
                            "bottleneck": "decode" if decode_cap <= prefill_cap else "prefill",
                            "decode_tpot_p95_ms": round(decode_tpot_p95(m, tp_d, b, w), 2),
                            "kv_transfer_ms_per_request": round(prompt * kv_bpt / (KV_LINK_GBPS * 1e9) * 1e3, 1),
                            "kv_link_gbps": round(kv_gbps, 1)}
                    if best is None or (cand["throughput_tok_s"], -gpus) > (best["throughput_tok_s"], -best["gpus"]):
                        best = cand
    return best


def compare(model: str, prompt: int, output: int, tpot_sla: float, n_gpus: int, mem_util: float = 0.9) -> dict:
    agg = plan(PlanRequest(model, prompt, output, tpot_sla, list(range(n_gpus)), gpu_mem_util=mem_util))
    dis = best_split(model, prompt, output, tpot_sla, n_gpus, mem_util)
    a = agg["chosen"]["throughput_tok_s"] if agg["status"] == "ok" else 0
    d = dis["throughput_tok_s"] if dis else 0
    return {"aggregated": agg["chosen"]["name"] if a else None, "aggregated_tok_s": a,
            "disaggregated": dis, "gain": round(d / a, 2) if a else math.inf,
            "recommend": "disaggregate" if d > 1.15 * a else "aggregated",
            "status": "prediction only: no disaggregated run validated on gpu-02 yet"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--output", type=int, required=True)
    ap.add_argument("--tpot-sla", type=float, required=True)
    ap.add_argument("--gpus", type=int, required=True, help="number of GPUs")
    ap.add_argument("--mem-util", type=float, default=0.9)
    a = ap.parse_args()
    r = compare(a.model, a.prompt, a.output, a.tpot_sla, a.gpus, a.mem_util)
    print(f"aggregated   : {r['aggregated']}  {r['aggregated_tok_s']:,.0f} tok/s")
    d = r["disaggregated"]
    if d:
        print(f"disaggregated: prefill {d['prefill']} + decode {d['decode']} on {d['gpus']} GPUs  "
              f"{d['throughput_tok_s']:,} tok/s  (bottleneck {d['bottleneck']}, decode TPOT p95 "
              f"{d['decode_tpot_p95_ms']} ms, KV transfer ~{d['kv_transfer_ms_per_request']} ms/request)")
    print(f"gain {r['gain']}x -> {r['recommend']}   [{r['status']}]")


if __name__ == "__main__":
    main()
