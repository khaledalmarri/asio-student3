"""Step 5: the parallelism planner.

enumerate (TP, PP, DP) -> filter (architecture rules, memory model) -> score (performance model under the SLA)
-> choose -> explain.

Run:  python -m planner.plan --model Qwen/Qwen2.5-7B-Instruct --prompt 512 --output 128 \
          --tpot-sla 15 --gpus 0,1,4,5 [--target-concurrency 64] [--json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

from .memory import estimate as mem_estimate
from .model_spec import KNOWN_MODELS, ModelSpec, resolve
from .perf import Workload, replica_load, throughput, tpot_p95_ms

ENGINE_CAP = 128          # max concurrent requests per engine (single engines lose throughput above this, Step 3b)
KV_SPARE = 0.10           # memory model untested outside its validation set: keep 10% spare KV
NEAR_TIE = 0.05           # closer than this (≈ 2x run-to-run noise) = "benchmark to decide"
SLA_HEADROOM = 0.85       # plan to 85% of the TPOT SLA. Was 0.90; the first end-to-end test (7B TP4 @106) measured
                          # TPOT p95 15.53 ms vs 13.45 predicted (x1.155) and missed a 15 ms SLA. 1/1.155 = 0.866.
PP_TPOT_PENALTY = 1.34    # measured once: Qwen-0.5B PP=2 vs TP=1 single-request TPOT (2.62 / 1.95 ms)
CALIBRATED_MAX_PARAMS = 7.7e9  # largest model in the performance-model fit/test set (Qwen2.5-7B)


@dataclass
class PlanRequest:
    model: str
    prompt_tokens: int
    output_tokens: int
    tpot_p95_ms: float
    gpus: list[int]                        # free GPU ids (from Student 1's allocator)
    target_concurrency: int | None = None  # requests in flight the deployment must sustain (from Student 2)
    max_model_len: int = 4096
    gpu_mem_util: float = 0.9
    gpu_mem_gib: float = 80.0
    vllm_version: str = "0.31"


@dataclass
class Candidate:
    tp: int
    pp: int
    dp: int
    gpus: int
    status: str = "scored"                 # scored | rejected
    reason: str = ""
    kv_tokens_per_replica: int = 0
    concurrency_per_replica: int = 0       # largest per-engine concurrency meeting the SLA
    concurrency_total: int = 0
    tpot_p95_ms: float = 0.0
    throughput_tok_s: float = 0.0
    per_gpu_tok_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"TP={self.tp} PP={self.pp} DP={self.dp}"


def enumerate_layouts(n_gpus: int) -> list[tuple[int, int, int]]:
    out = []
    for tp in (1, 2, 4, 8):
        for pp in (1, 2, 4):
            for dp in range(1, n_gpus + 1):
                if tp * pp * dp <= n_gpus:
                    out.append((tp, pp, dp))
    return out


def architecture_reason(m: ModelSpec, tp: int, pp: int) -> str:
    if m.heads % tp:
        return f"{m.heads} attention heads are not divisible by TP={tp}"
    if m.kv_heads % tp and tp % m.kv_heads:
        return f"{m.kv_heads} KV heads cannot be split or replicated evenly across TP={tp}"
    if pp > m.layers:
        return f"PP={pp} exceeds {m.layers} layers"
    return ""


def score(req: PlanRequest, m: ModelSpec, tp: int, pp: int, dp: int) -> Candidate:
    c = Candidate(tp, pp, dp, tp * pp * dp)
    w = Workload(req.prompt_tokens, req.output_tokens)

    why = architecture_reason(m, tp, pp)
    if why:
        c.status, c.reason = "rejected", why
        return c

    mem = mem_estimate(m, tp=tp, pp=pp, gpu_mem_gib=req.gpu_mem_gib, mem_util=req.gpu_mem_util,
                       max_model_len=req.max_model_len, vllm_version=req.vllm_version)
    c.kv_tokens_per_replica = mem.kv_tokens
    if not mem.fits:
        c.status, c.reason = "rejected", f"memory: {mem.reason}"
        return c
    kv_cap = int(mem.kv_tokens * (1 - KV_SPARE) // (req.prompt_tokens + req.output_tokens))
    if kv_cap < 1:
        c.status, c.reason = "rejected", "memory: KV cache cannot hold one request with 10% spare"
        return c

    penalty = PP_TPOT_PENALTY if pp > 1 else 1.0
    cap = min(ENGINE_CAP, kv_cap)
    limit = req.tpot_p95_ms * SLA_HEADROOM
    best = 0
    for b in range(1, cap + 1):
        if tpot_p95_ms(m, tp, b, w) * penalty <= limit:
            best = b
        else:
            break
    if best == 0:
        c.status = "rejected"
        c.reason = (f"SLA: even one request takes {tpot_p95_ms(m, tp, 1, w) * penalty:.1f} ms/token "
                    f"(p95) > {limit:.1f} ms (85% of the {req.tpot_p95_ms} ms SLA)")
        return c

    c.concurrency_per_replica = best
    c.concurrency_total = best * dp
    c.tpot_p95_ms = round(tpot_p95_ms(m, tp, best, w) * penalty, 2)
    c.throughput_tok_s = round(throughput(m, tp, best * dp, w, dp=dp) / penalty, 0)
    c.per_gpu_tok_s = round(c.throughput_tok_s / c.gpus, 0)
    if best == kv_cap and kv_cap < ENGINE_CAP:
        c.notes.append(f"concurrency limited by KV cache ({mem.kv_tokens:,} tokens per replica)")
    if best == ENGINE_CAP:
        c.notes.append(f"concurrency capped at {ENGINE_CAP} per engine (measured throughput drop above it)")
    if pp > 1:
        c.notes.append(f"PP speed is an estimate: TP model x {PP_TPOT_PENALTY} (one measurement)")
    return c


def plan(req: PlanRequest) -> dict:
    m = resolve(req.model)
    n = len(req.gpus)
    cands = [score(req, m, *lay) for lay in enumerate_layouts(n)]
    ok = [c for c in cands if c.status == "scored"]
    explanation, warnings = [], []

    if m.total_params > 2 * CALIBRATED_MAX_PARAMS:
        warnings.append(f"{m.name} ({m.total_params / 1e9:.0f}B params) is outside the performance model's calibration "
                        f"range (≤ {CALIBRATED_MAX_PARAMS / 1e9:.1f}B): speed predictions are extrapolated; benchmark before relying on them")

    # PP only when no TP-only layout fits: PP was slower for latency in every measurement.
    if any(c.pp == 1 for c in ok):
        for c in ok:
            if c.pp > 1:
                c.status, c.reason = "rejected", "PP not needed: a TP-only layout fits, and PP measured slower per token"
        ok = [c for c in ok if c.pp == 1]

    if req.target_concurrency:
        meets = [c for c in ok if c.concurrency_total >= req.target_concurrency]
        if meets:
            fewest = min(c.gpus for c in meets)
            pool = [c for c in meets if c.gpus == fewest]
            pool.sort(key=lambda c: (-c.per_gpu_tok_s, c.tp))
            for c in ok:
                if c not in meets:
                    c.status, c.reason = "rejected", (f"capacity: holds {c.concurrency_total} concurrent requests under the SLA, "
                                                      f"target is {req.target_concurrency}")
                elif c.gpus > fewest:
                    c.status, c.reason = "rejected", f"uses {c.gpus} GPUs; the target fits on {fewest}"
            objective = f"meet {req.target_concurrency} concurrent requests with the fewest GPUs, then most tokens/s per GPU"
        else:
            pool = sorted(ok, key=lambda c: -c.concurrency_total)
            warnings.append(f"no layout on {n} GPUs reaches {req.target_concurrency} concurrent requests under "
                            f"{req.tpot_p95_ms} ms TPOT p95; best is {pool[0].concurrency_total if pool else 0}")
            objective = "target unreachable: maximise SLA-compliant concurrency"
    else:
        pool = sorted(ok, key=lambda c: (-c.throughput_tok_s, -c.per_gpu_tok_s, c.tp))
        objective = f"maximise SLA-compliant throughput on the {n} GPUs given"

    if not pool:
        return {"status": "infeasible", "model": m.name, "objective": objective, "warnings": warnings,
                "explanation": ["no layout passes the rules, memory and SLA checks"],
                "rejected": [asdict(c) | {"name": c.name} for c in cands]}

    best = pool[0]
    key = (lambda c: c.per_gpu_tok_s) if req.target_concurrency else (lambda c: c.throughput_tok_s)
    ties = [c for c in pool[1:] if key(best) and (key(best) - key(c)) / key(best) < NEAR_TIE]

    explanation.append(f"Objective: {objective}.")
    explanation.append(
        f"Chosen {best.name} on {best.gpus} GPU(s): up to {best.concurrency_per_replica} requests per engine "
        f"({best.concurrency_total} total) at a predicted TPOT p95 of {best.tpot_p95_ms} ms "
        f"(planned to 85% of the {req.tpot_p95_ms} ms SLA), {best.throughput_tok_s:,.0f} tok/s "
        f"({best.per_gpu_tok_s:,.0f} per GPU).")
    for note in best.notes:
        explanation.append(f"Note: {note}.")
    for c in pool[1:3]:
        diff = (key(best) - key(c)) / key(best) * 100 if key(best) else 0
        explanation.append(f"Runner-up {c.name}: {c.throughput_tok_s:,.0f} tok/s ({c.per_gpu_tok_s:,.0f} per GPU), "
                           f"{diff:.0f}% lower on the objective.")
    if ties:
        explanation.append("Near-tie (< 5%, within ~2x measured run-to-run noise) with "
                           + ", ".join(c.name for c in ties) + ": benchmark both before committing.")
    by_reason: dict[str, list[str]] = {}
    for c in cands:
        if c.status == "rejected":
            by_reason.setdefault(c.reason, []).append(c.name)
    for reason, names in by_reason.items():
        explanation.append(f"Rejected {', '.join(names)}: {reason}.")

    gpu_groups = [req.gpus[i * best.tp * best.pp:(i + 1) * best.tp * best.pp] for i in range(best.dp)]
    args = ["--tensor-parallel-size", str(best.tp), "--pipeline-parallel-size", str(best.pp),
            "--gpu-memory-utilization", str(req.gpu_mem_util), "--max-model-len", str(req.max_model_len),
            "--max-num-seqs", str(best.concurrency_per_replica)]
    if best.dp > 1:
        args += ["--data-parallel-size", str(best.dp)]
    return {
        "status": "ok",
        "model": m.name,
        "objective": objective,
        "chosen": asdict(best) | {"name": best.name, "gpu_ids": [g for grp in gpu_groups for g in grp],
                                  "gpu_groups": gpu_groups, "vllm_args": args},
        "alternatives": [asdict(c) | {"name": c.name} for c in pool[1:4]],
        "near_ties": [c.name for c in ties],
        "rejected": [{"name": c.name, "reason": c.reason} for c in cands if c.status == "rejected"],
        "warnings": warnings,
        "explanation": explanation,
        "calibration": {"vllm_version": req.vllm_version, "gpu": "A100-SXM4-80GB", "node": "gpu-02"},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ASIO Student 3 parallelism planner")
    ap.add_argument("--model", required=True, help="known name (" + ", ".join(KNOWN_MODELS) + ") or path to config.json")
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--output", type=int, required=True)
    ap.add_argument("--tpot-sla", type=float, required=True, help="TPOT p95 limit, ms")
    ap.add_argument("--gpus", required=True, help="free GPU ids, e.g. 0,1,4,5")
    ap.add_argument("--target-concurrency", type=int)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--mem-util", type=float, default=0.9)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    req = PlanRequest(a.model, a.prompt, a.output, a.tpot_sla, [int(g) for g in a.gpus.split(",")],
                      a.target_concurrency, a.max_model_len, a.mem_util)
    out = plan(req)
    if a.json:
        print(json.dumps(out, indent=2))
        return
    for w in out["warnings"]:
        print("WARNING:", w)
    for line in out["explanation"]:
        print("-", line)
    if out["status"] == "ok":
        print("\nvllm serve <model>", " ".join(out["chosen"]["vllm_args"]), "  # GPUs", out["chosen"]["gpu_ids"])


if __name__ == "__main__":
    main()
