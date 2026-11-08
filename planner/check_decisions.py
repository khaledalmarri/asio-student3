"""Does the performance model make the same layout decisions as the measurements?

For each SLA, pick the largest benchmarked concurrency level whose TPOT p95 meets it, once from measured
data and once from the model, then rank layouts by tokens/s per GPU. Run: python -m planner.check_decisions
"""
import json
from pathlib import Path

from .model_spec import QWEN25_05B, QWEN25_7B
from .perf import Workload, replica_load, throughput, tpot_p95_ms

LEVELS = [1, 8, 32, 64, 128, 256]   # total concurrency levels benchmarked
CASES = [  # (model spec, label prefix, SLA ms, {name: (label, tp, dp, gpus)})
    (QWEN25_7B, 15, {"TP=1": ("qwen7b-tp1-v2", 1, 1, 1), "TP=2": ("qwen7b-tp2-v2", 2, 1, 2), "TP=4": ("qwen7b-tp4-v2", 4, 1, 4)}),
    (QWEN25_7B, 30, {"TP=1": ("qwen7b-tp1-v2", 1, 1, 1), "TP=2": ("qwen7b-tp2-v2", 2, 1, 2), "TP=4": ("qwen7b-tp4-v2", 4, 1, 4)}),
    (QWEN25_05B, 5, {"TP=1": ("qwen0.5b-tp1-v2", 1, 1, 1), "TP=2": ("qwen0.5b-tp2-v2", 2, 1, 2), "DP=2": ("qwen0.5b-dp2-v2", 1, 2, 2)}),
]


def main():
    rows = [json.loads(l) for l in Path("results/results.jsonl").read_text().splitlines()]
    by = {(r["label"], r["concurrency"]): r for r in rows}
    w = Workload(512, 128)
    agree = 0
    for m, sla, cfgs in CASES:
        print(f"\n{m.name}, TPOT p95 <= {sla} ms   (tok/s per GPU at the largest compliant level)")
        meas, pred = {}, {}
        for name, (label, tp, dp, g) in cfgs.items():
            levels = [c for c in LEVELS if (label, c) in by and replica_load(c, dp)[0] <= 128]
            ok = [c for c in levels if by[(label, c)]["tpot_p95_ms"] <= sla]
            mc = max(ok) if ok else 0
            meas[name] = by[(label, mc)]["throughput_tok_s"] / g if mc else 0
            okp = [c for c in levels if tpot_p95_ms(m, tp, replica_load(c, dp)[0], w) <= sla]
            pc = max(okp) if okp else 0
            pred[name] = throughput(m, tp, pc, w, dp=dp) / g if pc else 0
            print(f"  {name:5}  measured: level {mc:4} -> {meas[name]:7,.0f}/GPU   "
                  f"model: level {pc:4} -> {pred[name]:7,.0f}/GPU")
        rank_m = sorted(meas, key=meas.get, reverse=True)
        rank_p = sorted(pred, key=pred.get, reverse=True)
        same = rank_m[0] == rank_p[0] or abs(meas[rank_p[0]] - meas[rank_m[0]]) / meas[rank_m[0]] < 0.05
        agree += same
        print(f"  ranking measured {rank_m}  model {rank_p}  -> {'SAME DECISION' if same else 'DIFFERENT'}"
              f"{' (tie within 5% of measured best)' if same and rank_m[0] != rank_p[0] else ''}")
    print(f"\n{agree}/{len(CASES)} decisions agree")
    return agree == len(CASES)


if __name__ == "__main__":
    main()
