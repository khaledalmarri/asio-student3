"""Fit the Step 4b performance model on part of results.jsonl and test it on the rest.

Run:  python -m planner.fit_perf            (needs numpy + scipy; the model itself does not)

Fit set:  Qwen2.5-0.5B TP1, Qwen2.5-7B TP1, Qwen2.5-7B TP4      (two model sizes, the TP extremes)
Test set: Qwen2.5-0.5B TP2, Qwen2.5-7B TP2 (TP interpolation), Qwen2.5-0.5B DP2 (replicas, b/2 each)
Target:   TPOT p50 per concurrency level, concurrency <= 128 (above that one engine degrades; not modelled).
"""
from __future__ import annotations

import json
from pathlib import Path

from .model_spec import QWEN25_05B, QWEN25_7B
from .perf import CALIBRATION, Workload, replica_load, throughput, tpot_ms

MODELS = {"Qwen/Qwen2.5-0.5B-Instruct": QWEN25_05B, "Qwen/Qwen2.5-7B-Instruct": QWEN25_7B}
# label -> (tp, dp)
CONFIGS = {
    "qwen0.5b-tp1-v2": (1, 1), "qwen7b-tp1-v2": (1, 1), "qwen7b-tp4-v2": (4, 1),
    "qwen0.5b-tp2-v2": (2, 1), "qwen7b-tp2-v2": (2, 1), "qwen0.5b-dp2-v2": (1, 2),
}
FIT = {"qwen0.5b-tp1-v2", "qwen7b-tp1-v2", "qwen7b-tp4-v2"}
KEYS = ["eta", "fixed_ms", "allreduce_us", "per_request_us", "mfu"]   # fitted on TPOT; then "duty" on throughput


def load(path: str = "results/results.jsonl") -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["label"] in CONFIGS and r["concurrency"] <= 128 * CONFIGS[r["label"]][1]:
            tp, dp = CONFIGS[r["label"]]
            r.update(tp=tp, dp=dp, b=replica_load(r["concurrency"], dp)[0], spec=MODELS[r["model"]],
                     wl=Workload(r["prompt_tokens"], r["output_tokens"]))
            rows.append(r)
    return rows


def predict(r: dict, params: dict) -> tuple[float, float]:
    t = tpot_ms(r["spec"], r["tp"], r["b"], r["wl"], params=params)
    x = throughput(r["spec"], r["tp"], r["concurrency"], r["wl"], dp=r["dp"], params=params)
    return t, x


def fit(rows: list[dict]) -> dict:
    import numpy as np
    from scipy.optimize import least_squares

    train = [r for r in rows if r["label"] in FIT]

    def resid(v):
        p = dict(zip(KEYS, v), duty=1.0)
        return [np.log(predict(r, p)[0] / r["tpot_p50_ms"]) for r in train]

    x0 = [0.77, 1.4, 7.0, 10.0, 0.5]
    res = least_squares(resid, x0, bounds=([0.3, 0, 0, 0, 0.05], [1.0, 10, 100, 200, 1.0]))
    p = {k: round(float(v), 3 if k in ("eta", "mfu") else 2) for k, v in zip(KEYS, res.x)}
    # duty factor: geometric-mean ratio of measured to steady-state throughput on the fit set
    p["duty"] = 1.0
    ratios = [r["throughput_tok_s"] / predict(r, p)[1] for r in train]
    p["duty"] = round(float(np.exp(np.mean(np.log(ratios)))), 3)
    return p


def report(rows: list[dict], params: dict) -> None:
    print(f"{'config':18} {'conc':>4}  {'TPOT p50':>8} {'pred':>6} {'err':>6}   {'tok/s':>7} {'pred':>7} {'err':>6}  set")
    for split in ("fit", "test"):
        errs_t, errs_x = [], []
        for r in rows:
            if (r["label"] in FIT) != (split == "fit"):
                continue
            t, x = predict(r, params)
            et = (t / r["tpot_p50_ms"] - 1) * 100
            ex = (x / r["throughput_tok_s"] - 1) * 100
            errs_t.append(abs(et)); errs_x.append(abs(ex))
            print(f"{r['label']:18} {r['concurrency']:>4}  {r['tpot_p50_ms']:>8.2f} {t:>6.2f} {et:>+5.0f}%   "
                  f"{r['throughput_tok_s']:>7,.0f} {x:>7,.0f} {ex:>+5.0f}%  {split}")
        print(f"  {split}: mean |error| TPOT {sum(errs_t)/len(errs_t):.1f}%, throughput {sum(errs_x)/len(errs_x):.1f}%"
              f", worst TPOT {max(errs_t):.0f}%\n")


if __name__ == "__main__":
    data = load()
    p = fit(data)
    print("fitted:", p, "\n(stored calibration:", CALIBRATION["0.31"], ")\n")
    report(data, p)
