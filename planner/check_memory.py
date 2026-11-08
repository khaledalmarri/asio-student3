"""Compare the memory model with every KV-cache size measured on gpu-02.

Run:  python -m planner.check_memory
"""
from .memory import estimate
from .model_spec import LLAMA31_8B, LLAMA31_70B, QWEN25_05B

# (model, tp, pp, mem_util, measured weights GiB/GPU or None, measured KV tokens, source, used_for)
MEASURED = [
    (QWEN25_05B, 1, 1, 0.25, 0.93, 1_570_816, "vLLM 0.31, 2026-11-08", "fit: overhead base"),
    (QWEN25_05B, 2, 1, 0.25, 0.46, 3_128_480, "vLLM 0.31, 2026-11-08", "fit: TP extra"),
    (QWEN25_05B, 1, 2, 0.25, 0.59, 3_249_840, "vLLM 0.31, 2026-11-08", "fit: PP factor"),
    (LLAMA31_8B, 1, 1, 0.90, None, 442_464, "vLLM 0.8.5, LoadCast Exp 9", "fit: hidden slope"),
    (LLAMA31_8B, 2, 1, 0.90, None, 983_520, "vLLM 0.8.5, LoadCast Exp 9", "TEST"),
    (LLAMA31_70B, 4, 1, 0.90, None, 449_056, "vLLM 0.8.5, LoadCast Exp 11", "TEST"),
]


def main() -> list[float]:
    print(f"{'model':24} {'TP':>2} {'PP':>2} {'weights/GPU':>12} {'KV tokens pred':>15} {'measured':>11} {'error':>7}  role")
    errors = []
    for m, tp, pp, util, w_meas, kv_meas, src, role in MEASURED:
        e = estimate(m, tp=tp, pp=pp, mem_util=util)
        err = (e.kv_tokens - kv_meas) / kv_meas * 100
        errors.append(err)
        w = f"{e.weights_gib:.2f}" + (f" ({w_meas})" if w_meas else "")
        print(f"{m.name:24} {tp:>2} {pp:>2} {w:>12} {e.kv_tokens:>15,} {kv_meas:>11,} {err:>+6.1f}%  {role}  [{src}]")
    return errors


if __name__ == "__main__":
    main()
