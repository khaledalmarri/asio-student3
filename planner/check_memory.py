"""Compare the memory model with every KV-cache size measured on gpu-02.

Run:  python -m planner.check_memory
"""
from .memory import estimate
from .model_spec import LLAMA31_8B, LLAMA31_70B, QWEN25_05B, QWEN25_7B

# (model, tp, pp, mem_util, measured weights GiB/GPU or None, measured KV tokens, vLLM version, used_for)
MEASURED = [
    (QWEN25_05B, 1, 1, 0.25, 0.93, 1_570_816, "0.31", "fit: base + slope"),
    (QWEN25_7B, 1, 1, 0.30, 14.29, 144_832, "0.31", "fit: base + slope"),
    (QWEN25_05B, 2, 1, 0.25, 0.46, 3_128_480, "0.31", "fit: TP extra"),
    (QWEN25_05B, 1, 2, 0.25, 0.59, 3_249_840, "0.31", "fit: PP factor"),
    (QWEN25_7B, 2, 1, 0.30, 7.16, 564_176, "0.31", "TEST (form chosen after seeing it)"),
    (QWEN25_7B, 1, 2, 0.30, 7.15, 568_048, "0.31", "TEST (pre-registered)"),
    (LLAMA31_8B, 1, 1, 0.90, None, 442_464, "0.8.5", "fit: base + slope"),
    (LLAMA31_8B, 2, 1, 0.90, None, 983_520, "0.8.5", "fit: TP extra"),
    (LLAMA31_70B, 4, 1, 0.90, None, 449_056, "0.8.5", "TEST"),
]


def main() -> list[float]:
    print(f"{'model':24} {'TP':>2} {'PP':>2} {'weights/GPU':>12} {'KV tokens pred':>15} {'measured':>11} {'error':>7}  role")
    errors = []
    for m, tp, pp, util, w_meas, kv_meas, src, role in MEASURED:
        e = estimate(m, tp=tp, pp=pp, mem_util=util, vllm_version=src)
        err = (e.kv_tokens - kv_meas) / kv_meas * 100
        errors.append(err)
        w = f"{e.weights_gib:.2f}" + (f" ({w_meas})" if w_meas else "")
        print(f"{m.name:24} {tp:>2} {pp:>2} {w:>12} {e.kv_tokens:>15,} {kv_meas:>11,} {err:>+6.1f}%  vLLM {src}, {role}")
    return errors


if __name__ == "__main__":
    main()
