from planner.check_memory import MEASURED, main
from planner.memory import estimate
from planner.model_spec import LLAMA31_70B, QWEN25_05B


def test_matches_every_measurement_within_5_percent():
    assert all(abs(e) < 5 for e in main())


def test_param_counts():
    assert round(QWEN25_05B.total_params / 1e9, 2) == 0.49
    assert round(LLAMA31_70B.total_params / 1e9, 1) == 70.6


def test_70b_needs_at_least_4_gpus_for_32k_context():
    assert not estimate(LLAMA31_70B, tp=1, max_model_len=32768).fits
    assert not estimate(LLAMA31_70B, tp=2, max_model_len=32768).fits
    assert estimate(LLAMA31_70B, tp=4, max_model_len=32768).fits


def test_pp_cannot_split_embedding():
    tp2 = estimate(QWEN25_05B, tp=2, mem_util=0.25)
    pp2 = estimate(QWEN25_05B, pp=2, mem_util=0.25)
    assert pp2.weights_gib > tp2.weights_gib
