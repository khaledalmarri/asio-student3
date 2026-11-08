from planner.check_decisions import main as decisions
from planner.model_spec import QWEN25_05B, QWEN25_7B
from planner.perf import Workload, tpot_ms

W = Workload(512, 128)


def test_tp_pays_off_for_7b_not_for_05b():
    assert tpot_ms(QWEN25_7B, 2, 1, W) < 0.7 * tpot_ms(QWEN25_7B, 1, 1, W)
    assert tpot_ms(QWEN25_05B, 2, 1, W) > 0.95 * tpot_ms(QWEN25_05B, 1, 1, W)


def test_model_reproduces_measured_layout_decisions():
    assert decisions()
