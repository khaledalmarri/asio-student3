from planner.plan import PlanRequest, plan


def test_7b_strict_sla_chooses_tp4_on_4_gpus():
    out = plan(PlanRequest("Qwen/Qwen2.5-7B-Instruct", 512, 128, 15, [0, 1, 4, 5], gpu_mem_util=0.3))
    assert out["chosen"]["tp"] == 4 and out["chosen"]["pp"] == 1


def test_small_model_with_small_target_returns_spare_gpu():
    out = plan(PlanRequest("Qwen/Qwen2.5-0.5B-Instruct", 512, 128, 5, [0, 1], target_concurrency=32, gpu_mem_util=0.25))
    assert out["chosen"]["gpus"] == 1 and out["chosen"]["gpu_ids"] == [0]


def test_70b_needs_tp_and_warns_about_extrapolation():
    out = plan(PlanRequest("meta-llama/Llama-3.1-70B-Instruct", 2048, 256, 30, list(range(8)), max_model_len=32768))
    assert out["chosen"]["tp"] >= 4
    assert any("calibration range" in w for w in out["warnings"])
    assert any("memory" in r["reason"] for r in out["rejected"])


def test_impossible_sla_is_reported_not_hidden():
    out = plan(PlanRequest("Qwen/Qwen2.5-7B-Instruct", 512, 128, 2, [0, 1]))
    assert out["status"] == "infeasible"
