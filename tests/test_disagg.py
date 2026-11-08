from planner.disagg import compare

M = "Qwen/Qwen2.5-7B-Instruct"


def test_long_prompts_favour_disaggregation_long_answers_do_not():
    rag = compare(M, 2048, 128, 15, 4, mem_util=0.3)
    chat = compare(M, 128, 512, 15, 4, mem_util=0.3)
    assert rag["recommend"] == "disaggregate" and rag["gain"] > 1.3
    assert chat["recommend"] == "aggregated"
    assert "prediction only" in rag["status"]
