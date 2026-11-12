from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from asio_parallelism.api import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _body(**over: object) -> dict:
    b: dict = {
        "model_id": "qwen-demo",
        "hf_model": "Qwen/Qwen2.5-7B-Instruct",
        "workload": {"prompt_tokens": 512, "output_tokens": 128},
        "sla": {"tpot_p95_ms": 15},
        "gpu_count": 4,
        "gpu_memory_utilization": 0.3,
    }
    b.update(over)
    return b


def test_plan_is_feasible_and_ready_for_a_deployment(client: TestClient) -> None:
    r = client.post("/api/v1/parallelism/plan", json=_body())
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["feasible"] and p["evidence_label"] == "SIMULATED"
    assert p["tp"] * p["pp"] * p["dp"] == p["gpu_count"] <= 4
    t = p["deployment_template"]
    args = t["vllm_args"]
    assert args[args.index("--max-num-seqs") + 1] == str(t["max_num_seqs"])
    assert not {"--port", "--host", "--served-model-name"} & set(args)
    assert t["rated_capacity_rps"] > 0 and t["image_profile"] == "vllm-0.31-a100"
    assert p["gpu_constraints"]["count"] == p["gpu_count"]
    assert "gpu_ids" not in str(p)  # S1 picks the GPUs, the plan never names them
    got = client.get(f"/api/v1/parallelism/plans/{p['id']}")
    assert got.status_code == 200 and got.json()["id"] == p["id"]


def test_target_concurrency_uses_fewest_gpus(client: TestClient) -> None:
    r = client.post("/api/v1/parallelism/plan", json=_body(target_concurrency=8))
    assert r.status_code == 200 and r.json()["gpu_count"] == 1


def test_infeasible_is_422_with_reasons(client: TestClient) -> None:
    r = client.post(
        "/api/v1/parallelism/plan",
        json=_body(hf_model="meta-llama/Llama-3.1-70B-Instruct", gpu_count=1),
    )
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "GPU_CONFIGURATION_INFEASIBLE" and e["details"]["explanation"]


def test_unknown_model_is_422(client: TestClient) -> None:
    r = client.post("/api/v1/parallelism/plan", json=_body(hf_model="/nope/config.json"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "MODEL_UNKNOWN"


def test_moe_plan_id_is_kept_with_a_warning(client: TestClient) -> None:
    p = client.post("/api/v1/parallelism/plan", json=_body(moe_placement_plan_id="moe-1")).json()
    assert p["moe_placement_plan_id"] == "moe-1" and p["ep"] == 1
    assert any("expert parallelism" in w for w in p["warnings"])


def test_benchmarks_are_honest_stubs(client: TestClient) -> None:
    r = client.post("/api/v1/parallelism/benchmarks")
    assert r.status_code == 501 and r.json()["error"]["code"] == "NOT_IMPLEMENTED"
