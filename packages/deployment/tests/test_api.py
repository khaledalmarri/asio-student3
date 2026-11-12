from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from asio_deployment.app import create_app

from .conftest import Env, make_request


@pytest.fixture
def client(env: Env, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ASIO_AUTH_MODE", "development")
    return TestClient(create_app(env.controller))


def _body(ref: str = "s3-qwen-deployment-001") -> dict:
    return make_request(ref).model_dump(mode="json")


def test_post_is_202_with_status_url_and_replay(client: TestClient) -> None:
    r = client.post("/api/v1/deployments", json=_body(), headers={"X-Request-ID": "req-1"})
    assert r.status_code == 202, r.text
    b = r.json()
    assert b["observed_state"] == "PENDING" and b["status_url"] == f"/api/v1/deployments/{b['id']}"
    assert b["replayed"] is False
    again = client.post("/api/v1/deployments", json=_body())
    assert again.status_code == 202 and again.json()["id"] == b["id"] and again.json()["replayed"]
    got = client.get(b["status_url"])
    assert got.status_code == 200 and got.json()["pool_id"].startswith("pool_")


def test_errors_use_the_standard_envelope(client: TestClient) -> None:
    r = client.get("/api/v1/deployments/nope", headers={"X-Request-ID": "trace-9"})
    assert r.status_code == 404
    assert r.json() == {
        "error": {
            "code": "DEPLOYMENT_NOT_FOUND",
            "message": "deployment 'nope' not found",
            "details": {"deployment_id": "nope"},
            "request_id": "trace-9",
        }
    }
    assert r.headers["X-Request-ID"] == "trace-9"
    bad = client.post("/api/v1/deployments", json={"request_ref": "x"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "INVALID_REQUEST"


def test_student_cannot_deploy(client: TestClient) -> None:
    r = client.post("/api/v1/deployments", json=_body(), headers={"X-ASIO-Caller": "stu-7:student"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "FORBIDDEN"


def test_no_caller_without_dev_mode_is_401(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASIO_AUTH_MODE", raising=False)
    c = TestClient(create_app(env.controller))
    assert c.post("/api/v1/deployments", json=_body()).status_code == 401


def test_drain_and_stop_return_202_with_job(client: TestClient, env: Env) -> None:
    dep = client.post("/api/v1/deployments", json=_body()).json()["id"]
    env.drain_jobs()
    r = client.post(f"/api/v1/deployments/{dep}/drain", json={"drain_timeout_s": 5})
    assert r.status_code == 202 and r.json()["job_id"].startswith("job_")
    same = client.post(f"/api/v1/deployments/{dep}/stop")
    assert same.json()["job_id"] == r.json()["job_id"]
    env.drain_jobs()
    assert client.get(f"/api/v1/deployments/{dep}").json()["observed_state"] == "STOPPED"
