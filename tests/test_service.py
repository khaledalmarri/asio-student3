import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient

import service.app as svc
from tests.test_topology import fake_smi

client = TestClient(svc.app)
BODY = {"model": "Qwen/Qwen2.5-7B-Instruct", "workload": {"prompt_tokens": 512, "output_tokens": 128},
        "sla": {"tpot_p95_ms": 15}, "allocation": {"gpu_ids": [0, 1, 4, 5], "gpu_mem_util": 0.3}}


def test_plan_endpoint_follows_contract():
    r = client.post("/parallelism/plan", json=BODY)
    assert r.status_code == 200
    ch = r.json()["chosen"]
    assert ch["tp"] == 4 and "--max-num-seqs" in ch["vllm_args"] and r.json()["explanation"]


def test_plan_validation_errors():
    assert client.post("/parallelism/plan", json={**BODY, "sla": {"tpot_p95_ms": -1}}).status_code == 422
    assert client.post("/parallelism/plan", json={**BODY, "model": "unknown/model"}).status_code == 422


def test_deploy_dry_run_and_token(monkeypatch, tmp_path):
    snap = tmp_path / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setattr(svc.deploy, "HF_HOST", str(tmp_path))
    monkeypatch.setattr(svc.deploy.resolve_snapshot, "__defaults__", (str(tmp_path),))
    r = client.post("/parallelism/deploy", json=BODY)
    assert r.status_code == 200 and r.json()["status"] == "planned"
    assert r.json()["spec"]["max_concurrency_per_engine"] == 97
    monkeypatch.setenv("ASIO_TOKEN", "s3cret")
    assert client.post("/parallelism/deploy", json={**BODY, "launch": True}).status_code == 401
    assert client.delete("/parallelism/deployments/x", headers={"X-ASIO-Token": "s3cret"}).status_code == 403


def test_capabilities_and_topology(monkeypatch):
    caps = client.get("/parallelism/capabilities").json()
    assert {c["strategy"]: c["validated"] for c in caps}["Tensor parallel (TP)"] is True
    monkeypatch.setattr(svc.topology, "_smi", fake_smi)
    monkeypatch.setattr(svc.topology.discover, "__defaults__", (fake_smi,))
    t = client.get("/parallelism/topology?min_free_mib=25000").json()
    assert t["free_gpus"] == [0, 1, 4, 5] and t["nvlink"]["p2p_all_pairs_ok"]


class _FakeVLLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(body["max_tokens"]):
            self.wfile.write(b'data: {"choices":[{"text":"x"}]}\n\n')
        self.wfile.write(f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": body["max_tokens"]}})}\n\n'.encode())
        self.wfile.write(b"data: [DONE]\n\n")


def test_benchmark_job_end_to_end():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    r = client.post("/parallelism/benchmark", json={"url": url, "model": "m", "label": "t", "output_tokens": 16,
                                                    "concurrency": [1, 4], "procs": 2})
    assert r.status_code == 202
    job = r.json()["job_id"]
    for _ in range(100):
        st = client.get(f"/parallelism/benchmark/{job}").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    srv.shutdown()
    assert st["status"] == "done", st.get("log")
    assert [row["concurrency"] for row in st["rows"]] == [1, 4] and all(row["failed"] == 0 for row in st["rows"])
    assert "asio_parallelism_benchmarks_total 1" in client.get("/metrics").text
