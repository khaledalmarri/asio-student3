import json
import shlex

import pytest

from planner.deploy import PROFILES, compose_yaml, docker_run_cmd, resolve_snapshot, spec
from planner.plan import PlanRequest, plan

P = PROFILES["abujafar-0.31"]
M = "Qwen/Qwen2.5-7B-Instruct"
PATH = "/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"


def _plan(**kw):
    return plan(PlanRequest(M, 512, 128, 15, [0, 1, 4, 5], gpu_mem_util=0.3, **kw))


def test_docker_command_carries_plan_and_image_fixes():
    cmd = docker_run_cmd(_plan(), M, PATH, "x", 8401, P)
    s = shlex.join(cmd)
    assert '"device=0,1,4,5"' in cmd and "CUDA_VISIBLE_DEVICES=0,1,2,3" in s
    assert "--tensor-parallel-size 4" in s and "--max-num-seqs 97" in s
    assert "FLASH_ATTN" in s and "/root/.cache/huggingface:/hf:ro" in s
    assert cmd[cmd.index("--entrypoint") + 3:][:2] == ["serve", PATH]


def test_dp_plan_adds_api_servers():
    out = plan(PlanRequest("Qwen/Qwen2.5-0.5B-Instruct", 512, 128, 5, [0, 1], gpu_mem_util=0.25))
    assert out["chosen"]["dp"] == 2
    assert "--api-server-count 2" in shlex.join(docker_run_cmd(out, "m", "/hf/x", "x", 8401, P))


def test_compose_and_spec():
    out = _plan()
    y = compose_yaml(out, M, PATH, "svc", 8401, P)
    assert 'device_ids: ["0", "1", "4", "5"]' in y and "CUDA_VISIBLE_DEVICES" in y
    sp = spec(out, M, PATH, "svc", 8401, P)
    assert sp["max_concurrency_per_engine"] == 97 and sp["gpu_ids"] == [0, 1, 4, 5]
    json.dumps(sp)


def test_resolve_snapshot(tmp_path):
    snap = tmp_path / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert resolve_snapshot(M, str(tmp_path)) == "/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/abc"
    with pytest.raises(FileNotFoundError):
        resolve_snapshot("nobody/missing", str(tmp_path))
