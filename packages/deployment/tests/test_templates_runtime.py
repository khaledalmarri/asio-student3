from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from asio_deployment.contracts import Backend, Deployment, DeploymentConfig, ResourceClass
from asio_deployment.runtime import DockerRuntime, RuntimeError_, preflight
from asio_deployment.templates import (
    TemplateError,
    build_launch_spec,
    gpu_memory_utilization,
    gpus_required,
)

NOW = datetime(2026, 11, 12, tzinfo=UTC)


def _dep(args: list[str], uuids: list[str], **cfg: object) -> Deployment:
    return Deployment(
        id="dep_abc",
        owner_id="a",
        request_ref="r",
        resource_class=ResourceClass.GPU_SERVING,
        allocation_id="alloc-1",
        model_id="qwen-demo",
        backend=Backend.VLLM,
        pool_id="pool_abc",
        gpu_uuids=uuids,
        created_at=NOW,
        updated_at=NOW,
        config=DeploymentConfig(
            served_model_name="qwen-demo", port=8001, model_path="/hf/x", vllm_args=args, **cfg
        ),
    )  # type: ignore[arg-type]


def test_docker_args_pin_gpus_by_uuid_and_renumber_inside_container() -> None:
    spec = build_launch_spec(_dep(["--tensor-parallel-size", "2"], ["GPU-a", "GPU-b"]))
    a = spec.docker_args()
    assert a[a.index("--gpus") + 1] == '"device=GPU-a,GPU-b"'
    assert "CUDA_VISIBLE_DEVICES=0,1" in a and "-p" in a and "8001:8000" in a
    assert a[a.index("--served-model-name") + 1] == "qwen-demo"
    assert "asio.deployment_id=dep_abc" in a and spec.name == "asio-s3-dep_abc"


def test_dp_adds_one_api_server_per_rank() -> None:
    spec = build_launch_spec(_dep(["--data-parallel-size", "2"], ["GPU-a", "GPU-b"]))
    assert spec.command[-2:] == ("--api-server-count", "2")


@pytest.mark.parametrize(
    "args,uuids,msg",
    [
        (["--tensor-parallel-size", "4"], ["GPU-a", "GPU-b"], "needs 4"),
        (["--port", "9"], ["GPU-a"], "may not set"),
        ([], [], "no S1-granted"),
    ],
)
def test_template_refuses(args: list[str], uuids: list[str], msg: str) -> None:
    with pytest.raises(TemplateError, match=msg):
        build_launch_spec(_dep(args, uuids))


def test_only_allowlisted_images() -> None:
    with pytest.raises(TemplateError, match="not allowlisted"):
        build_launch_spec(_dep([], ["GPU-a"], image_profile="random/image:latest"))


def test_gpus_required() -> None:
    assert gpus_required(["--tensor-parallel-size", "2", "--pipeline-parallel-size", "2"]) == 4


class FakeShell:
    def __init__(self, outputs: dict[str, tuple[int, str]]) -> None:
        self.outputs = outputs
        self.cmds: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.cmds.append(list(cmd))
        key = " ".join(cmd[:2])
        code, out = self.outputs.get(key, (0, ""))
        return subprocess.CompletedProcess(list(cmd), code, out, "")


def test_docker_runtime_never_touches_other_containers() -> None:
    rt = DockerRuntime(run=FakeShell({}))
    with pytest.raises(RuntimeError_):
        rt.stop("abu-jafar-dynamo-frontend")


def test_verify_ignores_other_tenants_on_shared_gpu() -> None:
    shell = FakeShell(
        {
            "docker inspect": (1, ""),  # container gone
            "nvidia-smi --query-compute-apps=gpu_uuid,pid": (0, "GPU-a, 999\nGPU-b, 1234\n"),
        }
    )
    ok, ev = DockerRuntime(run=shell).verify_stopped("asio-s3-dep_abc", [77777777], ["GPU-a"])
    assert ok and ev["our_pids_still_on_gpu"] == [] and ev["container_removed"]


def test_verify_fails_if_our_process_still_holds_the_gpu() -> None:
    shell = FakeShell(
        {
            "docker inspect": (1, ""),
            "nvidia-smi --query-compute-apps=gpu_uuid,pid": (0, "GPU-a, 4242\n"),
        }
    )
    ok, ev = DockerRuntime(run=shell).verify_stopped("asio-s3-dep_abc", [4242], ["GPU-a"])
    assert not ok and ev["our_pids_still_on_gpu"] == [4242]


def test_verify_fails_closed_when_nvidia_smi_is_unavailable() -> None:
    shell = FakeShell(
        {"docker inspect": (1, ""), "nvidia-smi --query-compute-apps=gpu_uuid,pid": (9, "")}
    )
    ok, ev = DockerRuntime(run=shell).verify_stopped("asio-s3-dep_abc", [], ["GPU-a"])
    assert not ok and "gpu_check_error" in ev


def test_gpu_status_reads_memory_and_processes() -> None:
    shell = FakeShell(
        {
            "nvidia-smi --query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu": (
                0,
                "GPU-a, 81920, 61440, 20480, 99\nGPU-b, 81920, 0, 81920, 0\n"
                "GPU-c, 81920, 1, 2, 3\n",
            ),
            "nvidia-smi --query-compute-apps=gpu_uuid,pid": (0, "GPU-a, 999\n"),
        }
    )
    st = DockerRuntime(run=shell).gpu_status(["GPU-a", "GPU-b", "GPU-x"])
    assert set(st) == {"GPU-a", "GPU-b"}  # GPU-x is not on this node
    assert (st["GPU-a"].free_mib, st["GPU-a"].util_pct, st["GPU-a"].pids) == (20480, 99, [999])
    res = preflight(st, ["GPU-a", "GPU-b", "GPU-x"], 25600)
    assert not res.ok and res.permanent and len(res.problems) == 2
    assert preflight(st, ["GPU-b"], 25600).ok


def test_gpu_memory_utilization_parsing() -> None:
    assert gpu_memory_utilization([]) == 0.9
    assert gpu_memory_utilization(["--gpu-memory-utilization", "0.25"]) == 0.25
    with pytest.raises(TemplateError):
        gpu_memory_utilization(["--gpu-memory-utilization", "2"])
