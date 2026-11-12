"""GPU-pinned vLLM launch templates (S3 P0).

Only images listed in `PROFILES` can be launched (contract §2.2: workers accept allowlisted images
and S1-granted GPU UUIDs only). GPUs are passed to Docker by UUID, never by index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import Deployment

CONTAINER_PREFIX = "asio-s3-"
HF_IN_CONTAINER = "/hf"
CONTAINER_PORT = 8000


@dataclass(frozen=True)
class ImageProfile:
    """Everything one serving image needs. The quirks below were all found on gpu-02."""

    name: str
    image: str
    entrypoint: str = "vllm"
    env: dict[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()


PROFILES: dict[str, ImageProfile] = {
    # vLLM 0.31.0 built for CUDA 12.6 / A100 (driver 560 cannot run the official CUDA 13 wheel).
    # Runs as uid 1000 (cannot read /root), bakes in CUDA_VISIBLE_DEVICES=0,1, has no FlashInfer and
    # FlashAttention 2 only. Pin the image by digest via ASIO_S3_VLLM_IMAGE before the demo.
    "vllm-0.31-a100": ImageProfile(
        name="vllm-0.31-a100",
        image=os.environ.get("ASIO_S3_VLLM_IMAGE", "fbdaa5259f0a"),
        env={
            "HF_HOME": HF_IN_CONTAINER,
            "HF_HUB_CACHE": f"{HF_IN_CONTAINER}/hub",
            "HF_HUB_OFFLINE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        },
        extra_args=(
            "--attention-backend",
            "FLASH_ATTN",
            "--attention-config",
            '{"flash_attn_version":2}',
        ),
    ),
}


class TemplateError(Exception):
    """The deployment cannot be turned into a launch command (permanent)."""


@dataclass(frozen=True)
class LaunchSpec:
    name: str
    image: str
    entrypoint: str
    command: tuple[str, ...]
    gpu_uuids: tuple[str, ...]
    host_port: int
    env: dict[str, str]
    labels: dict[str, str]
    volumes: tuple[str, ...]

    def docker_args(self) -> list[str]:
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            self.name,
            "--ipc=host",
            "--gpus",
            f'"device={",".join(self.gpu_uuids)}"',
            "-p",
            f"{self.host_port}:{CONTAINER_PORT}",
        ]
        for k, v in sorted(self.labels.items()):
            args += ["--label", f"{k}={v}"]
        for vol in self.volumes:
            args += ["-v", vol]
        for k, v in sorted(self.env.items()):
            args += ["-e", f"{k}={v}"]
        return args + ["--entrypoint", self.entrypoint, self.image, *self.command]


def hf_root() -> str:
    return os.environ.get("ASIO_HF_HOME", "/root/.cache/huggingface")


def resolve_snapshot(hf_model: str, root: str | None = None) -> str:
    """Container path of the cached snapshot. Launches never download: missing weights fail fast."""
    base = Path(root or hf_root())
    snaps = base / "hub" / f"models--{hf_model.replace('/', '--')}" / "snapshots"
    found = (
        sorted(p for p in snaps.glob("*") if (p / "config.json").exists()) if snaps.exists() else []
    )
    if not found:
        raise TemplateError(f"{hf_model} is not cached under {base}; download it on the node first")
    return f"{HF_IN_CONTAINER}/hub/{found[0].relative_to(base / 'hub')}"


def _int_arg(args: list[str], flag: str, default: int = 1) -> int:
    if flag in args:
        i = args.index(flag)
        try:
            return int(args[i + 1])
        except (IndexError, ValueError) as e:
            raise TemplateError(f"bad value for {flag}") from e
    return default


def gpus_required(vllm_args: list[str]) -> int:
    return (
        _int_arg(vllm_args, "--tensor-parallel-size")
        * _int_arg(vllm_args, "--pipeline-parallel-size")
        * _int_arg(vllm_args, "--data-parallel-size")
    )


_FORBIDDEN_ARGS = {"--port", "--host", "--served-model-name", "--model"}


def build_launch_spec(d: Deployment, model_path: str | None = None) -> LaunchSpec:
    cfg = d.config
    profile = PROFILES.get(cfg.image_profile)
    if profile is None:
        raise TemplateError(f"image profile {cfg.image_profile!r} is not allowlisted")
    if not d.gpu_uuids:
        raise TemplateError("no S1-granted GPU UUIDs on the deployment")
    args = list(cfg.vllm_args)
    bad = _FORBIDDEN_ARGS.intersection(args)
    if bad:
        raise TemplateError(f"vllm_args may not set {sorted(bad)}; S3 owns them")
    need = gpus_required(args)
    if need != len(d.gpu_uuids):
        raise TemplateError(
            f"layout needs {need} GPU(s) (TP x PP x DP) but the allocation grants "
            f"{len(d.gpu_uuids)}"
        )
    dp = _int_arg(args, "--data-parallel-size")
    if dp > 1 and "--api-server-count" not in args:
        args += ["--api-server-count", str(dp)]  # measured: one API server per DP rank helps
    path = model_path or cfg.model_path or resolve_snapshot(cfg.hf_model or d.model_id)
    env = dict(profile.env)
    # Docker exposes the granted GPUs as 0..n-1 inside the container; override the image default.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(d.gpu_uuids)))
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    command = (
        "serve",
        path,
        "--served-model-name",
        cfg.served_model_name,
        *profile.extra_args,
        *args,
    )
    return LaunchSpec(
        name=f"{CONTAINER_PREFIX}{d.id}",
        image=profile.image,
        entrypoint=profile.entrypoint,
        command=command,
        gpu_uuids=tuple(d.gpu_uuids),
        host_port=cfg.port,
        env=env,
        labels={
            "asio.owner": "s3",
            "asio.deployment_id": d.id,
            "asio.allocation_id": d.allocation_id or "",
        },
        volumes=(f"{hf_root()}:{HF_IN_CONTAINER}:ro",),
    )
