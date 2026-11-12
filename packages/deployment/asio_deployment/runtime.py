"""Container runtime used by the deployment worker (the only process with the Docker socket).

Shutdown is *verified*, not assumed: before stopping, the worker records the host PIDs of the
container; afterwards it checks that the container is gone and that none of those PIDs still holds
memory on the granted GPU UUIDs (`nvidia-smi --query-compute-apps`). Other tenants' processes on a
shared GPU do not count against us, so this also works on a shared node.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .templates import CONTAINER_PREFIX, LaunchSpec

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=120)


@dataclass
class ProbeResult:
    ok: bool
    detail: str


class Runtime(Protocol):
    def start(self, spec: LaunchSpec) -> None: ...
    def exists(self, name: str) -> bool: ...
    def is_running(self, name: str) -> bool: ...
    def logs(self, name: str, tail: int = 40) -> str: ...
    def host_pids(self, name: str) -> list[int]: ...
    def stop(self, name: str) -> None: ...
    def verify_stopped(
        self, name: str, pids: list[int], gpu_uuids: list[str]
    ) -> tuple[bool, dict[str, Any]]: ...


class RuntimeError_(Exception):
    pass


class DockerRuntime:
    def __init__(self, run: Runner = _run) -> None:
        self.run = run

    @staticmethod
    def _ours(name: str) -> None:
        # Never touch a container this controller did not create (shared node rule).
        if not name.startswith(CONTAINER_PREFIX):
            raise RuntimeError_(f"refusing to act on non-S3 container {name!r}")

    def start(self, spec: LaunchSpec) -> None:
        self._ours(spec.name)
        args = spec.docker_args()
        # `--gpus "device=a,b"`: the quotes are for Docker's CSV parser, not the shell.
        p = self.run(args)
        if p.returncode != 0:
            raise RuntimeError_(f"docker run failed: {p.stderr.strip()[:500]}")

    def _inspect(self, name: str, fmt: str) -> str | None:
        p = self.run(["docker", "inspect", "-f", fmt, name])
        return p.stdout.strip() if p.returncode == 0 else None

    def exists(self, name: str) -> bool:
        return self._inspect(name, "{{.Id}}") is not None

    def is_running(self, name: str) -> bool:
        return self._inspect(name, "{{.State.Running}}") == "true"

    def logs(self, name: str, tail: int = 40) -> str:
        p = self.run(["docker", "logs", "--tail", str(tail), name])
        return (p.stdout + p.stderr)[-4000:]

    def host_pids(self, name: str) -> list[int]:
        p = self.run(["docker", "top", name, "-eo", "pid"])
        if p.returncode != 0:
            return []
        return [int(x) for x in p.stdout.split()[1:] if x.isdigit()]

    def stop(self, name: str) -> None:
        self._ours(name)
        self.run(["docker", "stop", "-t", "30", name])
        self.run(["docker", "rm", "-f", name])

    def gpu_pids(self, gpu_uuids: list[str]) -> dict[str, list[int]]:
        p = self.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
        if p.returncode != 0:
            raise RuntimeError_(f"nvidia-smi failed: {p.stderr.strip()[:300]}")
        out: dict[str, list[int]] = {u: [] for u in gpu_uuids}
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0] in out and parts[1].isdigit():
                out[parts[0]].append(int(parts[1]))
        return out

    def verify_stopped(
        self, name: str, pids: list[int], gpu_uuids: list[str]
    ) -> tuple[bool, dict[str, Any]]:
        evidence: dict[str, Any] = {
            "container": name,
            "gpu_uuids": gpu_uuids,
            "container_pids_before_stop": pids,
        }
        evidence["container_removed"] = not self.exists(name)
        try:
            on_gpu = self.gpu_pids(gpu_uuids)
        except RuntimeError_ as e:
            evidence["gpu_check_error"] = str(e)
            return False, evidence
        ours = sorted({pid for lst in on_gpu.values() for pid in lst if pid in set(pids)})
        alive = [pid for pid in pids if os.path.exists(f"/proc/{pid}")]
        evidence["our_pids_still_on_gpu"] = ours
        evidence["our_pids_still_alive"] = alive
        evidence["method"] = "docker inspect + nvidia-smi compute-apps + /proc"
        ok = evidence["container_removed"] and not ours and not alive
        return ok, evidence


def probe(endpoint_url: str, served_model_name: str, timeout_s: float = 3.0) -> ProbeResult:
    """READY needs both: the entry point answers /health AND the model is loaded (/v1/models)."""
    try:
        with httpx.Client(base_url=endpoint_url, timeout=timeout_s) as c:
            h = c.get("/health")
            if h.status_code != 200:
                return ProbeResult(False, f"/health -> {h.status_code}")
            m = c.get("/v1/models")
            if m.status_code != 200:
                return ProbeResult(False, f"/v1/models -> {m.status_code}")
            ids = [x.get("id") for x in m.json().get("data", [])]
            if served_model_name not in ids:
                return ProbeResult(False, f"model {served_model_name!r} not in {ids}")
            return ProbeResult(True, "health ok; model listed")
    except (httpx.HTTPError, ValueError) as e:
        return ProbeResult(False, f"unreachable: {e.__class__.__name__}")


def in_flight(endpoint_url: str, timeout_s: float = 3.0) -> int | None:
    """Running + waiting requests from vLLM's Prometheus metrics; None if unknown."""
    try:
        r = httpx.get(f"{endpoint_url}/metrics", timeout=timeout_s)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    total, seen = 0.0, False
    for line in r.text.splitlines():
        if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
            try:
                total += float(line.rsplit(" ", 1)[1])
                seen = True
            except ValueError:
                pass
    return int(total) if seen else None


# --------------------------------------------------------------------------- fake


@dataclass
class FakeRuntime:
    """In-memory runtime for tests. `fail_start`/`crash_after_start`/`leave_pids` inject faults."""

    fail_start: bool = False
    crash_after_start: bool = False
    leave_pids: bool = False
    containers: dict[str, dict[str, Any]] = field(default_factory=dict)
    started: list[LaunchSpec] = field(default_factory=list)
    _next_pid: int = 40000

    def start(self, spec: LaunchSpec) -> None:
        if self.fail_start:
            raise RuntimeError_("docker run failed: simulated")
        self.started.append(spec)
        self._next_pid += 1
        self.containers[spec.name] = {
            "running": not self.crash_after_start,
            "pids": [self._next_pid],
            "spec": spec,
        }

    def exists(self, name: str) -> bool:
        return name in self.containers

    def is_running(self, name: str) -> bool:
        return bool(self.containers.get(name, {}).get("running"))

    def logs(self, name: str, tail: int = 40) -> str:
        return "simulated crash: CUDA out of memory" if self.crash_after_start else ""

    def host_pids(self, name: str) -> list[int]:
        return list(self.containers.get(name, {}).get("pids", []))

    def stop(self, name: str) -> None:
        if not self.leave_pids:
            self.containers.pop(name, None)

    def verify_stopped(
        self, name: str, pids: list[int], gpu_uuids: list[str]
    ) -> tuple[bool, dict[str, Any]]:
        removed = name not in self.containers
        return removed, {
            "container": name,
            "container_removed": removed,
            "gpu_uuids": gpu_uuids,
            "our_pids_still_on_gpu": [] if removed else pids,
            "method": "fake",
        }
