"""Step 6: turn a plan into a running deployment (or into templates Student 1 can run).

Three outputs from one plan:
  docker   - the exact `docker run` command (what run_config.sh does, generated from the plan)
  compose  - a docker-compose service (for Student 1's deployer)
  spec     - a JSON deployment spec (image, command, env, GPUs, health check, concurrency limit)
and `launch` actually starts it and waits for /health.

Run:  python -m planner.deploy --model Qwen/Qwen2.5-7B-Instruct --prompt 512 --output 128 --tpot-sla 15 \
          --gpus 0,1,4,5 --mem-util 0.3 [--format docker|compose|spec] [--launch]
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .plan import PlanRequest, plan

HF_HOST = "/root/.cache/huggingface"
HF_IN_CONTAINER = "/hf"


@dataclass
class ImageProfile:
    """Everything a specific serving image needs. Each quirk below was found on gpu-02 (README)."""
    name: str
    image: str
    entrypoint: str = "vllm"
    env: dict = field(default_factory=dict)
    extra_args: list = field(default_factory=list)


PROFILES = {
    # vLLM 0.31.0+cu126.abujafar1: runs as uid 1000 (cannot read /root), bakes in HF_HOME=/model-cache and
    # CUDA_VISIBLE_DEVICES=0,1, has no FlashInfer, FlashAttention 2 only on A100.
    "abujafar-0.31": ImageProfile(
        name="abujafar-0.31", image="fbdaa5259f0a",
        env={"HF_HOME": HF_IN_CONTAINER, "HF_HUB_CACHE": f"{HF_IN_CONTAINER}/hub", "HF_HUB_OFFLINE": "1",
             "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        extra_args=["--attention-backend", "FLASH_ATTN", "--attention-config", '{"flash_attn_version":2}'],
    ),
}
DEFAULT_PROFILE = "abujafar-0.31"


def resolve_snapshot(model_id: str, hf_root: str = HF_HOST) -> str:
    """Container path of the cached snapshot. Deployments never download: missing weights fail fast."""
    snaps = Path(hf_root) / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    found = sorted(p for p in snaps.glob("*") if (p / "config.json").exists()) if snaps.exists() else []
    if not found:
        raise FileNotFoundError(f"{model_id} is not in {hf_root}: run `hf download {model_id}` on the node first")
    return f"{HF_IN_CONTAINER}/hub/{found[0].relative_to(Path(hf_root) / 'hub')}"


def _vllm_command(model_id: str, model_path: str, chosen: dict, profile: ImageProfile) -> list[str]:
    args = list(chosen["vllm_args"])
    if chosen["dp"] > 1 and "--api-server-count" not in args:
        args += ["--api-server-count", str(chosen["dp"])]     # Step 3b: one API server per DP rank helps
    return ["serve", model_path, "--served-model-name", model_id, *profile.extra_args, *args]


def docker_run_cmd(out: dict, model_id: str, model_path: str, name: str, port: int,
                   profile: ImageProfile) -> list[str]:
    ch = out["chosen"]
    ids = ch["gpu_ids"]
    env = dict(profile.env, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(len(ids))))
    cmd = ["docker", "run", "-d", "--name", name, "--gpus", f'"device={",".join(map(str, ids))}"',
           "--ipc=host", "-p", f"{port}:8000", "-v", f"{HF_HOST}:{HF_IN_CONTAINER}:ro"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["--entrypoint", profile.entrypoint, profile.image]
    return cmd + _vllm_command(model_id, model_path, ch, profile)


def compose_yaml(out: dict, model_id: str, model_path: str, name: str, port: int, profile: ImageProfile) -> str:
    ch = out["chosen"]
    ids = ch["gpu_ids"]
    env = dict(profile.env, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(len(ids))))
    command = json.dumps(_vllm_command(model_id, model_path, ch, profile))
    envs = "\n".join(f"      {k}: {json.dumps(v)}" for k, v in env.items())
    return f"""services:
  {name}:
    image: {profile.image}
    entrypoint: ["{profile.entrypoint}"]
    command: {command}
    ipc: host
    ports: ["{port}:8000"]
    volumes: ["{HF_HOST}:{HF_IN_CONTAINER}:ro"]
    environment:
{envs}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: {json.dumps([str(i) for i in ids])}
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 10s
      retries: 60
"""


def spec(out: dict, model_id: str, model_path: str, name: str, port: int, profile: ImageProfile) -> dict:
    ch = out["chosen"]
    return {
        "name": name, "model": model_id, "image": profile.image, "image_profile": profile.name,
        "gpu_ids": ch["gpu_ids"], "port": port, "health": "/health", "openai_base": f"http://localhost:{port}/v1",
        "command": docker_run_cmd(out, model_id, model_path, name, port, profile),
        "layout": ch["name"], "max_concurrency_per_engine": ch["concurrency_per_replica"],
        "max_concurrency_total": ch["concurrency_total"],
        "predicted": {"tpot_p95_ms": ch["tpot_p95_ms"], "throughput_tok_s": ch["throughput_tok_s"]},
        "explanation": out["explanation"], "warnings": out["warnings"],
    }


def launch(cmd: list[str], name: str, port: int, timeout_s: int = 900) -> dict:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3):
                return {"status": "healthy", "seconds": round(time.time() - t0)}
        except Exception:
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                                   capture_output=True, text=True).stdout.strip()
            if state != "true":
                logs = subprocess.run(["docker", "logs", "--tail", "30", name], capture_output=True, text=True)
                return {"status": "crashed", "logs": (logs.stdout + logs.stderr)[-3000:]}
            time.sleep(5)
    return {"status": "timeout", "seconds": timeout_s}


def stop(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def record(entry: dict, path: str = "results/deployments.jsonl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry | {"time": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan and deploy")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--output", type=int, required=True)
    ap.add_argument("--tpot-sla", type=float, required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--target-concurrency", type=int)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--mem-util", type=float, default=0.9)
    ap.add_argument("--port", type=int, default=8401)
    ap.add_argument("--name", default="student3-deploy")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=list(PROFILES))
    ap.add_argument("--model-path", help="container path of the weights (default: cached HF snapshot)")
    ap.add_argument("--format", default="docker", choices=["docker", "compose", "spec"])
    ap.add_argument("--launch", action="store_true", help="start it and wait for /health")
    a = ap.parse_args()

    out = plan(PlanRequest(a.model, a.prompt, a.output, a.tpot_sla, [int(g) for g in a.gpus.split(",")],
                           a.target_concurrency, a.max_model_len, a.mem_util))
    if out["status"] != "ok":
        raise SystemExit("no feasible plan:\n- " + "\n- ".join(out["explanation"]))
    prof = PROFILES[a.profile]
    path = a.model_path or resolve_snapshot(a.model)
    cmd = docker_run_cmd(out, a.model, path, a.name, a.port, prof)
    for line in out["explanation"][:2]:
        print("#", line)
    if a.format == "compose":
        print(compose_yaml(out, a.model, path, a.name, a.port, prof))
    elif a.format == "spec":
        print(json.dumps(spec(out, a.model, path, a.name, a.port, prof), indent=2))
    else:
        print(shlex.join(cmd))
    if a.launch:
        res = launch(cmd, a.name, a.port)
        record({"name": a.name, "model": a.model, "layout": out["chosen"]["name"], "gpu_ids": out["chosen"]["gpu_ids"],
                "max_num_seqs": out["chosen"]["concurrency_per_replica"], "result": res["status"],
                "startup_s": res.get("seconds")})
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
