"""GPU topology discovery from nvidia-smi (deliverable: "GPU topology discovery").

Reports per GPU: memory used/free, utilisation, running processes; per pair: PCIe path class and whether
peer-to-peer over NVLink works; per node: NVLink link count and whether all GPUs are NVLink-connected.

Run:  python -m planner.topology            (on the node)
      python -m planner.topology --free 30000   GPUs with >= 30,000 MiB free and no compute load
Tests feed recorded gpu-02 outputs instead of calling nvidia-smi (see tests/fixtures).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Callable


def _smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=30).stdout


@dataclass
class GPUState:
    index: int
    uuid: str
    name: str
    mem_total_mib: int
    mem_used_mib: int
    util_pct: int
    processes: list[int] = field(default_factory=list)
    numa: str = ""

    @property
    def mem_free_mib(self) -> int:
        return self.mem_total_mib - self.mem_used_mib


def parse_gpus(csv: str) -> list[GPUState]:
    out = []
    for line in csv.strip().splitlines():
        i, uuid, name, total, used, util = [x.strip() for x in line.split(",")]
        out.append(GPUState(int(i), uuid, name, int(total), int(used), int(util)))
    return out


def parse_apps(csv: str) -> dict[str, list[int]]:
    procs: dict[str, list[int]] = {}
    for line in csv.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, _mem = [x.strip() for x in line.split(",")]
        procs.setdefault(uuid, []).append(int(pid))
    return procs


def parse_matrix(text: str) -> dict[tuple[int, int], str]:
    """Parse `nvidia-smi topo -m` or `topo -p2p n` into {(i, j): value} for GPU pairs."""
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in text.splitlines()]
    header = next(l for l in lines if l.split()[:2] == ["GPU0", "GPU1"])   # column header row
    n = len(re.findall(r"\bGPU\d+\b", header))
    m = {}
    for l in lines:
        tok = l.split()
        if tok and re.fullmatch(r"GPU\d+", tok[0]):
            i = int(tok[0][3:])
            for j, v in enumerate(tok[1:1 + n]):
                if i != j:
                    m[(i, j)] = v
    return m


def parse_numa(topo_text: str, n: int) -> dict[int, str]:
    out = {}
    for l in topo_text.splitlines():
        tok = l.split()
        if tok and re.fullmatch(r"GPU\d+", tok[0]):
            rest = tok[1:]
            # after the GPU and NIC columns: CPU affinity (e.g. 0-63,128-191), then NUMA affinity
            for k, t in enumerate(rest):
                if re.fullmatch(r"[\d,\-]+", t) and ("-" in t) and k + 1 < len(rest):
                    out[int(tok[0][3:])] = rest[k + 1]
                    break
    return out


def count_nvlinks(text: str) -> int:
    return len(re.findall(r"Link \d+: [\d.]+ GB/s", text))


def discover(run: Callable[..., str] = _smi) -> dict:
    gpus = parse_gpus(run("--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"))
    procs = parse_apps(run("--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"))
    for g in gpus:
        g.processes = procs.get(g.uuid, [])
    topo_text = run("topo", "-m")
    pcie = parse_matrix(topo_text)
    for i, numa in parse_numa(topo_text, len(gpus)).items():
        gpus[i].numa = numa
    p2p = parse_matrix(run("topo", "-p2p", "n"))
    links = count_nvlinks(run("nvlink", "--status", "-i", "0"))
    all_nvlink = bool(p2p) and all(v == "OK" for v in p2p.values())
    return {
        "gpus": [asdict(g) | {"mem_free_mib": g.mem_free_mib} for g in gpus],
        "nvlink": {"links_per_gpu": links, "gbps_per_gpu_per_direction": links * 25,
                   "p2p_all_pairs_ok": all_nvlink},
        "pcie_paths": {f"{i}-{j}": v for (i, j), v in sorted(pcie.items()) if i < j},
        "tp_placement": ("any GPUs: all pairs have NVLink peer-to-peer" if all_nvlink
                         else "keep TP groups on GPUs with NVLink / the same PCIe switch"),
    }


def free_gpus(topo: dict, min_free_mib: int, max_util_pct: int = 10) -> list[int]:
    """GPUs with enough free memory and no meaningful compute load (other tenants' jobs)."""
    return [g["index"] for g in topo["gpus"]
            if g["mem_free_mib"] >= min_free_mib and g["util_pct"] <= max_util_pct]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", type=int, help="list GPUs with at least this many MiB free and <= 10%% utilisation")
    a = ap.parse_args()
    t = discover()
    if a.free:
        print(",".join(map(str, free_gpus(t, a.free))))
    else:
        print(json.dumps(t, indent=2))


if __name__ == "__main__":
    main()
