from pathlib import Path

from planner.topology import discover, free_gpus

FX = Path(__file__).parent / "fixtures"


def fake_smi(*args):
    if args[0].startswith("--query-gpu"):
        return (FX / "gpus.csv").read_text()
    if args[0].startswith("--query-compute-apps"):
        return (FX / "apps.csv").read_text()
    if args[:2] == ("topo", "-m"):
        return (FX / "topo.txt").read_text()
    if args[:2] == ("topo", "-p2p"):
        return (FX / "p2p_nvlink.txt").read_text()
    if args[0] == "nvlink":
        return (FX / "nvlink0.txt").read_text()
    raise AssertionError(args)


def test_gpu02_topology():
    t = discover(run=fake_smi)
    assert len(t["gpus"]) == 8
    assert t["nvlink"] == {"links_per_gpu": 12, "gbps_per_gpu_per_direction": 300, "p2p_all_pairs_ok": True}
    assert t["pcie_paths"]["0-1"] == "PXB" and t["pcie_paths"]["0-4"] == "SYS"
    assert t["gpus"][0]["numa"] == "0" and t["gpus"][4]["numa"] == "1"
    assert t["gpus"][4]["processes"] == [1846117]


def test_free_gpus_excludes_busy_and_full():
    t = discover(run=fake_smi)
    # GPUs 2,3,7 run at 90-99% (training jobs); GPU 6 has only ~12 GB free
    assert free_gpus(t, 25000) == [0, 1, 4, 5]
    assert free_gpus(t, 40000) == [0]
