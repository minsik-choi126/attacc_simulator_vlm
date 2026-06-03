"""Phase 0.1 — Cosmos 3 environment / framework / weight sanity check.

Verifies that the H100 node can run both vLLM-Omni and PyTorch+Diffusers
pipelines for Cosmos 3 Nano (and reports Super readiness given disk +
TP=2 availability).  Does NOT actually generate any video — it only
reports presence / versions / weight availability / GPU layout.

Output: results/cosmos_env_check.json (+ per-host copy)
"""
import importlib
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host, gpu_count, peak_bw_tbs
from result_aggregator import save
from cosmos_facts import NANO, SUPER


def _try_import(name):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "unknown")
        return {"present": True, "version": v}
    except Exception as e:
        return {"present": False, "error": str(e)[:200]}


def _nvidia_smi():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu="
             "name,memory.total,memory.free,driver_version",
             "--format=csv,noheader"],
            timeout=5).decode().strip()
    except Exception as e:
        return f"unavailable: {e}"


def _disk_free_gb(path="."):
    try:
        usage = shutil.disk_usage(str(path))
        return round(usage.free / 1e9, 1)
    except Exception:
        return None


def _huggingface_weight_check():
    """Look at common HF cache dirs and the canonical Cosmos repo IDs."""
    candidates = [
        pathlib.Path.home() / ".cache" / "huggingface" / "hub",
        pathlib.Path("/root/.cache/huggingface/hub"),
        pathlib.Path("/data/huggingface"),
    ]
    repos = {
        "Cosmos3-Nano": "nvidia/Cosmos3-Nano",
        "Cosmos3-Super": "nvidia/Cosmos3-Super",
        "Cosmos3-Reasoner1-7B": "nvidia/Cosmos-Reason1-7B",
        "Cosmos3-Predict2-Diffusion": "nvidia/Cosmos-Predict2",
    }
    found = {}
    for label, repo in repos.items():
        slug = "models--" + repo.replace("/", "--")
        present = False
        size_gb = None
        for cand in candidates:
            target = cand / slug
            if target.exists():
                present = True
                total = 0
                for f in target.rglob("*"):
                    try:
                        if f.is_file():
                            total += f.stat().st_size
                    except OSError:
                        pass
                size_gb = round(total / 1e9, 1)
                break
        found[label] = {"repo_id": repo, "present": present,
                        "size_gb": size_gb}
    return found


def main():
    host = detect_host()
    n_gpu = gpu_count()
    print(f"[Phase 0.1] env_setup on host={host}, gpu_count={n_gpu}")
    print(f"  peak BW   = {peak_bw_tbs(host)} TB/s")
    print(f"  nvidia-smi: {_nvidia_smi()}")

    # Package presence ----------
    frameworks = {
        "torch": _try_import("torch"),
        "diffusers": _try_import("diffusers"),
        "transformers": _try_import("transformers"),
        "vllm": _try_import("vllm"),
        "vllm_omni": _try_import("vllm_omni"),
        "flash_attn": _try_import("flash_attn"),
    }
    for n, v in frameworks.items():
        tag = v.get("version", "-")
        mark = "OK" if v["present"] else "MISS"
        print(f"  {n:14s} {mark:>5s}  v={tag}")

    weights = _huggingface_weight_check()
    print("HF weight check:")
    for k, v in weights.items():
        mark = "OK" if v["present"] else "MISS"
        sz = f"{v['size_gb']} GB" if v["size_gb"] else "-"
        print(f"  {k:30s} {mark:>5s}  size={sz}  ({v['repo_id']})")

    disk_free = _disk_free_gb(pathlib.Path.home())
    print(f"Disk free on $HOME: {disk_free} GB")

    readiness = {
        "Cosmos3-Nano_TP1": (
            n_gpu >= 1 and weights["Cosmos3-Nano"]["present"]
            and frameworks["torch"]["present"]
        ),
        "Cosmos3-Nano_TP2": (
            n_gpu >= 2 and weights["Cosmos3-Nano"]["present"]
        ),
        "Cosmos3-Super_TP2": (
            n_gpu >= 2 and weights["Cosmos3-Super"]["present"]
        ),
        "vLLM-Omni": frameworks["vllm_omni"]["present"],
        "PyTorch+Diffusers": (frameworks["torch"]["present"]
                              and frameworks["diffusers"]["present"]),
    }
    print("\nReadiness:")
    for k, v in readiness.items():
        print(f"  {k:25s} {'READY' if v else 'BLOCKED'}")

    config = {
        "phase": "0.1",
        "host": host,
        "platform": host,
        "gpu_count": n_gpu,
        "peak_bw_tbs": peak_bw_tbs(host),
        "model_facts": {"Nano": {k: NANO[k] for k in
                                  ("hidden", "n_layers", "n_kv_heads",
                                   "d_head", "max_position_embeddings")},
                         "Super": {k: SUPER[k] for k in
                                   ("hidden", "n_layers", "n_kv_heads",
                                    "d_head", "max_position_embeddings")}},
    }
    results = {
        "frameworks": frameworks,
        "weights": weights,
        "disk_free_gb": disk_free,
        "nvidia_smi": _nvidia_smi(),
        "readiness": readiness,
    }
    paths = save("cosmos_env_check", config, results)
    print(f"\nSaved -> {[str(p) for p in paths]}")


if __name__ == "__main__":
    main()
