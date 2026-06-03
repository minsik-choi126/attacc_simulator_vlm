"""Phase 0.2 — Sanity check: do our H100 numbers match NVIDIA's published
inference_benchmarks.md table?

The official table is organized as (model, task, GPU, engine) ->
{resolution/TP -> seconds}.  Schema confirmed from
https://github.com/NVIDIA/cosmos/blob/main/inference_benchmarks.md
fetched 2026-06-04:
    columns = 256p/1, 256p/4, 256p/8, 480p/1, 480p/4, 480p/8,
              720p/1, 720p/4, 720p/8
    GPUs    = RTX_PRO_6000_Blackwell, H20, H100_NVL, H200_NVL,
              H100_80GB_HBM3_SXM, H200_141GB_HBM3, B200, B300
    engines = PyTorch, vLLM-Omni, Diffusers

Phase 1.1 (cosmos_e2e_latency.json) records (model, task, framework, tp,
batch, resolution, ...) rows; we map those to the (resolution/TP) column
key and compare to the published number if NVIDIA_BENCHMARK has that
cell.

Output: results/cosmos_nvidia_repro.json
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import load, save
from cosmos_facts import NVIDIA_BENCHMARK, ENGINE_RESOLUTION_OVERRIDES


# Map our detected host tag -> NVIDIA benchmark GPU key.
# Our hw_detect only returns A6000/H100/A100; NVIDIA labels are finer.
# For H100 we don't know NVL vs 80GB SXM from nvidia-smi name alone --
# require user override via env COSMOS_NVIDIA_KEY if needed.
HOST_TO_NVIDIA_GPU_DEFAULT = {
    "H100": "H100_80GB_HBM3_SXM",  # NVL = different SKU; user can override
    "H200": "H200_141GB_HBM3",
    "B200": "B200",
    "A100": None,                  # not in benchmark
    "A6000": None,
}

FRAMEWORK_TO_ENGINE = {
    "pytorch":   "PyTorch",
    "vllm_omni": "vLLM-Omni",
    "diffusers": "Diffusers",
}


def _col_key(resolution, tp):
    res_tag = {"256p": "256p", "480p": "480p", "720p": "720p"}.get(resolution)
    if res_tag is None:
        return None
    return f"{res_tag}/{tp}"


def compare(measured_s, reference_s, tolerance=0.15):
    if measured_s is None or reference_s is None:
        return None
    delta = (measured_s - reference_s) / reference_s
    return {"measured_s": measured_s, "reference_s": reference_s,
            "delta_pct": round(delta * 100, 2),
            "within_tolerance": abs(delta) <= tolerance,
            "tolerance_pct": tolerance * 100}


def main():
    import os
    host = detect_host()
    gpu_override = os.environ.get("COSMOS_NVIDIA_KEY")
    nvidia_gpu = gpu_override or HOST_TO_NVIDIA_GPU_DEFAULT.get(host)
    print(f"[Phase 0.2] nvidia_repro on host={host}  "
          f"nvidia_gpu_key={nvidia_gpu!r}")
    print(f"   (override with COSMOS_NVIDIA_KEY=... e.g. H100_NVL)")

    if nvidia_gpu is None:
        print(f"  no NVIDIA table for host={host} -> skipping comparison")
        save("cosmos_nvidia_repro",
              {"phase": "0.2", "host": host, "platform": host,
               "note": "no NVIDIA benchmark row for this host"},
              {})
        return

    measured = load("cosmos_e2e_latency")
    if measured is None:
        print("  cosmos_e2e_latency.json missing -- run Phase 1.1 first")
        return
    rows = measured.get("results", {}).get("rows", [])

    comparisons = []
    coverage = {"checked": 0, "no_reference": 0, "no_match": 0}
    for row in rows:
        if row.get("status") != "ok":
            continue
        engine = FRAMEWORK_TO_ENGINE.get(row.get("framework"))
        if engine is None:
            continue
        col = _col_key(row.get("resolution"), row.get("tp"))
        if col is None:
            continue
        ref_table = NVIDIA_BENCHMARK.get(
            (row.get("model"), row.get("task"), nvidia_gpu, engine))
        if ref_table is None:
            coverage["no_reference"] += 1
            continue
        ref = ref_table.get(col)
        if ref is None:
            coverage["no_match"] += 1
            continue
        cmp = compare(row.get("e2e_s"), ref)
        # Resolution-override caveat (e.g. Diffusers 256p = 320x192,
        # not 448x256).
        caveats = []
        res = row.get("resolution")
        if (engine in ENGINE_RESOLUTION_OVERRIDES
                and res in ENGINE_RESOLUTION_OVERRIDES[engine]):
            ovr = ENGINE_RESOLUTION_OVERRIDES[engine][res]
            caveats.append(
                f"resolution_mismatch: NVIDIA {engine} {res} uses "
                f"{ovr[0]}x{ovr[1]} internally; we run "
                f"{res} = canonical resolution -- comparison is "
                f"approximate")
        cmp.update({"model": row.get("model"), "task": row.get("task"),
                     "engine": engine, "col": col,
                     "caveats": caveats})
        comparisons.append(cmp)
        coverage["checked"] += 1
        mark = "OK" if cmp["within_tolerance"] else "DRIFT"
        if caveats: mark += " [CAVEAT]"
        print(f"  {row['model']:14s} {row['task']:6s} {engine:10s} "
              f"{col:>8s}  meas={cmp['measured_s']:.2f}s  "
              f"ref={cmp['reference_s']:.2f}s  "
              f"delta={cmp['delta_pct']:+.1f}%  [{mark}]")

    print(f"\nCoverage: checked={coverage['checked']}  "
          f"no_ref_cell={coverage['no_reference']}  "
          f"no_resolution_match={coverage['no_match']}")
    save("cosmos_nvidia_repro",
          {"phase": "0.2", "host": host, "platform": host,
           "nvidia_gpu_key": nvidia_gpu, "tolerance": 0.15},
          {"comparisons": comparisons, "coverage": coverage,
           "any_drift": any(not c["within_tolerance"]
                              for c in comparisons)})


if __name__ == "__main__":
    main()
