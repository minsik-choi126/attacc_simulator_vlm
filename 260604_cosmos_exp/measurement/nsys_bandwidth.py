"""Phase 1.4 — DRAM bandwidth profile via Nsight Systems.

Wraps a single-shot generation under `nsys profile` (with NVTX ranges so
the analyzer can correlate phases) and post-processes the report.

This script does NOT compute bandwidth itself -- it just runs nsys and
records the report path; bandwidth analysis is done by
analysis/bandwidth_from_nsys.py against the .nsys-rep file (which can
be exported to .csv via `nsys stats`).

Output: results/cosmos_bandwidth_profile.json (paths + nsys exit code)
"""
import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import save, _results_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano")
    ap.add_argument("--task", default="t2v")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--framework", default="pytorch",
                    choices=["pytorch", "vllm_omni"])
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--frames", type=int, default=189)
    ap.add_argument("--denoise-steps", type=int, default=35)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    host = detect_host()
    out_dir = pathlib.Path(args.out_dir
                             or _results_dir() / "nsys_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"nsys_{host}_{args.framework}_{args.model}_"
            f"tp{args.tp}_b{args.batch}_{args.resolution}_"
            f"f{args.frames}_s{args.denoise_steps}_{int(time.time())}")
    rep = out_dir / f"{stem}.nsys-rep"

    runner = HERE / args.framework / "runner.py"
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt",
        "--gpu-metrics-device=all",
        "--cuda-memory-usage=true",
        "--output", str(rep.with_suffix("")),
        sys.executable, str(runner),
        "--model", args.model, "--task", args.task,
        "--tp", str(args.tp), "--batch", str(args.batch),
        "--resolution", args.resolution, "--frames", str(args.frames),
        "--denoise-steps", str(args.denoise_steps),
        "--reps", "1", "--json",
    ]
    print(f"[Phase 1.4] nsys profile -> {rep}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=3600)
    elapsed = time.perf_counter() - t0
    print(f"  nsys rc={proc.returncode}  wall={elapsed:.1f}s")

    # Try to export stats.
    stats_csv = rep.with_suffix(".gputrace.csv")
    try:
        subprocess.check_call(
            ["nsys", "stats", "--report", "gputrace",
             "--format", "csv", "--output", str(stats_csv), str(rep)],
            timeout=300)
    except Exception as e:
        print(f"  stats export skipped: {e}")

    config = {"phase": "1.4", "host": host, "platform": host, **vars(args)}
    save("cosmos_bandwidth_profile", config,
          {"nsys_rep": str(rep),
           "stats_csv": str(stats_csv),
           "nsys_rc": proc.returncode,
           "wall_s": elapsed,
           "note": "Run analysis/bandwidth_from_nsys.py against the rep "
                   "to compute per-phase DRAM read GB."})


if __name__ == "__main__":
    main()
