"""Grid-search SCALING_FACTOR (MAX_COMPUTE_UTIL × MAX_OFF_MEM_BW_UTIL) for the
GPU baseline so simulator predictions match the paper-grade vLLM measurements.

Compares simulator g_time and s_time against measured ITL and TTFT for the
3 paper-grade VLM TP=1 results in §17.8. Outputs a calibration table and
suggests the (compute_util, mem_util) pair that minimizes the maximum
abs(log2(sim/measured)) error across all 6 (3 models × 2 metrics) datapoints.

Usage:
    python3 tests/calibrate_scaling.py
"""

import json
import math
import os
import sys
import subprocess
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


REFERENCE = {
    "LLaVA-1.5-7B": {
        "image_size": 336,
        "lin": 569,
        "lout": 128,
        "ttft_ms": 33.76,
        "itl_ms_per_tok": 7.254,
    },
    "Qwen2.5-VL-7B": {
        "image_size": 672,
        "lin": 569,
        "lout": 128,
        "ttft_ms": 160.52,
        "itl_ms_per_tok": 8.578,
    },
    "LLaVA-Next-Mistral-7B": {
        "image_size": 672,
        "lin": 569,
        "lout": 128,
        "ttft_ms": 100.67,
        "itl_ms_per_tok": 7.713,
    },
}


def run_simulator(model_name, image_size, lin, lout, compute_util, mem_util):
    out = os.path.join(ROOT, "output.csv")
    if os.path.exists(out):
        os.remove(out)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ATTACC_MAX_COMPUTE_UTIL"] = str(compute_util)
    env["ATTACC_MAX_OFF_MEM_BW_UTIL"] = str(mem_util)
    cmd = [
        sys.executable, "main.py", "--system", "dgx", "--gpu", "H100", "--ngpu",
        "1", "--tp", "1", "--num_attacc", "1", "--num_hbm", "5", "--interface",
        "NVLINK4", "--model", model_name, "--lin", str(lin), "--lout",
        str(lout), "--batch", "1", "--image_size", str(image_size), "--max_L",
        "2048", "--pipeopt", "--ffopt"
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    with open(out) as fp:
        for line in fp:
            pass
        last = line.strip().split(",")
    s_time = float(last[18])
    g_time = float(last[26])
    return s_time, g_time


def log2_err(sim, measured):
    if sim <= 0 or measured <= 0:
        return float("inf")
    return math.log2(sim / measured)


def main():
    grid = []
    for cu in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for mu in [0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
            errors = []
            details = []
            for model_name, ref in REFERENCE.items():
                s_time, g_time = run_simulator(model_name, ref["image_size"],
                                               ref["lin"], ref["lout"], cu, mu)
                err_ttft = log2_err(s_time, ref["ttft_ms"])
                err_itl = log2_err(g_time, ref["itl_ms_per_tok"])
                errors.append(abs(err_ttft))
                errors.append(abs(err_itl))
                details.append((model_name, s_time, ref["ttft_ms"], g_time,
                                ref["itl_ms_per_tok"]))
            max_err = max(errors)
            mean_err = sum(errors) / len(errors)
            grid.append((cu, mu, max_err, mean_err, details))
            print("compute={:.2f} mem={:.2f} max|log2|={:.3f} mean|log2|={:.3f}".
                  format(cu, mu, max_err, mean_err))

    grid.sort(key=lambda x: x[2])  # by max log2 error
    print("\n=== Top 5 by minimax log2 error ===")
    for cu, mu, mx, mn, _ in grid[:5]:
        print("compute={:.2f} mem={:.2f} max|log2|={:.3f} mean|log2|={:.3f}".
              format(cu, mu, mx, mn))

    best_cu, best_mu, _, _, best_details = grid[0]
    print("\n=== Best config: compute_util={}, mem_util={} ===".format(
        best_cu, best_mu))
    for name, s_time, ttft, g_time, itl in best_details:
        print(("  {}: sim_s={:.2f} (meas TTFT={:.2f}, ratio={:.3f}) | "
               "sim_g={:.3f} (meas ITL={:.3f}, ratio={:.3f})").format(
                   name, s_time, ttft, s_time / ttft, g_time, itl,
                   g_time / itl))


if __name__ == "__main__":
    main()
