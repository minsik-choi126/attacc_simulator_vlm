"""Apply a constant multiplicative correction to simulator s_time and g_time
based on the batch-invariant ratio observed in §17.11 (paper-grade vLLM
measurements), and recompute the residual error.

Design idea: ratio sim/measured for prefill ~= 0.12 across batch={1,4,8,16},
ratio for decode ~= 0.69. Apply prefill_correction=1/0.12=8.33 and
decode_correction=1/0.69=1.45 to simulator outputs. If residual error
collapses to ~0 across all batch/lout/lin points, the simulator captures
relative performance trends faithfully and only needs an absolute scaling
fix; that is a strong calibration argument for the paper's methodology
section.
"""

import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def run_simulator(model, image_size, lin, lout, batch):
    out = os.path.join(ROOT, "output.csv")
    if os.path.exists(out):
        os.remove(out)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable, "main.py", "--system", "dgx", "--gpu", "H100",
        "--ngpu", "1", "--tp", "1", "--num_attacc", "1", "--num_hbm", "5",
        "--interface", "NVLINK4", "--model", model, "--lin", str(lin),
        "--lout", str(lout), "--batch", str(batch), "--image_size",
        str(image_size), "--max_L", "2048", "--pipeopt", "--ffopt"
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as fp:
        for line in fp:
            pass
        last = line.strip().split(",")
    return float(last[18]), float(last[26])  # s_time, g_time


def load_measured(path):
    with open(path) as fp:
        return json.load(fp)["stats"]


def compare(label, model, image_size, lin, lout, batch, measured_stats):
    s_sim, g_sim = run_simulator(model, image_size, lin, lout, batch)
    ttft_meas = measured_stats["ttft_ms"]["p50"]
    itl_meas = measured_stats["itl_ms"]["p50"]
    return {
        "label": label,
        "lin": lin, "lout": lout, "batch": batch,
        "s_sim": s_sim, "g_sim": g_sim,
        "ttft_meas": ttft_meas, "itl_meas": itl_meas,
        "raw_s_ratio": s_sim / ttft_meas,
        "raw_g_ratio": g_sim / itl_meas,
    }


def main():
    # Calibrate using batch sweep at lin=569
    print("=== Step 1: Calibrate from batch sweep (lin=569) ===")
    batch_calib = []
    for batch in [1, 4, 8, 16]:
        path = os.path.join(
            ROOT, "results",
            "r8_qwen25_batch{}_tp1_vllm.json".format(batch))
        m = load_measured(path)
        c = compare("batch={}".format(batch), "Qwen2.5-VL-7B", 672, 569, 128,
                    batch, m)
        batch_calib.append(c)
        print("  batch={:2d} s_sim={:.2f} ttft_meas={:.2f} ratio={:.3f} | "
              "g_sim={:.3f} itl_meas={:.3f} ratio={:.3f}".format(
                  batch, c["s_sim"], c["ttft_meas"], c["raw_s_ratio"],
                  c["g_sim"], c["itl_meas"], c["raw_g_ratio"]))

    # Mean correction factor
    s_corr = statistics.mean(1 / c["raw_s_ratio"] for c in batch_calib)
    g_corr = statistics.mean(1 / c["raw_g_ratio"] for c in batch_calib)
    print("\nDerived corrections: s_corr (prefill) = {:.3f}, "
          "g_corr (decode) = {:.3f}".format(s_corr, g_corr))

    # Residual after correction (across batch sweep)
    print("\n=== Step 2: Residual after applying corrections to batch sweep ===")
    residual_s = []
    residual_g = []
    for c in batch_calib:
        s_corrected = c["s_sim"] * s_corr
        g_corrected = c["g_sim"] * g_corr
        rs = s_corrected / c["ttft_meas"]
        rg = g_corrected / c["itl_meas"]
        residual_s.append(rs)
        residual_g.append(rg)
        print("  batch={:2d} s_corr/meas={:.3f} g_corr/meas={:.3f}".format(
            c["batch"], rs, rg))
    print("Residual stats: s mean={:.3f} std={:.3f}; g mean={:.3f} std={:.3f}".
          format(statistics.mean(residual_s), statistics.stdev(residual_s),
                 statistics.mean(residual_g), statistics.stdev(residual_g)))

    # Apply to held-out lin sweep / lout sweep
    print("\n=== Step 3: Validate correction on held-out lout sweep ===")
    for lout in [32, 64, 256, 512]:
        path = os.path.join(
            ROOT, "results", "r8_qwen25_lout{}_tp1_vllm.json".format(lout))
        if not os.path.exists(path):
            continue
        m = load_measured(path)
        s_sim, g_sim = run_simulator("Qwen2.5-VL-7B", 672, 569, lout, 1)
        rs = (s_sim * s_corr) / m["ttft_ms"]["p50"]
        rg = (g_sim * g_corr) / m["itl_ms"]["p50"]
        print("  lout={:4d} s_corr/meas={:.3f} g_corr/meas={:.3f}".format(
            lout, rs, rg))

    print("\n=== Step 4: Apply to LLaVA-1.5 / LLaVA-Next (cross-model) ===")
    cross = [
        ("LLaVA-1.5-7B", 336, "results/r7_llava-hf_llava-1.5-7b-hf_tp1_vllm.json"),
        ("LLaVA-Next-Mistral-7B", 672,
         "results/r7_llava-hf_llava-v1.6-mistral-7b-hf_tp1_vllm.json"),
    ]
    for model, image_size, rel in cross:
        path = os.path.join(ROOT, rel)
        m = load_measured(path)
        s_sim, g_sim = run_simulator(model, image_size, 569, 128, 1)
        rs = (s_sim * s_corr) / m["ttft_ms"]["p50"]
        rg = (g_sim * g_corr) / m["itl_ms"]["p50"]
        print("  {} s_corr/meas={:.3f} g_corr/meas={:.3f}".format(
            model, rs, rg))


if __name__ == "__main__":
    main()
