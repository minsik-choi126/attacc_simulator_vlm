"""ViT recalibration -- derive per-model s_corr (vision tower complexity)
from measured TTFT data and analyze fit.

Reads measured TTFT from existing tier2_measurement results (or main
results/r9_*.json from earlier campaign), runs simulator with default
ViT cost, computes per-model correction factor needed.

Output: fit table, recommended _build_vit() scaling adjustment.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save


# Measured TTFT p50 (ms) from prior campaign (results/r9_*.json or r7_*.json).
# Source for these numbers: ../../results/r9_*_mmmu_tp1.json + r7_*.json.
MEASURED = {
    "Qwen2.5-VL-7B": {
        "ttft_ms_p50": 107.43,
        "itl_ms_p50": 8.72,
        "image_size": 672,
        "lin_text": 334,        # MMMU-Pro median seq_in
        "lout": 128,
    },
    "LLaVA-1.5-7B": {
        "ttft_ms_p50": 41.16,
        "itl_ms_p50": 7.27,
        "image_size": 336,
        "lin_text": 620,
        "lout": 128,
    },
    "LLaVA-Next-Mistral-7B": {
        "ttft_ms_p50": 102.53,
        "itl_ms_p50": 7.63,
        "image_size": 672,
        "lin_text": 1979,
        "lout": 128,
    },
}


def simulate_one(model, image_size, lin, lout):
    m = sr.run(
        model=model,
        system="dgx",                     # GPU only -- matches vLLM
        gpu="H100",
        ngpu=1, tp=1, num_attacc=1, num_hbm=5,
        interface="NVLINK4",
        pim="bank",
        lin=lin, lout=lout, batch=1,
        image_size=image_size,
        prefill_chunk=512, prefill_samples=8,
        max_L=2048,
        powerlimit=False, ffopt=True, pipeopt=False,
        word=2,
    )
    if m is None:
        return None, None
    return m.get("s_time"), m.get("g_time")


def main():
    print("ViT recalibration -- measured vs simulated TTFT cross-model")
    rows = []
    for model, meas in MEASURED.items():
        s_sim, g_sim = simulate_one(model, meas["image_size"],
                                     meas["lin_text"], meas["lout"])
        if s_sim is None:
            rows.append({"model": model, "status": "sim_fail"})
            continue
        s_corr = meas["ttft_ms_p50"] / s_sim if s_sim else None
        g_corr = meas["itl_ms_p50"] / g_sim if g_sim else None
        rows.append({
            "model": model,
            "image_size": meas["image_size"],
            "lin_text": meas["lin_text"],
            "sim_s_ms": round(s_sim, 3),
            "sim_g_ms": round(g_sim, 4),
            "meas_ttft_ms": meas["ttft_ms_p50"],
            "meas_itl_ms": meas["itl_ms_p50"],
            "s_corr": round(s_corr, 3) if s_corr else None,
            "g_corr": round(g_corr, 3) if g_corr else None,
        })
        print("  {:25s}: sim_s={:>6.2f}ms meas={:>6.2f}ms -> s_corr={:.2f}x | "
              "sim_g={:.3f} meas={:.3f} -> g_corr={:.3f}".format(
                  model, s_sim, meas["ttft_ms_p50"], s_corr,
                  g_sim, meas["itl_ms_p50"], g_corr))

    # Analyze: what vit_layers x tokens x hidden complexity correlates with s_corr?
    # Provide diagnostic numbers (not fit yet -- manual paper analysis).
    archs = {
        "Qwen2.5-VL-7B":          (32, 1280, 2304),    # vit_layers, vit_hidden, tokens
        "LLaVA-1.5-7B":           (24, 1024, 576),
        "LLaVA-Next-Mistral-7B":  (24, 1024, 2304),
    }
    print("\nViT architecture complexity check:")
    print("  {:25s} {:>6s} {:>6s} {:>8s} {:>10s} {:>8s}".format(
        "model", "layers", "hidden", "tokens", "FLOPs_est", "s_corr"))
    diag = []
    for model, (L, H, T) in archs.items():
        flops = L * (4 * T * H * H + 2 * T * T * H + 8 * T * H * (H * 4))
        s_corr = next((r["s_corr"] for r in rows if r.get("model") == model),
                      None)
        diag.append({"model": model, "vit_layers": L, "vit_hidden": H,
                      "vit_tokens": T, "flops_est": flops, "s_corr": s_corr})
        print("  {:25s} {:>6d} {:>6d} {:>8d} {:>10.2e} {:>8.2f}".format(
            model, L, H, T, flops, s_corr if s_corr else 0))

    save("vit_recalibration",
         {"purpose": "Derive per-model s_corr to identify _build_vit() fix path",
          "method": "compare simulated TTFT vs vLLM measured TTFT p50 (MMMU-Pro)"},
         {"per_model": rows, "architecture_diag": diag,
          "decode_g_corr_universal_est": 1.46})
    print("\nDone -- Paper figure source for prefill correction breakdown")


if __name__ == "__main__":
    main()
