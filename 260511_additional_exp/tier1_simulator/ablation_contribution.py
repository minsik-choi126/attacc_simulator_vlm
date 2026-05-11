"""Ablation: contribution of each modification component.

For Qwen3-VL-4B S1, sequentially disable each modification group and
measure E2E gain change. Baseline = full proposal.

Variants (only those toggleable via CLI without code patches):
  A_no_pim     -- pure GPU baseline (dgx system, no AttAcc)
  A_no_chunked -- disable chunked sampled prefill (prefill_chunk = lin)
  A_no_routing -- conservative single-group routing (no per-layer dispatch)
  A_full       -- full proposal (all M-mods on)

NOTE: eff_lat (M6.4) and DeepStack injection (M9) are NOT exposed via CLI
toggles in current simulator. To ablate those, modify src/system.py or
src/model.py and re-run. We list them as "future ablation" entries with
status='requires_code_patch' for traceability.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

MODEL = "Qwen3-VL-4B"
LIN, LOUT, BATCH = 569, 128, 1
IMG = 672
DEFAULT_PIM_LAYERS = ("0,8,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,"
                       "28,29,31,33")


def total(m):
    return sr.e2e_ms(m, LOUT)


def run_variant(label, **overrides):
    base = dict(
        model=MODEL, system="dgx-attacc", gpu="H100",
        ngpu=1, tp=1, num_attacc=1, num_hbm=5, interface="NVLINK4",
        pim="bank", lin=LIN, lout=LOUT, batch=BATCH, image_size=IMG,
        prefill_chunk=512, prefill_samples=8, max_L=2048,
        powerlimit=True, ffopt=True, pipeopt=True, word=2,
        routing="list",
        pim_layers=DEFAULT_PIM_LAYERS,
    )
    base.update(overrides)
    m = sr.run(**base)
    return {"label": label, "overrides": overrides,
            "s_ms": m.get("s_time") if m else None,
            "g_ms": m.get("g_time") if m else None,
            "total_ms": total(m)}


def main():
    print("Ablation contribution decomposition -- Qwen3-VL-4B S1")
    variants = [
        run_variant("A_no_pim",      system="dgx"),
        run_variant("A_no_chunked",  prefill_chunk=LIN),
        run_variant("A_no_routing",  routing="conservative", pim_layers=""),
        run_variant("A_full"),
    ]
    future = [
        {"label": "A_no_efflat",   "status": "requires_code_patch",
         "note": "Toggle eff_lat in src/system.py _apply_eff_lat()"},
        {"label": "A_no_deepstack", "status": "requires_code_patch",
         "note": "Set has_deepstack=False in Qwen3-VL-4B config in src/config.py"},
    ]

    base = next(v["total_ms"] for v in variants if v["label"] == "A_no_pim")
    full = next(v["total_ms"] for v in variants if v["label"] == "A_full")
    if base and full:
        print("\n  Baseline (GPU only): {:.1f}ms".format(base))
        print("  Full proposal:       {:.1f}ms".format(full))
        print("  E2E speedup:         {:.2f}x".format(base / full))
    else:
        print("  WARN: some configs failed (base={} full={})".format(base, full))

    print()
    for v in variants:
        if v["total_ms"] and base:
            v["gain_vs_gpu"] = round(base / v["total_ms"], 3)
        else:
            v["gain_vs_gpu"] = None
        gain_str = "{:.3f}x".format(v["gain_vs_gpu"]) if v["gain_vs_gpu"] else "n/a"
        print("    {:14s} total={:>7.1f}ms  gain={}".format(
            v["label"], v["total_ms"] or -1, gain_str))

    save("ablation_contribution",
         {"model": MODEL, "lin": LIN, "lout": LOUT, "batch": BATCH,
          "platform": "H100 x 1 S1", "image_size": IMG,
          "deepstack_indices": [5, 11, 17]},
         {"variants": variants, "future_ablations": future,
          "baseline_gpu_ms": base, "full_proposal_ms": full})
    print("\nDone")


if __name__ == "__main__":
    main()
