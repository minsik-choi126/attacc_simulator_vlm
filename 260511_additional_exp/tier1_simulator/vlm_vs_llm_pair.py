"""B1 -- LLM <-> VLM pair speedup comparison.

For each backbone with both LLM-only and VLM variant in config, run
sim with system in {dgx, dgx-attacc} at varying batch and report:

    speedup_llm  = e2e_dgx_llm  / e2e_dgx-attacc_llm
    speedup_vlm  = e2e_dgx_vlm  / e2e_dgx-attacc_vlm
    delta        = speedup_vlm - speedup_llm
    relative_pct = 100 * delta / speedup_llm

delta > 0 means AttAcc helps VLM more than the equivalent text-only
LLM, supporting the paper claim that VLM is a stronger fit for AttAcc.

Pairs are matched at identical (lin, lout, batch) to isolate the
backbone-shared LLM portion vs the visual-pipeline-added VLM portion.

Output: results/vlm_vs_llm_pair.json
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save


# (LLM_label, VLM_label, image_size for VLM, lin for matched runs)
PAIRS = [
    ("Vicuna-7B",   "LLaVA-1.5-7B",            336, 704),
    ("Mistral-7B",  "LLaVA-Next-Mistral-7B",   336, 704),
    ("Qwen3-4B",    "Qwen3-VL-4B",             672, 569),
]
BATCHES = [1, 2, 4, 8, 16, 32, 64]
SYSTEMS = ["dgx", "dgx-attacc"]
LOUT = 128


def _e2e_ms(metrics):
    s = metrics.get("s_time")
    g = metrics.get("g_time")
    if s is None or g is None:
        return None
    return s + g * (LOUT - 1)


def _run(model, system, batch, lin, image_size):
    return sr.run(
        model=model, system=system, gpu="A6000",
        ngpu=1, tp=1, num_attacc=1, num_hbm=5,
        interface="NVLINK_BRIDGE", pim="bank",
        lin=lin, lout=LOUT, batch=batch,
        image_size=image_size,
        prefill_chunk=512, prefill_samples=8,
        max_L=4096,
        powerlimit=False, ffopt=True, pipeopt=False,
        word=2,
    )


def measure_pair(llm_label, vlm_label, image_size, lin, batch):
    """Run all 4 cells (LLM/VLM × dgx/dgx-attacc) at one batch."""
    cells = {}
    for system in SYSTEMS:
        cells[("llm", system)] = _run(llm_label, system, batch, lin, image_size)
        cells[("vlm", system)] = _run(vlm_label, system, batch, lin, image_size)

    def speedup(side):
        dgx = cells[(side, "dgx")]
        att = cells[(side, "dgx-attacc")]
        if dgx is None or att is None:
            return None
        e_dgx = _e2e_ms(dgx)
        e_att = _e2e_ms(att)
        if not e_dgx or not e_att:
            return None
        return e_dgx / e_att

    speedup_llm = speedup("llm")
    speedup_vlm = speedup("vlm")
    delta = (speedup_vlm - speedup_llm
             if (speedup_llm is not None and speedup_vlm is not None) else None)
    rel = (100 * delta / speedup_llm
           if (delta is not None and speedup_llm) else None)

    return {
        "llm_label": llm_label, "vlm_label": vlm_label,
        "image_size": image_size, "lin": lin, "lout": LOUT, "batch": batch,
        "cells": {
            f"{side}_{sys}": ({"s_time": cells[(side, sys)].get("s_time"),
                                "g_time": cells[(side, sys)].get("g_time"),
                                "e2e_ms": _e2e_ms(cells[(side, sys)])}
                              if cells[(side, sys)] else None)
            for side, sys in [("llm", "dgx"), ("llm", "dgx-attacc"),
                              ("vlm", "dgx"), ("vlm", "dgx-attacc")]
        },
        "speedup_llm": speedup_llm,
        "speedup_vlm": speedup_vlm,
        "delta": delta,
        "delta_pct_of_llm": rel,
    }


def main():
    print("B1 -- LLM <-> VLM pair speedup comparison")
    print(f"  pairs: {[(p[0], p[1]) for p in PAIRS]}")
    print(f"  batches: {BATCHES}")
    results = []
    for llm, vlm, img, lin in PAIRS:
        print(f"\n=== {llm} vs {vlm}  (img={img}, lin={lin}) ===")
        for b in BATCHES:
            row = measure_pair(llm, vlm, img, lin, b)
            results.append(row)
            sl = row["speedup_llm"]
            sv = row["speedup_vlm"]
            d = row["delta"]
            sl_s = f"{sl:.2f}x" if sl is not None else "—"
            sv_s = f"{sv:.2f}x" if sv is not None else "—"
            d_s = f"{d:+.2f}x ({row['delta_pct_of_llm']:+.0f}%)" \
                  if d is not None else "—"
            print(f"  b={b:>3d}  speedup_llm={sl_s:>7s}  "
                  f"speedup_vlm={sv_s:>7s}  delta={d_s}")

    save("vlm_vs_llm_pair",
         {"pairs": [(l, v) for l, v, _, _ in PAIRS],
          "batches": BATCHES, "systems": SYSTEMS, "lout": LOUT},
         {"rows": results})
    print("\nSaved -> results/vlm_vs_llm_pair.json")


if __name__ == "__main__":
    main()
