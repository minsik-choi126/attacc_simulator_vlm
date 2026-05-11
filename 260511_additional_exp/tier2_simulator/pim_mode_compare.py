"""PIM mode comparison -- bank / bank_group / buffer level (AttAcc paper sec.5).

bank: per-bank GEMV unit (highest parallelism)
bg:   per-bank-group (less parallelism, less overhead)
buffer: per-pCH on buffer die

For each VLM x each mode -> s_time, g_time, energy.
Confirms our simulator reproduces relative ordering of paper sec.5.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

MODELS = [
    ("Qwen3-VL-4B",            672,  569),
    ("Qwen2.5-VL-7B",          672,  704),
    ("LLaVA-1.5-7B",           336,  704),
]
PIM_MODES = ["bank", "bg", "buffer"]


def main():
    print("PIM mode comparison -- bank/bg/buffer (paper sec.5)")
    results = []
    for model, img, lin in MODELS:
        per_model = {"model": model, "image_size": img, "lin": lin,
                     "modes": []}
        for pim_mode in PIM_MODES:
            m = sr.run(
                model=model, system="dgx-attacc", gpu="H100",
                ngpu=1, tp=1, num_attacc=1, num_hbm=5, interface="NVLINK4",
                pim=pim_mode, lin=lin, lout=128, batch=1,
                image_size=img,
                prefill_chunk=512, prefill_samples=8, max_L=2048,
                powerlimit=True, ffopt=True, pipeopt=True, word=2,
            )
            s = m.get("s_time") if m else None
            g = m.get("g_time") if m else None
            energy = m.get("g_energy") if m else None
            per_model["modes"].append({
                "pim_mode": pim_mode, "s_ms": s, "g_ms": g,
                "g_energy_nJ": energy,
                "total_ms": sr.e2e_ms(m, 128),
            })
            print("  {:25s} {:8s}: s={:>7.2f}ms g={:>5.2f}ms".format(
                model, pim_mode, s or -1, g or -1))
        results.append(per_model)

    save("pim_mode_compare",
         {"pim_modes": PIM_MODES,
          "platform": "H100 x 1 S1 dgx-attacc",
          "note": "Expected ordering: bank fastest > bg > buffer (per AttAcc sec.5.4)"},
         {"models": results})
    print("Done")


if __name__ == "__main__":
    main()
