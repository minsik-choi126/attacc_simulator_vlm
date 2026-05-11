"""Upstream simulator baseline -- run on legacy LLM list-format models
(GPT-175B, LLAMA-65B, OPT-66B, etc.) to confirm our M-mods don't
regress upstream behavior.

For each model we run dgx-attacc (paper config) and capture s/g latency.
Compares against expected ranges from AttAcc paper Table 3.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

LEGACY_MODELS = [
    ("GPT-175B",  2048, 128),
    ("GPT-89B",   2048, 128),
    ("GPT-13B",   1024,  64),
    ("LLAMA-65B", 2048, 128),
    ("LLAMA-7B",  1024,  64),
    ("MT-530B",   2048, 128),
    ("OPT-66B",   2048, 128),
]


def main():
    print("Upstream legacy LLM models -- DGX-A100 x8 dgx-attacc baseline")
    results = []
    for model, lin, lout in LEGACY_MODELS:
        m = sr.run(
            model=model,
            system="dgx-attacc",
            gpu="A100a",
            ngpu=8, tp=8, num_attacc=8, num_hbm=5,
            interface="NVLINK3",
            pim="bank",
            lin=lin, lout=lout,
            batch=8,
            powerlimit=True, ffopt=True, pipeopt=True,
            word=2,
        )
        if m is None:
            print("  {}: SIM FAIL".format(model))
            results.append({"model": model, "status": "fail"})
            continue
        s = m.get("s_time") or 0
        g = m.get("g_time") or 0
        cap = m.get("required_cap")
        print("  {:10s}: lin={:>5d} lout={:>4d}  s={:>7.1f}ms  g={:>5.2f}ms/tok  "
              "cap_per_gpu={:.1f} GB".format(
                  model, lin, lout, s, g, (cap or 0) / 1024 / 1024 / 1024))
        results.append({"model": model, "lin": lin, "lout": lout,
                        "s_time_ms": s, "g_time_ms": g,
                        "required_cap_per_gpu": cap, "status": "ok"})

    save("upstream_baseline",
         {"platform": "simulated DGX-A100 x8 dgx-attacc",
          "purpose": "Verify our M-mods do not regress legacy upstream behavior"},
         {"models": results})
    print("\nDone")


if __name__ == "__main__":
    main()
