"""Routing mode comparison: conservative / optimistic / list.

For each VLM (DeepStack model = Qwen3-VL highlighted), measure E2E latency
under each routing mode. Show DeepStack injection effect (M9) on list mode.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

MODELS = [
    {"model": "Qwen3-VL-4B",           "image_size": 672, "lin": 569,  "ds": True},
    {"model": "Qwen2.5-VL-7B",         "image_size": 672, "lin": 704,  "ds": False},
    {"model": "LLaVA-1.5-7B",          "image_size": 336, "lin": 704,  "ds": False},
    {"model": "LLaVA-Next-Mistral-7B", "image_size": 672, "lin": 3008, "ds": False},
]
MODES = ["conservative", "optimistic", "list"]


def main():
    print("Routing mode comparison -- A6000 x 1 A1 dgx-attacc")
    all_results = []
    for cfg in MODELS:
        per_model = {**cfg, "modes": []}
        for mode in MODES:
            m = sr.run(
                model=cfg["model"], system="dgx-attacc", gpu="A6000",
                ngpu=1, tp=1, num_attacc=1, num_hbm=5, interface="NVLINK_BRIDGE",
                pim="bank", lin=cfg["lin"], lout=128, batch=1,
                image_size=cfg["image_size"],
                prefill_chunk=512, prefill_samples=8, max_L=4096,
                powerlimit=True, ffopt=True, pipeopt=True, word=2,
                routing=mode,
            )
            s = m.get("s_time") if m else None
            g = m.get("g_time") if m else None
            per_model["modes"].append({
                "mode": mode, "s_time_ms": s, "g_time_ms": g,
                "total_ms": sr.e2e_ms(m, 128),
            })
            print("  {:25s} {:13s} s={:>7.2f}ms g={:>5.2f}ms".format(
                cfg["model"], mode, s or -1, g or -1))
        all_results.append(per_model)

    save("routing_mode_compare",
         {"modes": MODES, "platform": "A6000 x 1 A1 dgx-attacc"},
         {"models": all_results,
          "note": "list mode for DeepStack model auto-forced (M7)"})
    print("Done")


if __name__ == "__main__":
    main()
