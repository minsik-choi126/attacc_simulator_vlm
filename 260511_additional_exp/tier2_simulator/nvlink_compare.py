"""NVLink3 vs NVLink4 comparison via simulator.

At TP=1 the inter-GPU g2g is 0 (single GPU). To exercise NVLink we
also run a TP=2-equivalent (simulator allows --ngpu 2 --tp 2 --num_attacc 2).
NOTE: TP=2 vLLM measurement is driver 545+ blocked; simulator works fine.
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
INTERFACES = ["NVLINK3", "NVLINK4"]


def main():
    print("NVLink3 vs NVLink4 comparison (simulator only)")
    rows = []
    for model, img, lin in MODELS:
        # TP=1 reference (no inter-GPU traffic)
        for interface in INTERFACES:
            for ngpu in [1, 2]:
                m = sr.run(
                    model=model, system="dgx-attacc", gpu="H100",
                    ngpu=ngpu, tp=ngpu, num_attacc=ngpu, num_hbm=5,
                    interface=interface,
                    pim="bank", lin=lin, lout=128, batch=1,
                    image_size=img,
                    prefill_chunk=512, prefill_samples=8, max_L=2048,
                    powerlimit=True, ffopt=True, pipeopt=True, word=2,
                )
                s = m.get("s_time") if m else None
                g = m.get("g_time") if m else None
                # g2g_comm column captures comm time per decode step
                g2g = m.get("g2g_comm") if m else None
                rows.append({
                    "model": model, "ngpu": ngpu,
                    "interface": interface, "lin": lin,
                    "s_ms": s, "g_ms": g, "g2g_comm_per_step": g2g,
                })
                print("  {:25s} ngpu={} {:8s}: s={:>7.2f}ms g={:>5.2f}ms "
                      "g2g={:>6.4f}ms/step".format(
                          model, ngpu, interface, s or -1, g or -1, g2g or 0))

    save("nvlink_compare",
         {"interfaces": INTERFACES, "ngpus": [1, 2],
          "platform": "simulator only (TP=2 vLLM 측정은 driver 545+ 필요)"},
         {"rows": rows})
    print("Done")


if __name__ == "__main__":
    main()
