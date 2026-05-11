"""Inter-GPU interface comparison via simulator.

Compares NVLink Bridge (A6000 workstation, 112 GB/s) vs NVLink3 (DGX-A100,
600 GB/s) vs NVLink4 (DGX-H100, 900 GB/s). At TP=1 inter-GPU g2g is 0;
to exercise the interface we also run TP=2 (--ngpu 2 --tp 2 --num_attacc 2).
NOTE: TP=2 vLLM measurement is driver-545+ blocked; simulator works fine.

Default deployment is A6000/NVLink Bridge so the A1/A2 paper scenario is
the primary row; NVLINK3/NVLINK4 stay as upstream reference rows.
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
INTERFACES = ["NVLINK_BRIDGE", "NVLINK3", "NVLINK4"]


def main():
    print("Inter-GPU interface comparison (simulator only): {}".format(INTERFACES))
    rows = []
    for model, img, lin in MODELS:
        # TP=1 reference (no inter-GPU traffic)
        for interface in INTERFACES:
            for ngpu in [1, 2]:
                m = sr.run(
                    model=model, system="dgx-attacc", gpu="A6000",
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
                print("  {:25s} ngpu={} {:14s}: s={:>7.2f}ms g={:>5.2f}ms "
                      "g2g={:>6.4f}ms/step".format(
                          model, ngpu, interface, s or -1, g or -1, g2g or 0))

    save("nvlink_compare",
         {"interfaces": INTERFACES, "ngpus": [1, 2],
          "platform": "simulator only (TP=2 vLLM 측정은 driver 545+ 필요)"},
         {"rows": rows})
    print("Done")


if __name__ == "__main__":
    main()
