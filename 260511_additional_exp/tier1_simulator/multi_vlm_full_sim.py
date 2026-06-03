"""Multi-VLM full simulation matrix on A6000 x 1 (A1 deployment).

5 in-framework VLMs x 4 (lin, lout, batch) configs x {dgx GPU-only, dgx-attacc}.
Captures s_time / g_time / energy / capacity, computes per-model PIM speedup.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save
from hw_detect import detect_host, sim_gpu_tag, sim_interface_tag

HOST = detect_host()
SIM_GPU = sim_gpu_tag(HOST)
SIM_INTERFACE = sim_interface_tag(HOST)

# 5 in-framework VLM configs (A6000 x 1, A1 deployment)
VLM_CONFIGS = [
    {"model": "Qwen3-VL-4B",            "image_size": 672, "lin": 569},
    {"model": "Qwen2.5-VL-7B",          "image_size": 672, "lin": 704},
    {"model": "InternVL3-8B-hf",        "image_size": 448, "lin": 384},
    {"model": "LLaVA-1.5-7B",           "image_size": 336, "lin": 704},
    {"model": "LLaVA-Next-Mistral-7B",  "image_size": 672, "lin": 3008},
]

LOUT = 128
BATCHES = [1, 4, 8]


def max_L_for(lin):
    """max_L must cover lin; cap at common power-of-two."""
    for cap in (2048, 4096, 8192, 16384):
        if cap >= lin:
            return cap
    return lin * 2


def run_one(cfg, batch, system):
    return sr.run(
        model=cfg["model"],
        system=system,
        gpu=SIM_GPU,
        ngpu=1, tp=1, num_attacc=1, num_hbm=5,
        interface=SIM_INTERFACE,
        pim="bank",
        lin=cfg["lin"], lout=LOUT, batch=batch,
        image_size=cfg["image_size"],
        prefill_chunk=512, prefill_samples=8,
        max_L=max_L_for(cfg["lin"] + LOUT),
        powerlimit=True, ffopt=True, pipeopt=True,
        word=2,
    )


def total_ms(m):
    return sr.e2e_ms(m, LOUT)


def main():
    print("Multi-VLM full simulation -- A6000 x 1 (A1), dgx vs dgx-attacc")
    matrix = []
    for cfg in VLM_CONFIGS:
        per_model = {"model": cfg["model"], "image_size": cfg["image_size"],
                     "lin": cfg["lin"], "lout": LOUT,
                     "max_L": max_L_for(cfg["lin"] + LOUT),
                     "batches": []}
        for batch in BATCHES:
            base = run_one(cfg, batch, "dgx")
            attacc = run_one(cfg, batch, "dgx-attacc")
            base_total = total_ms(base)
            acc_total = total_ms(attacc)
            speedup = (base_total / acc_total
                        if base_total and acc_total else None)
            per_model["batches"].append({
                "batch": batch,
                "dgx_s_ms": base.get("s_time") if base else None,
                "dgx_g_ms": base.get("g_time") if base else None,
                "attacc_s_ms": attacc.get("s_time") if attacc else None,
                "attacc_g_ms": attacc.get("g_time") if attacc else None,
                "speedup_e2e": round(speedup, 3) if speedup else None,
            })
            print("  {:25s} batch={:>2d}: dgx={:>7.1f}ms  attacc={:>7.1f}ms  "
                  "speedup={:.2f}x".format(
                      cfg["model"], batch,
                      base_total or -1, acc_total or -1, speedup or 0))
        matrix.append(per_model)

    meta = {"platform": f"{HOST} x 1 A1", "host_detected": HOST,
            "lout": LOUT, "batches": BATCHES,
            "system": "dgx vs dgx-attacc"}
    payload = {"models": matrix}
    save("multi_vlm_full_sim", meta, payload)
    save(f"multi_vlm_full_sim_{HOST.lower()}", meta, payload)
    print(f"\nDone -- results/multi_vlm_full_sim{{,_{HOST.lower()}}}.json")


if __name__ == "__main__":
    main()
