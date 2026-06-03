"""SLO-throughput curves (AttAcc paper Fig.14 style).

Sweep batch from 1 -> 64. For each batch, use simulator decode latency
(`g_time`) as ITL in ms/token. Plot throughput (tokens/sec) vs per-token
latency SLO.

For each VLM x (dgx baseline, dgx-attacc A1):
  - For SLO ∈ {30, 50, 70, 100, 150 ms}, find max batch that stays below SLO
  - Throughput = batch * 1000 / ITL_ms tokens/sec
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

MODELS = [
    ("Qwen3-VL-4B",            672,  569),
    ("Qwen2.5-VL-7B",          672,  704),
    ("LLaVA-1.5-7B",           336,  704),
]
BATCHES = [1, 2, 4, 8, 16, 32, 64]
LOUT = 128
SLO_LIST_MS = [30, 50, 70, 100, 150, 200]
SYSTEMS = [("dgx", 1, "GPU only"),
           ("dgx-attacc", 1, "AttAcc A1")]
# Future hook, intentionally not used in the first paper pass.
FUTURE_SYSTEMS = [("dgx-attacc", 2, "AttAcc A2")]


def measure(model, img, lin, batch, system, ngpu=1):
    m = sr.run(
        model=model, system=system, gpu=SIM_GPU,
        ngpu=ngpu, tp=ngpu, num_attacc=ngpu, num_hbm=5, interface=SIM_INTERFACE,
        pim="bank", lin=lin, lout=LOUT, batch=batch,
        image_size=img,
        prefill_chunk=512, prefill_samples=8, max_L=4096,
        powerlimit=True, ffopt=True, pipeopt=True, word=2,
    )
    if m is None:
        return None
    s = m.get("s_time")
    g = m.get("g_time")
    if s is None or g is None:
        return None
    # ITL approximation: g_time per token
    itl_per_tok = g
    return {"s_ms": s, "g_ms_per_tok": g,
            "e2e_ms": sr.e2e_ms(m, LOUT),
            "throughput_tok_per_sec": batch * 1000 / max(itl_per_tok, 1e-9)}


def find_slo_throughput(curve, slo_ms):
    """Max throughput where ITL (g_ms_per_tok) stays within SLO."""
    best = None
    for pt in curve:
        if pt["g_ms_per_tok"] <= slo_ms and (best is None or
                                              pt["throughput_tok_per_sec"] > best["throughput_tok_per_sec"]):
            best = pt
    return best


def main():
    print("SLO-throughput sweep")
    results = []
    for model, img, lin in MODELS:
        per_model = {"model": model, "image_size": img, "lin": lin,
                     "systems": []}
        for system, ngpu, label in SYSTEMS:
            curve = []
            for b in BATCHES:
                r = measure(model, img, lin, b, system, ngpu)
                if r is None:
                    continue
                r["batch"] = b
                curve.append(r)
            slo_points = []
            for slo in SLO_LIST_MS:
                best = find_slo_throughput(curve, slo)
                slo_points.append({"slo_per_token_ms": slo,
                                    "best": best})
            per_model["systems"].append({
                "system": system, "ngpu": ngpu, "label": label,
                "curve": curve, "slo_points": slo_points,
            })
            print("  {:25s} {:10s} (ngpu={}):  pts={}".format(
                model, label, ngpu, len(curve)))
        results.append(per_model)

    meta = {"batches": BATCHES, "lout": LOUT,
            "slo_per_token_ms": SLO_LIST_MS,
            "deployment_scope": "A1 TP=1 only",
            "future_hooks": [label for _, _, label in FUTURE_SYSTEMS],
            "platform": f"{HOST} simulator A1",
            "host_detected": HOST}
    payload = {"models": results}
    save("slo_throughput", meta, payload)
    save(f"slo_throughput_{HOST.lower()}", meta, payload)
    print("Done")


if __name__ == "__main__":
    main()
