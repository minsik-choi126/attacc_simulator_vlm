"""W4A16 / W8A16 + PIM analytical projection (simulator-only).

The native simulator supports only DataType.W16A16 (BF16/FP16) and W8A8.
W4A16 (AWQ) and W8A16 (GPTQ-Int8) keep activations at 2 byte but reduce
weight memory traffic. This script projects the BF16 simulator output by
scaling the FC-layer time component by `weight_byte / 2`, which matches
AttAcc paper sec.7.5 (Fig.16) quant-projection methodology.

Decomposition (per-stage):

  s_time = s_fc + s_matmul + s_comm + s_softmax + s_act + s_lnorm + s_x2g
  g_time = g_fc + g_matmul + g_comm + g_etc

Projection (only FC weight load scales):

  s_time_quant = s_time + s_fc * (w_ratio - 1)
  g_time_quant = g_time + g_fc * (w_ratio - 1)

w_ratio = weight_byte / 2 -> 1.00 (BF16), 0.50 (W8A16), 0.25 (W4A16).

Assumptions / caveats:
- FC at small batch is weight-load memory-bound (paper sec.4.1). At very
  large batch FC becomes compute-bound and projection is pessimistic.
- Attention (MATMUL / softmax) keeps activation precision (W4A16 / W8A16
  do not quantize activations) so leave unchanged.
- This is an analytical projection, not a new simulator path. Validate
  against the matched W4A16 / W8A16 vLLM measurement runs in
  tier2_measurement/w4a16_awq_measure.py and w8a16_gptq_measure.py.

Outputs the (PIM gain x quant level) matrix for paper Fig.8 sim panel.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

VLM_CONFIGS = [
    {"model": "Qwen3-VL-4B",            "image_size": 672, "lin": 569},
    {"model": "Qwen2.5-VL-7B",          "image_size": 672, "lin": 704},
    {"model": "InternVL3-8B-hf",        "image_size": 448, "lin": 384},
    {"model": "LLaVA-1.5-7B",           "image_size": 336, "lin": 704},
    {"model": "LLaVA-Next-Mistral-7B",  "image_size": 672, "lin": 3008},
]
LOUT = 128
BATCHES = [1, 4, 8]

QUANT_LEVELS = [
    {"label": "BF16",  "w_ratio": 1.00},
    {"label": "W8A16", "w_ratio": 0.50},
    {"label": "W4A16", "w_ratio": 0.25},
]

CAPTURE = (
    "s_time", "s_fc", "s_matmul", "s_comm", "s_softmax", "s_act", "s_lnorm",
    "s_x2g",
    "g_time", "g_fc", "g_matmul", "g_comm", "g_etc",
    "g_qkv_time", "g_prj_time", "g_ff_time",
    "required_cap",
)


def max_L_for(lin_plus_lout):
    for cap in (2048, 4096, 8192, 16384):
        if cap >= lin_plus_lout:
            return cap
    return lin_plus_lout * 2


def project(metrics, w_ratio):
    """Project FC-portion of latency by weight-byte ratio."""
    if metrics is None:
        return None, None
    s = metrics.get("s_time")
    g = metrics.get("g_time")
    s_fc = metrics.get("s_fc") or 0
    g_fc = metrics.get("g_fc") or 0
    if s is None or g is None:
        return None, None
    s_q = s + s_fc * (w_ratio - 1)
    g_q = g + g_fc * (w_ratio - 1)
    return s_q, g_q


def run_pair(cfg, batch):
    """Return (BF16 baseline metrics dgx, dgx-attacc)."""
    common = dict(
        gpu="A6000", ngpu=1, tp=1, num_attacc=1, num_hbm=5,
        interface="NVLINK_BRIDGE", pim="bank",
        lin=cfg["lin"], lout=LOUT, batch=batch,
        image_size=cfg["image_size"],
        prefill_chunk=512, prefill_samples=8,
        max_L=max_L_for(cfg["lin"] + LOUT),
        powerlimit=True, ffopt=True, pipeopt=True, word=2,
        capture=CAPTURE,
    )
    dgx = sr.run(model=cfg["model"], system="dgx", **common)
    attacc = sr.run(model=cfg["model"], system="dgx-attacc", **common)
    return dgx, attacc


def main():
    print("W4A16 / W8A16 + PIM analytical projection (sim sec.7.5 style)")
    matrix = []
    for cfg in VLM_CONFIGS:
        per_model = {"model": cfg["model"], "image_size": cfg["image_size"],
                     "lin": cfg["lin"], "lout": LOUT, "batches": []}
        for batch in BATCHES:
            dgx_m, acc_m = run_pair(cfg, batch)
            batch_row = {"batch": batch, "quant": {}}
            for q in QUANT_LEVELS:
                w_ratio = q["w_ratio"]
                s_dgx, g_dgx = project(dgx_m, w_ratio)
                s_acc, g_acc = project(acc_m, w_ratio)
                e2e_dgx = (s_dgx + g_dgx * (LOUT - 1)
                            if s_dgx is not None and g_dgx is not None
                            else None)
                e2e_acc = (s_acc + g_acc * (LOUT - 1)
                            if s_acc is not None and g_acc is not None
                            else None)
                speedup = (e2e_dgx / e2e_acc
                            if e2e_dgx and e2e_acc else None)
                batch_row["quant"][q["label"]] = {
                    "w_ratio": w_ratio,
                    "dgx_s_ms": s_dgx, "dgx_g_ms": g_dgx,
                    "attacc_s_ms": s_acc, "attacc_g_ms": g_acc,
                    "dgx_e2e_ms": e2e_dgx, "attacc_e2e_ms": e2e_acc,
                    "speedup_e2e": round(speedup, 3) if speedup else None,
                }
            per_model["batches"].append(batch_row)
            row_str = "  ".join(
                "{}={:.2f}x".format(q["label"],
                                     batch_row["quant"][q["label"]]["speedup_e2e"] or 0)
                for q in QUANT_LEVELS)
            print("  {:25s} batch={:>2d}:  {}".format(
                cfg["model"], batch, row_str))
        matrix.append(per_model)

    save("w4a16_pim_sim",
         {"platform": "A6000 x 1 A1 simulator", "lout": LOUT,
          "batches": BATCHES,
          "method": "analytical projection: s_time + s_fc*(w_ratio-1)",
          "caveats": [
              "FC assumed memory-bound (paper sec.4.1); large-batch overestimate",
              "Activations stay at 2 byte (W4A16, W8A16 = weight-only quant)",
              "Validate against measured w4a16_awq / w8a16_gptq runs",
          ]},
         {"models": matrix})
    print("\nDone -- results/w4a16_pim_sim.json")


if __name__ == "__main__":
    main()
