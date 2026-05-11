"""Roofline analysis per VLM -- identify which layers benefit from PIM.

For each VLM x each sub-layer (qkv / score / softmax / context / proj / ff),
compute arithmetic intensity (FLOPs/byte) and compare against H100 ridge
(989 TFLOPS / 3352 GB/s = 295 ops/byte) and AttAcc PIM ridge (per-AttAcc).

Layer below ridge = memory-bound = PIM benefit candidate.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from result_aggregator import save


# Hardware ridges (ops/byte at FP16)
H100_RIDGE = 989.4e12 / 3352e9         # ~295
ATTACC_RIDGE_PER = 670.4e9 * 9         # internal scale 9
# Effective PIM ridge per AttAcc (no compute saturation in AttAcc)
# AttAcc has GEMV+softmax dedicated, so ridge is roughly 1 op/byte (BW-bound).
# We mark layers as PIM-target if AI(GPU) < H100 ridge (i.e., GPU memory-bound).


MODELS = {
    "Qwen3-VL-4B": dict(
        ndec=36, hdim=2560, n_q=32, n_kv=8, dhead=128, ff=9728,
    ),
    "Qwen2.5-VL-7B": dict(
        ndec=28, hdim=3584, n_q=28, n_kv=4, dhead=128, ff=18944,
    ),
    "InternVL3-8B-hf": dict(
        ndec=28, hdim=3584, n_q=28, n_kv=4, dhead=128, ff=18944,
    ),
    "LLaVA-1.5-7B": dict(
        ndec=32, hdim=4096, n_q=32, n_kv=32, dhead=128, ff=11008,
    ),
    "LLaVA-Next-Mistral-7B": dict(
        ndec=32, hdim=4096, n_q=32, n_kv=8, dhead=128, ff=14336,
    ),
}


def ai_qkv(cfg, L, dbyte=2):
    """qkv = (B*L) x hdim x qkv_proj_out matmul."""
    qkv_out = cfg["n_q"] * cfg["dhead"] + 2 * cfg["n_kv"] * cfg["dhead"]
    flops = 2 * L * cfg["hdim"] * qkv_out
    bytes_ = (L * cfg["hdim"] + cfg["hdim"] * qkv_out + L * qkv_out) * dbyte
    return flops / bytes_, flops, bytes_


def ai_score(cfg, L, dbyte=2):
    """score: per Q head LxL matmul with K (n_kv heads via GQA broadcast)."""
    flops = 2 * cfg["n_q"] * L * L * cfg["dhead"]
    bytes_ = (cfg["n_kv"] * L * cfg["dhead"]  # K
              + cfg["n_q"] * L * cfg["dhead"]  # Q
              + cfg["n_q"] * L * L) * dbyte    # scores
    return flops / bytes_, flops, bytes_


def ai_context(cfg, L, dbyte=2):
    """context: scores x V."""
    flops = 2 * cfg["n_q"] * L * L * cfg["dhead"]
    bytes_ = (cfg["n_q"] * L * L                # softmax
              + cfg["n_kv"] * L * cfg["dhead"]  # V
              + cfg["n_q"] * L * cfg["dhead"]) * dbyte  # out
    return flops / bytes_, flops, bytes_


def ai_ffn(cfg, L, gated=True, dbyte=2):
    """gated FFN: 3 GEMMs of (L x ff) and 1 of (L x hdim, ff x hdim)."""
    if gated:
        # gate, up, down -- 3 GEMMs
        flops = 3 * 2 * L * cfg["hdim"] * cfg["ff"]
        bytes_ = (3 * (L * cfg["hdim"] + cfg["hdim"] * cfg["ff"] + L * cfg["ff"])
                  ) * dbyte
    else:
        flops = 2 * 2 * L * cfg["hdim"] * cfg["ff"]
        bytes_ = 2 * (L * cfg["hdim"] + cfg["hdim"] * cfg["ff"] + L * cfg["ff"]
                       ) * dbyte
    return flops / bytes_, flops, bytes_


def classify(ai):
    if ai < H100_RIDGE / 3:
        return "memory-bound (PIM target)"
    if ai < H100_RIDGE:
        return "marginal (PIM may help)"
    return "compute-bound (no PIM benefit)"


def main():
    print("Roofline per VLM (H100 ridge ~ {:.1f} ops/byte)".format(H100_RIDGE))
    results = []
    for L_label, L in [("prefill_L569", 569), ("decode_L1", 1)]:
        for model, cfg in MODELS.items():
            rows = []
            for layer_name, fn in [
                ("qkv",      lambda c=cfg: ai_qkv(c, L)),
                ("score",    lambda c=cfg: ai_score(c, L)),
                ("context",  lambda c=cfg: ai_context(c, L)),
                ("ffn",      lambda c=cfg: ai_ffn(c, L, gated=True)),
            ]:
                ai, flops, bytes_ = fn()
                rows.append({"layer": layer_name, "ai": round(ai, 2),
                              "flops": flops, "bytes": bytes_,
                              "verdict": classify(ai)})
            results.append({"regime": L_label, "model": model, "L": L,
                             "layers": rows})
            print("  {:12s} {:25s}".format(L_label, model))
            for r in rows:
                print("    {:8s} AI={:>7.2f}  {}".format(
                    r["layer"], r["ai"], r["verdict"]))

    save("roofline_per_vlm",
         {"H100_ridge_ops_per_byte": round(H100_RIDGE, 2),
          "regimes": ["prefill_L569", "decode_L1"],
          "models": list(MODELS.keys())},
         {"matrix": results})
    print("Done")


if __name__ == "__main__":
    main()
