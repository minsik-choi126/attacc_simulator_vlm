"""B2 -- Prefill vs decode PIM contribution decomposition per VLM.

Each VLM's e2e AttAcc speedup breaks down into a prefill component
and a decode component:

    e2e_speedup     = (s_dgx + g_dgx*(lout-1)) / (s_att + g_att*(lout-1))
    prefill_speedup = s_dgx / s_att
    decode_speedup  = g_dgx / g_att
    prefill_contrib_pct = 100 * (s_dgx - s_att) / (e2e_dgx - e2e_att)
    decode_contrib_pct  = 100 - prefill_contrib_pct

VLM-specific claim: visual tokens lengthen prefill into memory-bound
regime, so prefill_speedup > 1.0 even at batch=1.  In LLM (paper c2
argument) prefill is compute-bound and prefill_speedup ~ 1.0.

Output: results/prefill_decomp_vlm.json
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save


MODELS = [
    ("Qwen3-VL-4B",            672, 569),
    ("Qwen2.5-VL-7B",          672, 704),
    ("InternVL3-8B-hf",        448, 704),
    ("LLaVA-1.5-7B",           336, 704),
    ("LLaVA-Next-Mistral-7B",  672, 704),
]
BATCHES = [1, 4, 8]
LOUT = 128


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


def decompose(dgx, att):
    s_d = dgx.get("s_time")
    g_d = dgx.get("g_time")
    s_a = att.get("s_time")
    g_a = att.get("g_time")
    if not all(v is not None for v in (s_d, g_d, s_a, g_a)):
        return None
    e_d = s_d + g_d * (LOUT - 1)
    e_a = s_a + g_a * (LOUT - 1)
    return {
        "s_dgx_ms":         s_d * 1000.0,
        "s_attacc_ms":      s_a * 1000.0,
        "g_dgx_ms_per_tok": g_d * 1000.0,
        "g_attacc_ms_per_tok": g_a * 1000.0,
        "e2e_dgx_ms":       e_d * 1000.0,
        "e2e_attacc_ms":    e_a * 1000.0,
        "prefill_speedup": s_d / s_a if s_a else None,
        "decode_speedup":  g_d / g_a if g_a else None,
        "e2e_speedup":     e_d / e_a if e_a else None,
        "prefill_contrib_pct": (100.0 * (s_d - s_a) / (e_d - e_a)
                                if (e_d - e_a) else None),
    }


def main():
    print("B2 -- Prefill vs decode PIM contribution per VLM")
    rows = []
    for model, img, lin in MODELS:
        print(f"\n=== {model} (img={img}, lin={lin}) ===")
        for b in BATCHES:
            dgx = _run(model, "dgx", b, lin, img)
            att = _run(model, "dgx-attacc", b, lin, img)
            if dgx is None or att is None:
                print(f"  b={b}: sim failed")
                rows.append({"model": model, "batch": b, "status": "sim_fail"})
                continue
            dec = decompose(dgx, att)
            if dec is None:
                print(f"  b={b}: decompose failed")
                rows.append({"model": model, "batch": b, "status": "decompose_fail"})
                continue
            print(f"  b={b:>2d}  prefill {dec['prefill_speedup']:.2f}x  "
                  f"decode {dec['decode_speedup']:.2f}x  "
                  f"e2e {dec['e2e_speedup']:.2f}x  "
                  f"prefill_contrib {dec['prefill_contrib_pct']:>5.1f}%")
            rows.append({"model": model, "image_size": img, "lin": lin,
                         "batch": b, **dec, "status": "ok"})

    save("prefill_decomp_vlm",
         {"models": [m[0] for m in MODELS], "batches": BATCHES, "lout": LOUT,
          "method": "prefill_contrib = 100 * (s_dgx-s_att) / (e2e_dgx-e2e_att)"},
         {"rows": rows})
    print("\nSaved -> results/prefill_decomp_vlm.json")


if __name__ == "__main__":
    main()
