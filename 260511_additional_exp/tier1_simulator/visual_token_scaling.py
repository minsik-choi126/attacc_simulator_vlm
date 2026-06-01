"""B3 -- Visual token sensitivity sweep.

Sweep image_size for the two VLMs that handle multi-resolution input
well (Qwen2.5-VL native dynamic, LLaVA-Next anyres) and report how
AttAcc speedup scales with visual token count.

Hypothesis: more visual tokens -> longer prefill -> attention compute
crosses into memory-bound regime -> AttAcc gain grows.  Supports the
"high-res / video VLM extension" claim.

Output: results/visual_token_scaling.json
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

# Pull Transformer to call compute_visual_tokens for annotation only.
sys.path.insert(0, str(HERE.parents[2]))
from src.model import Transformer
from src.config import make_model_config
from src.type import DataType


MODELS = [
    "Qwen2.5-VL-7B",
    "LLaVA-Next-Mistral-7B",
]
IMAGE_SIZES = [336, 672, 1008, 1344]
BATCHES = [1, 4]
LIN = 704
LOUT = 128


def _run(model, system, batch, image_size):
    return sr.run(
        model=model, system=system, gpu="A6000",
        ngpu=1, tp=1, num_attacc=1, num_hbm=5,
        interface="NVLINK_BRIDGE", pim="bank",
        lin=LIN, lout=LOUT, batch=batch,
        image_size=image_size,
        prefill_chunk=512, prefill_samples=8,
        max_L=4096,
        powerlimit=False, ffopt=True, pipeopt=False,
        word=2,
    )


def _visual_tokens(model_name, image_size):
    """Use Transformer.compute_visual_tokens() for annotation."""
    cfg = make_model_config(model_name, DataType.W16A16)
    t = Transformer(cfg, tensor_parallel=1)
    return t.compute_visual_tokens(image_size)


def main():
    print("B3 -- Visual token sensitivity sweep")
    rows = []
    for model in MODELS:
        print(f"\n=== {model} ===")
        for img in IMAGE_SIZES:
            try:
                vis_tok = _visual_tokens(model, img)
            except Exception as e:
                print(f"  img={img}: visual token compute failed ({e})")
                vis_tok = None
            for b in BATCHES:
                dgx = _run(model, "dgx", b, img)
                att = _run(model, "dgx-attacc", b, img)
                if dgx is None or att is None:
                    print(f"  img={img} b={b}: sim_fail")
                    rows.append({"model": model, "image_size": img,
                                 "batch": b, "status": "sim_fail"})
                    continue
                s_d, g_d = dgx.get("s_time"), dgx.get("g_time")
                s_a, g_a = att.get("s_time"), att.get("g_time")
                if not all(v is not None for v in (s_d, g_d, s_a, g_a)):
                    rows.append({"model": model, "image_size": img,
                                 "batch": b, "status": "no_output"})
                    continue
                e_d = s_d + g_d * (LOUT - 1)
                e_a = s_a + g_a * (LOUT - 1)
                speedup = e_d / e_a if e_a else None
                pref_speedup = s_d / s_a if s_a else None
                print(f"  img={img:>4d}  visual_tokens={vis_tok}  "
                      f"b={b}  e2e {speedup:.2f}x  prefill {pref_speedup:.2f}x")
                rows.append({
                    "model": model, "image_size": img,
                    "visual_tokens": vis_tok,
                    "batch": b,
                    "s_dgx_ms": s_d * 1000.0,
                    "s_attacc_ms": s_a * 1000.0,
                    "g_dgx_ms_per_tok": g_d * 1000.0,
                    "g_attacc_ms_per_tok": g_a * 1000.0,
                    "e2e_dgx_ms": e_d * 1000.0,
                    "e2e_attacc_ms": e_a * 1000.0,
                    "e2e_speedup": speedup,
                    "prefill_speedup": pref_speedup,
                    "status": "ok",
                })

    save("visual_token_scaling",
         {"models": MODELS, "image_sizes": IMAGE_SIZES,
          "batches": BATCHES, "lin": LIN, "lout": LOUT},
         {"rows": rows})
    print("\nSaved -> results/visual_token_scaling.json")


if __name__ == "__main__":
    main()
