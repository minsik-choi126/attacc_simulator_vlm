"""Prompt pattern matrix -- L_in x L_out 9 combo.

For each (text_in_tokens, lout) ∈ {short, medium, long}^2, measure
TTFT/ITL/E2E on Qwen2.5-VL-7B. Captures AttAcc paper Fig.14 style
L_in/L_out sensitivity grid.
"""
import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from result_aggregator import save, summarize
from vllm_helpers import make_image_input

try:
    from vllm import LLM, SamplingParams
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False


# Text-length presets (will be padded with repetition to hit target tokens)
LIN_PATTERNS = [
    ("short",  "Describe this image briefly."),
    ("medium", "Describe this image with 100 specific words covering "
               "objects, colors, composition, mood, and any text content."),
    ("long",   ("Provide an exhaustive analysis of this image including "
                "all visible objects, their spatial relationships, colors and "
                "textures, the overall composition and balance, the lighting "
                "conditions and time of day, any text or symbols present, the "
                "emotional or narrative content, possible cultural references, "
                "and any technical aspects of the photography. ") * 3),
]
LOUT_VALUES = [32, 128, 512]


def make_dummy_image(size=672):
    try:
        from PIL import Image
        return Image.new("RGB", (size, size), color=(128, 128, 128))
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--image_size", type=int, default=672)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    if not HAVE_VLLM:
        print("FATAL: vllm not installed", file=sys.stderr)
        sys.exit(1)

    print("Prompt pattern matrix -- {}".format(args.model))
    llm = LLM(model=args.model, tensor_parallel_size=1,
               trust_remote_code=True, max_model_len=8192,
               dtype="bfloat16", enforce_eager=True,
               disable_log_stats=True,
               limit_mm_per_prompt={"image": 1})

    img = make_dummy_image(args.image_size)
    rows = []
    for lin_label, prompt in LIN_PATTERNS:
        for lout in LOUT_VALUES:
            sp = SamplingParams(temperature=0.0, max_tokens=lout,
                                 min_tokens=lout, ignore_eos=True)
            inputs = [make_image_input(args.model, prompt, img)]
            per = []
            for i in range(args.warmup + args.repeats):
                outs = llm.generate(inputs, sp, use_tqdm=False)
                out = outs[0]
                mt = out.metrics
                ttft = (mt.first_token_time - mt.arrival_time) * 1000.0
                e2e = (mt.finished_time - mt.arrival_time) * 1000.0
                seq_in = len(out.prompt_token_ids) if out.prompt_token_ids else None
                seq_out = len(out.outputs[0].token_ids) if out.outputs else lout
                itl = (e2e - ttft) / max(seq_out - 1, 1)
                if i >= args.warmup:
                    per.append({"ttft_ms": ttft, "e2e_ms": e2e,
                                 "itl_ms": itl, "seq_in": seq_in})
            ttft_stats = summarize([p["ttft_ms"] for p in per])
            itl_stats = summarize([p["itl_ms"] for p in per])
            e2e_stats = summarize([p["e2e_ms"] for p in per])
            rows.append({
                "lin_label": lin_label, "lout": lout,
                "seq_in_avg": per[0]["seq_in"] if per else None,
                "ttft": ttft_stats, "itl": itl_stats, "e2e": e2e_stats,
            })
            print("  {:>6s}xlout={:>4d}: seq_in={:>4} TTFT={:.1f}ms ITL={:.3f}".format(
                lin_label, lout, per[0]["seq_in"] if per else "?",
                ttft_stats["p50"] or 0, itl_stats["p50"] or 0))

    save("prompt_pattern_matrix",
         {"model": args.model, "lin_patterns": [p[0] for p in LIN_PATTERNS],
          "lout_values": LOUT_VALUES, "image_size": args.image_size,
          "repeats": args.repeats, "warmup": args.warmup,
          "platform": "H100 x 1 vLLM 0.7.3 bf16"},
         {"rows": rows})
    print("Done")


if __name__ == "__main__":
    main()
