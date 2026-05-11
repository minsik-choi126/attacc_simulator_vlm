"""Image size sweep -- vLLM TTFT vs image_size.

Image sizes: {336, 448, 672, 1008}. Each -> number of visual tokens varies.
Measures TTFT scaling vs visual token count for Qwen2.5-VL-7B baseline.

Paper figure: TTFT vs visual_token_count, log-log slope.
"""
import argparse
import pathlib
import statistics
import subprocess
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


SIZES = [336, 448, 672, 1008]


def make_dummy_image(size):
    try:
        from PIL import Image
        return Image.new("RGB", (size, size), color=(128, 128, 128))
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--lout", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1,
                     help="vLLM tensor_parallel_size (1 = TP=1, 2 = TP=2 on A6000 x 2)")
    args = ap.parse_args()

    if not HAVE_VLLM:
        print("FATAL: vllm not installed", file=sys.stderr)
        sys.exit(1)

    print("Image size sweep -- {} (TP={})".format(args.model, args.tp))
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
               trust_remote_code=True, max_model_len=8192,
               dtype="bfloat16", enforce_eager=True,
               disable_log_stats=True,
               limit_mm_per_prompt={"image": 1})
    sp = SamplingParams(temperature=0.0, max_tokens=args.lout,
                         min_tokens=args.lout, ignore_eos=True)
    prompt = "Describe this image with 100 specific words."

    rows = []
    for size in args.sizes:
        img = make_dummy_image(size)
        if img is None:
            continue
        inputs = [make_image_input(args.model, prompt, img)]
        per_size = []
        for i in range(args.warmup + args.repeats):
            outs = llm.generate(inputs, sp, use_tqdm=False)
            out = outs[0]
            mt = out.metrics
            ttft_ms = (mt.first_token_time - mt.arrival_time) * 1000.0
            e2e_ms = (mt.finished_time - mt.arrival_time) * 1000.0
            seq_in = len(out.prompt_token_ids) if out.prompt_token_ids else None
            seq_out = len(out.outputs[0].token_ids) if out.outputs else args.lout
            itl_ms = (e2e_ms - ttft_ms) / max(seq_out - 1, 1)
            if i >= args.warmup:
                per_size.append({"ttft_ms": ttft_ms, "e2e_ms": e2e_ms,
                                  "itl_ms": itl_ms, "seq_in": seq_in})
        rows.append({
            "image_size": size,
            "seq_in_avg": statistics.fmean([p["seq_in"] for p in per_size
                                              if p["seq_in"]])
                            if any(p["seq_in"] for p in per_size) else None,
            "ttft": summarize([p["ttft_ms"] for p in per_size]),
            "itl": summarize([p["itl_ms"] for p in per_size]),
            "e2e": summarize([p["e2e_ms"] for p in per_size]),
            "raw": per_size,
        })
        print("  size={:>5d}  seq_in={:>5.0f}  TTFT p50={:.1f}ms  ITL p50={:.3f}".format(
            size, rows[-1]["seq_in_avg"] or 0,
            rows[-1]["ttft"]["p50"] or 0,
            rows[-1]["itl"]["p50"] or 0))

    save("image_size_sweep",
         {"model": args.model, "sizes": args.sizes, "lout": args.lout,
          "repeats": args.repeats, "warmup": args.warmup,
          "platform": "vLLM 0.7.3 bf16 TP={}".format(args.tp)},
         {"rows": rows})
    print("Done")


if __name__ == "__main__":
    main()
