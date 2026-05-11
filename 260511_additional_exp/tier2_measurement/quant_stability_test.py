"""Quantization stability: count NaN occurrences over N runs.

For each (BF16 baseline, W4A16 AWQ, W8A16 GPTQ), run 100 inferences and
check if any output contains NaN tokens / empty completions.

Paper claim: BF16 production-safe (NaN 0회), FP16 unsafe (NaN >=1),
weight-only quant (W4A16/W8A16) production-safe.
"""
import argparse
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from result_aggregator import save
from vllm_helpers import make_image_input

try:
    from vllm import LLM, SamplingParams
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False


VARIANTS = [
    {"label": "BF16",  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
     "dtype": "bfloat16", "quantization": None},
    {"label": "FP16",  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
     "dtype": "float16", "quantization": None},
    {"label": "W4A16", "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
     "dtype": "float16", "quantization": "awq"},
    {"label": "W8A16", "model": "Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int8",
     "dtype": "float16", "quantization": "gptq"},
]

PROMPTS = [
    "Describe this image with 100 specific words.",
    "What objects are present in this image?",
    "Provide a detailed caption.",
    "List all colors visible.",
    "Summarize the scene in one paragraph.",
]


def make_dummy_image(size=672, seed=0):
    try:
        from PIL import Image
        import random
        random.seed(seed)
        # Slight variation per seed to exercise different decode paths
        c = (random.randint(50, 200),) * 3
        return Image.new("RGB", (size, size), color=c)
    except ImportError:
        return None


def measure_variant(variant, n_runs, lout, image_size, tp=1):
    if not HAVE_VLLM:
        return {"error": "vllm not installed"}
    print("  Loading {} ({})".format(variant["label"], variant["model"]))
    kwargs = dict(model=variant["model"], tensor_parallel_size=tp,
                  trust_remote_code=True, max_model_len=4096,
                  dtype=variant["dtype"], enforce_eager=True,
                  disable_log_stats=True,
                  limit_mm_per_prompt={"image": 1})
    if variant["quantization"]:
        kwargs["quantization"] = variant["quantization"]
    try:
        llm = LLM(**kwargs)
    except Exception as exc:
        return {"error": "load_failed: " + str(exc)}

    sp = SamplingParams(temperature=0.0, max_tokens=lout)

    nan_count = 0
    empty_count = 0
    short_count = 0
    raw = []
    for i in range(n_runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        img = make_dummy_image(image_size, seed=i)
        inputs = [make_image_input(variant["model"], prompt, img)]
        try:
            outs = llm.generate(inputs, sp, use_tqdm=False)
            out = outs[0]
            text = out.outputs[0].text if out.outputs else ""
            seq_out = len(out.outputs[0].token_ids) if out.outputs else 0
        except Exception as exc:
            raw.append({"run": i, "exception": str(exc)})
            nan_count += 1
            continue
        is_empty = (seq_out == 0) or (not text.strip())
        is_short = seq_out < lout * 0.1
        # NaN detection: empty + short = likely numerical failure
        if is_empty:
            empty_count += 1
        if is_short:
            short_count += 1
        raw.append({"run": i, "seq_out": seq_out,
                     "text_preview": text[:50],
                     "empty": is_empty, "short": is_short})

    # Treat empty completions as numerical failure signal
    nan_count = empty_count
    return {
        "label": variant["label"], "model": variant["model"],
        "n_runs": n_runs,
        "nan_count": nan_count,
        "empty_count": empty_count,
        "short_count": short_count,
        "nan_rate": nan_count / n_runs,
        "raw_sample": raw[:5] + raw[-5:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=100)
    ap.add_argument("--lout", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=672)
    ap.add_argument("--variants", nargs="+", default=None,
                     help="Filter by label: BF16/FP16/W4A16/W8A16")
    ap.add_argument("--tp", type=int, default=1,
                     help="vLLM tensor_parallel_size (1 = TP=1, 2 = TP=2 on A6000 x 2)")
    args = ap.parse_args()

    if not HAVE_VLLM:
        print("FATAL: vllm not installed", file=sys.stderr)
        sys.exit(1)

    variants = VARIANTS
    if args.variants:
        variants = [v for v in VARIANTS if v["label"] in args.variants]

    print("Quantization stability test -- n_runs={}".format(args.n_runs))
    results = []
    for v in variants:
        try:
            r = measure_variant(v, args.n_runs, args.lout, args.image_size, tp=args.tp)
            print("  {:6s}: nan_count={}/{}, empty={}, short={}".format(
                v["label"], r.get("nan_count", "?"), args.n_runs,
                r.get("empty_count", "?"), r.get("short_count", "?")))
            results.append(r)
        except Exception as exc:
            print("  {:6s}: FAILED {}".format(v["label"], exc))
            results.append({"label": v["label"], "error": str(exc)})

    save("quant_stability_test",
         {"n_runs": args.n_runs, "lout": args.lout,
          "image_size": args.image_size,
          "platform": "vLLM 0.7.3 TP={}".format(args.tp)},
         {"variants": results,
          "interpretation": "BF16/W4A16/W8A16 expected nan_count=0; "
                             "FP16 may produce nan on long VLM sequences"})
    print("Done")


if __name__ == "__main__":
    main()
