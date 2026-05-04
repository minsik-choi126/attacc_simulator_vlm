"""R6/R7 — Paper-grade single-shot TP measurement via vLLM.

Methodology (matches AttAcc simulator R3 baseline contract):
  - Single H100 (TP=1) or 2x H100 (TP=2 via vLLM tensor_parallel_size=2 + NCCL).
  - BF16, batch=1, fixed image (672x672 or model-specific) + text prompt.
  - L_out = 128 generated tokens, ignore_eos so prefill+decode shape is fixed.
  - Reports per-request TTFT (prefill+first-token) and ITL (mean inter-token
    latency over remaining tokens). Aggregates p50/p95/p99 over N repeats with
    warmup discarded.
  - Output JSON includes raw per-request timings so paper-grade analysis can
    re-derive any percentile.

Usage:
  python3 tests/r6_vllm_measurement.py --model Qwen/Qwen3-VL-4B-Instruct \
      --tp 1 --image_size 672 --lout 128 --repeats 12 --warmup 3 \
      --output results/r6_qwen3_vl_4b_tp1_vllm.json

  python3 tests/r6_vllm_measurement.py --model Qwen/Qwen3-VL-4B-Instruct \
      --tp 2 --image_size 672 --lout 128 --repeats 12 --warmup 3 \
      --output results/r6_qwen3_vl_4b_tp2_vllm.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

from PIL import Image


def make_image(width, height):
    return Image.new("RGB", (width, height), color=(127, 127, 127))


def detect_template_text(model_name: str, prompt: str) -> str:
    """Return a chat-templated prompt string with the image placeholder.

    vLLM's `LLM.generate` with `multi_modal_data` substitutes the image at
    the first <image> placeholder. Different model families use different
    chat templates; for paper-grade measurement we follow each family's
    canonical template.
    """
    name = model_name.lower()
    if "qwen3-vl" in name or "qwen2.5-vl" in name or "qwen2-vl" in name:
        return ("<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>"
                "{}<|im_end|>\n<|im_start|>assistant\n").format(prompt)
    if "llava-1.5" in name or "llava-v1.5" in name:
        return "USER: <image>\n{} ASSISTANT:".format(prompt)
    if "llava-v1.6" in name or "llava-1.6" in name or "llava-next" in name:
        return ("[INST] <image>\n{} [/INST]").format(prompt)
    if "internvl" in name:
        return ("<|im_start|>user\n<image>\n{}<|im_end|>"
                "<|im_start|>assistant\n").format(prompt)
    return "{}".format(prompt)


@dataclass
class RequestMetric:
    request_idx: int
    seq_in_len: int
    seq_out_len: int
    e2e_ms: float
    ttft_ms: Optional[float]
    itl_ms: Optional[float]


def run(args):
    import torch
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        os.environ["TORCH_CUDNN_DISABLED"] = "1"
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    width = args.image_width or args.image_size
    height = args.image_height or args.image_size
    image = make_image(width, height)
    prompt_text = args.prompt or "Describe this image with 100 specific words."
    template_text = detect_template_text(args.model, prompt_text)

    init_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 1},
    )
    print("[R6] LLM init kwargs: {}".format(init_kwargs), flush=True)
    llm = LLM(**init_kwargs)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.lout,
        min_tokens=args.lout,
        ignore_eos=True,
    )

    inputs = [{
        "prompt": template_text,
        "multi_modal_data": {
            "image": image
        },
    }] * args.batch

    metrics: List[RequestMetric] = []
    total_runs = args.warmup + args.repeats
    for idx in range(total_runs):
        t0 = time.perf_counter()
        outputs = llm.generate(inputs, sampling, use_tqdm=False)
        t1 = time.perf_counter()

        # Aggregate batch metrics: report median over batch elements per
        # iteration so the per-iteration entry is comparable across batch
        # sizes; raw per-request data preserved through repeats.
        seq_in_len = len(outputs[0].prompt_token_ids) if outputs[
            0].prompt_token_ids else -1
        seq_out_len = len(outputs[0].outputs[0].token_ids)

        ttft_list = []
        itl_list = []
        for out in outputs:
            rm = getattr(out, "metrics", None)
            if rm is not None:
                arrival = getattr(rm, "arrival_time", None)
                first_token = getattr(rm, "first_token_time", None)
                finished = getattr(rm, "finished_time", None)
                if arrival is not None and first_token is not None:
                    ttft_list.append((first_token - arrival) * 1000.0)
                if (first_token is not None and finished is not None
                        and seq_out_len > 1):
                    itl_list.append(
                        (finished - first_token) * 1000.0 / (seq_out_len - 1))

        ttft_ms = sorted(ttft_list)[len(ttft_list) // 2] if ttft_list else None
        itl_ms = sorted(itl_list)[len(itl_list) // 2] if itl_list else None
        e2e_ms = (t1 - t0) * 1000.0

        m = RequestMetric(request_idx=idx,
                          seq_in_len=seq_in_len,
                          seq_out_len=seq_out_len,
                          e2e_ms=e2e_ms,
                          ttft_ms=ttft_ms,
                          itl_ms=itl_ms)
        metrics.append(m)
        is_warm = idx < args.warmup
        marker = "warmup" if is_warm else "measure"
        print("[{}] idx={} batch={} in={} out={} e2e={:.2f}ms ttft={} itl={}".
              format(marker, idx, args.batch, seq_in_len, seq_out_len, e2e_ms,
                     "{:.2f}ms".format(ttft_ms) if ttft_ms is not None else "n/a",
                     "{:.3f}ms".format(itl_ms) if itl_ms is not None else "n/a"),
              flush=True)

    measured = [asdict(m) for m in metrics[args.warmup:]]

    def stats(key):
        values = [r[key] for r in measured if r[key] is not None]
        if not values:
            return None
        sorted_v = sorted(values)
        n = len(sorted_v)

        def pct(p):
            return sorted_v[max(0, min(n - 1, int(p * (n - 1))))]

        return {
            "n": n,
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if n > 1 else 0.0,
            "p50": pct(0.5),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "min": min(values),
            "max": max(values),
        }

    summary = {
        "config": {
            "model": args.model,
            "tp": args.tp,
            "image_size": [width, height],
            "lout": args.lout,
            "batch": args.batch,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "prompt_text": prompt_text,
        },
        "raw_warmup": [asdict(m) for m in metrics[:args.warmup]],
        "raw_measured": measured,
        "stats": {
            "e2e_ms": stats("e2e_ms"),
            "ttft_ms": stats("ttft_ms"),
            "itl_ms": stats("itl_ms"),
            "seq_in_len": stats("seq_in_len"),
            "seq_out_len": stats("seq_out_len"),
        },
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary["stats"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--image_size", type=int, default=672)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--lout", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1,
                        help="Concurrent identical requests per iteration")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_cudnn", action="store_true",
                        help="Disable cuDNN (workaround for old driver/cuDNN mismatch)")
    parser.add_argument("--disable_mm_preprocessor_cache",
                        action="store_true",
                        default=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
