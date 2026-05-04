"""R10 — Concurrent serving benchmark.

Submits N MMMU-Pro questions at a target arrival rate (Poisson) using vLLM's
async engine + AsyncLLMEngine.generate; measures per-request TTFT + total
latency under continuous batching, isolating queue wait from compute.

Output: per-request raw + p50/p95/p99 + throughput (req/s, tok/s).

Usage:
  python3 tests/r10_concurrent_serving.py \\
      --model Qwen/Qwen2.5-VL-7B-Instruct --tp 1 --num_requests 32 \\
      --rate 4 --lout 128 --output results/r10_qwen25_concurrent_4qps.json
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import List

from PIL import Image


def detect_template_text(model_name, prompt):
    name = model_name.lower()
    if "qwen3-vl" in name or "qwen2.5-vl" in name or "qwen2-vl" in name:
        return ("<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>"
                "{}<|im_end|>\n<|im_start|>assistant\n").format(prompt)
    if "llava-1.5" in name:
        return "USER: <image>\n{} ASSISTANT:".format(prompt)
    if "llava-v1.6" in name or "llava-next" in name:
        return "[INST] <image>\n{} [/INST]".format(prompt)
    return prompt


def format_mmmu_question(item):
    q = item.get("question") or item.get("Question") or ""
    options = item.get("options") or item.get("Options") or []
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = []
    parts = [q.strip()]
    if options:
        for idx, opt in enumerate(options):
            parts.append("{}. {}".format(chr(ord("A") + idx), str(opt).strip()))
        parts.append("Answer with only the letter.")
    return "\n".join(parts)


def load_mmmu_samples(num_samples, image_size_max):
    from datasets import load_dataset
    print("[R10] Loading MMMU/MMMU_Pro standard split...", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", "standard (4 options)", split="test",
                      streaming=False)
    items = []
    for row in ds:
        img = row.get("image")
        if img is None and "image_1" in row:
            img = row["image_1"]
        if img is None:
            continue
        if isinstance(img, dict) and "bytes" in img:
            from io import BytesIO
            img = Image.open(BytesIO(img["bytes"])).convert("RGB")
        elif not isinstance(img, Image.Image):
            try:
                img = img.convert("RGB")
            except Exception:
                continue
        if image_size_max:
            w, h = img.size
            scale = min(image_size_max / max(w, h), 1.0)
            if scale < 1.0:
                img = img.resize((max(1, int(w * scale)),
                                  max(1, int(h * scale))))
        prompt = format_mmmu_question(row)
        if not prompt.strip():
            continue
        items.append({
            "id": str(row.get("id") or row.get("question_id") or len(items)),
            "image": img,
            "prompt": prompt,
        })
        if len(items) >= num_samples:
            break
    return items


async def submit_request(engine, sampling_params, request_id, template_text,
                         image, t_arrival_global):
    from vllm import TextPrompt

    arrival = time.time() - t_arrival_global
    first_token_time = None
    last_token_time = None
    out_tokens = 0
    out_text = ""
    in_tokens = 0
    async for output in engine.generate(
        TextPrompt(prompt=template_text,
                   multi_modal_data={"image": image}),
        sampling_params,
        request_id=str(request_id),
    ):
        in_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else -1
        out_tokens = len(output.outputs[0].token_ids)
        out_text = output.outputs[0].text
        if first_token_time is None and out_tokens > 0:
            first_token_time = time.time()
        last_token_time = time.time()
    if first_token_time is None:
        first_token_time = last_token_time
    return {
        "request_id": request_id,
        "arrival": arrival,
        "ttft_ms": (first_token_time - (t_arrival_global + arrival)) * 1000,
        "completion_ms": (last_token_time - (t_arrival_global + arrival)) * 1000,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "itl_ms": ((last_token_time - first_token_time) * 1000 /
                   max(1, out_tokens - 1)),
    }


async def run_async(args):
    import torch
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True,
        disable_log_requests=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    samples = load_mmmu_samples(args.num_requests, args.image_size_max)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.lout,
        min_tokens=args.lout,
        ignore_eos=True,
    )

    # Warmup with one request to build cudagraphs
    print("[R10] warmup...", flush=True)
    await submit_request(engine, sampling_params, -1,
                         detect_template_text(args.model,
                                              samples[0]["prompt"]),
                         samples[0]["image"], time.time())
    print("[R10] warmup done", flush=True)

    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    t_start = time.time()
    tasks = []
    for idx, item in enumerate(samples):
        await asyncio.sleep(interval)  # Poisson approximated by uniform
        template_text = detect_template_text(args.model, item["prompt"])
        task = asyncio.create_task(
            submit_request(engine, sampling_params, idx, template_text,
                           item["image"], t_start))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    t_end = time.time()
    duration = t_end - t_start

    ttfts = [r["ttft_ms"] for r in results]
    itls = [r["itl_ms"] for r in results]
    completion = [r["completion_ms"] for r in results]
    total_out_tokens = sum(r["out_tokens"] for r in results)

    def stat(vals):
        s = sorted(vals); n = len(s)
        return {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if n > 1 else 0,
            "p50": s[n // 2],
            "p95": s[max(0, int(n * 0.95))],
            "p99": s[max(0, int(n * 0.99))],
            "min": min(vals), "max": max(vals),
        }

    summary = {
        "config": {
            "model": args.model, "tp": args.tp,
            "num_requests": args.num_requests,
            "target_rate_qps": args.rate,
            "lout": args.lout, "max_model_len": args.max_model_len,
        },
        "throughput": {
            "duration_s": duration,
            "actual_qps": args.num_requests / duration,
            "tok_per_sec": total_out_tokens / duration,
        },
        "stats": {
            "ttft_ms": stat(ttfts),
            "itl_ms": stat(itls),
            "completion_ms": stat(completion),
        },
        "raw": results,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    print(json.dumps(
        {"throughput": summary["throughput"], "stats": summary["stats"]},
        indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num_requests", type=int, default=32)
    parser.add_argument("--rate", type=float, default=4.0,
                        help="Arrival rate (req/s)")
    parser.add_argument("--lout", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--image_size_max", type=int, default=672)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_cudnn", action="store_true", default=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
