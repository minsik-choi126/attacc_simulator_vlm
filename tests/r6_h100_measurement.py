"""R6 — Real H100 measurement for Qwen3-VL-4B baseline.

Compares wall-clock prefill/decode latency against the simulator's R3 baseline
(GPU-only) predictions. Pass criterion: ±50% per plan §3 R6.

Usage:
    python3 tests/r6_h100_measurement.py --tp 1
    python3 tests/r6_h100_measurement.py --tp 2

Both runs use BF16, batch=1, 672x672 image, ~128 text tokens (so total
prefill length ~569 tokens for Qwen3-VL-4B).
"""

import argparse
import os
import time
import sys
import json

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def make_image(size=672):
    return Image.new("RGB", (size, size), color=(127, 127, 127))


def warmup_and_time(model, processor, image, prompt_text, lout, device, repeats):
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image",
            "image": image
        }, {
            "type": "text",
            "text": prompt_text
        }]
    }]
    try:
        text = processor.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        text = prompt_text
    inputs = processor(text=[text], images=[image], padding=True,
                       return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[-1]

    # Warmup
    with torch.inference_mode():
        for _ in range(2):
            _ = model.generate(**inputs,
                               max_new_tokens=4,
                               do_sample=False,
                               use_cache=True)
        torch.cuda.synchronize()

    prefill_times = []
    decode_times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        # Prefill = forward pass that fills KV
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        prefill_times.append((t1 - t0) * 1000.0)

        # Decode = lout-1 generation steps
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(**inputs,
                               max_new_tokens=lout - 1,
                               do_sample=False,
                               use_cache=True,
                               temperature=None,
                               top_p=None)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        decode_times.append((t1 - t0) * 1000.0)

    return seq_len, prefill_times, decode_times


def median(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--tp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--lout", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=672)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt",
                        default="Describe this image in detail. " * 8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.tp == 1:
        device_map = {"": 0}
        target_device = "cuda:0"
    else:
        device_map = "auto"
        target_device = "cuda:0"

    print("[R6] Loading {} (tp={})...".format(args.model, args.tp))
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    image = make_image(args.image_size)
    seq_len, prefill_ms, decode_ms = warmup_and_time(model, processor, image,
                                                     args.prompt, args.lout,
                                                     target_device,
                                                     args.repeats)

    prefill_med = median(prefill_ms)
    decode_total_med = median(decode_ms)
    decode_per_token = decode_total_med / max(1, args.lout - 1)
    e2e = prefill_med + decode_total_med

    result = {
        "tp": args.tp,
        "model": args.model,
        "image_size": args.image_size,
        "seq_len": int(seq_len),
        "lout": args.lout,
        "repeats": args.repeats,
        "prefill_ms_runs": prefill_ms,
        "decode_total_ms_runs": decode_ms,
        "prefill_ms_med": prefill_med,
        "decode_total_ms_med": decode_total_med,
        "decode_ms_per_token": decode_per_token,
        "e2e_ms": e2e,
    }
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
