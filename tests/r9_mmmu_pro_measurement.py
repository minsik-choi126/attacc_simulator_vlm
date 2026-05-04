"""R9 — Paper-grade VLM latency measurement on real MMMU-Pro data.

Replaces the dummy gray image + fixed prompt of r6_vllm_measurement.py with
actual MMMU-Pro questions (real images + multi-choice prompts). Reports
per-question TTFT/ITL/E2E plus aggregated p50/p95/p99 across the question set.

Adds:
  - GPU power sampling (nvidia-smi nvml) in parallel thread, computes energy.
  - Per-question seq_in/seq_out distribution + percentiles (so a paper table
    can show input length stats from a real workload).
  - Cold-start vs steady-state separation (warmup_count discarded).

Usage:
  python3 tests/r9_mmmu_pro_measurement.py \\
      --model Qwen/Qwen2.5-VL-7B-Instruct --tp 1 \\
      --num_samples 32 --warmup 2 --lout 128 \\
      --output results/r9_qwen25_mmmu_tp1.json
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

from PIL import Image


@dataclass
class RequestMetric:
    request_idx: int
    question_id: str
    seq_in_len: int
    seq_out_len: int
    e2e_ms: float
    ttft_ms: Optional[float]
    itl_ms: Optional[float]


class PowerSampler(threading.Thread):
    """Background poller of `nvidia-smi --query-gpu=power.draw` for energy.

    Records per-sample (timestamp, watts_per_gpu) into self.samples. Energy
    over an interval is integrated by `energy_joules(start, end)`.
    """

    def __init__(self, gpu_indices, interval_ms=50):
        super().__init__(daemon=True)
        self.gpu_indices = gpu_indices
        self.interval = interval_ms / 1000.0
        self.samples = []  # list of (t, [w0, w1, ...])
        self._stop_event = threading.Event()

    def run(self):
        index_arg = ",".join(str(i) for i in self.gpu_indices)
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--id={}".format(index_arg),
                     "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                t = time.time()
                lines = [
                    float(line.strip()) for line in out.stdout.strip().splitlines()
                    if line.strip()
                ]
                if lines:
                    self.samples.append((t, lines))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

    def energy_joules(self, t0, t1):
        joules = 0.0
        prev_t = None
        for t, ws in self.samples:
            if t < t0 or t > t1:
                continue
            total_w = sum(ws)
            if prev_t is not None:
                dt = t - prev_t
                joules += total_w * dt
            prev_t = t
        return joules


def detect_template_text(model_name, prompt):
    name = model_name.lower()
    if "qwen3-vl" in name or "qwen2.5-vl" in name or "qwen2-vl" in name:
        return ("<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>"
                "{}<|im_end|>\n<|im_start|>assistant\n").format(prompt)
    if "llava-1.5" in name or "llava-v1.5" in name:
        return "USER: <image>\n{} ASSISTANT:".format(prompt)
    if "llava-v1.6" in name or "llava-1.6" in name or "llava-next" in name:
        return ("[INST] <image>\n{} [/INST]").format(prompt)
    return prompt


def format_mmmu_question(item):
    """Construct a multi-choice VQA prompt from an MMMU-Pro item."""
    question = item.get("question") or item.get("Question") or ""
    options = item.get("options") or item.get("Options") or []
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = []
    parts = [question.strip()]
    if options:
        for idx, opt in enumerate(options):
            parts.append("{}. {}".format(chr(ord("A") + idx), str(opt).strip()))
        parts.append("Answer with only the letter of the correct option.")
    return "\n".join(parts)


def load_mmmu_samples(num_samples, image_size_max):
    from datasets import load_dataset
    print("[R9] Loading MMMU/MMMU_Pro standard split...", flush=True)
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


def median(xs):
    return sorted(xs)[len(xs) // 2]


def stats(values):
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


def run(args):
    import torch
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    samples = load_mmmu_samples(args.num_samples + args.warmup, args.image_size_max)
    print("[R9] Loaded {} MMMU-Pro samples".format(len(samples)), flush=True)

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
    print("[R9] LLM init kwargs: {}".format(init_kwargs), flush=True)
    llm = LLM(**init_kwargs)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.lout,
        min_tokens=args.lout,
        ignore_eos=True,
    )

    sampler = PowerSampler(list(range(args.tp)), interval_ms=50)
    sampler.start()

    metrics: List[RequestMetric] = []
    energy_total_j = 0.0
    duration_total_s = 0.0
    for idx, item in enumerate(samples):
        template_text = detect_template_text(args.model, item["prompt"])
        inputs = [{
            "prompt": template_text,
            "multi_modal_data": {"image": item["image"]},
        }]

        t0 = time.time()
        outputs = llm.generate(inputs, sampling, use_tqdm=False)
        t1 = time.time()

        out = outputs[0]
        seq_in_len = len(out.prompt_token_ids) if out.prompt_token_ids else -1
        seq_out_len = len(out.outputs[0].token_ids)

        rm = getattr(out, "metrics", None)
        ttft_ms = None
        itl_ms = None
        if rm is not None:
            arrival = getattr(rm, "arrival_time", None)
            first_token = getattr(rm, "first_token_time", None)
            finished = getattr(rm, "finished_time", None)
            if arrival is not None and first_token is not None:
                ttft_ms = (first_token - arrival) * 1000.0
            if (first_token is not None and finished is not None
                    and seq_out_len > 1):
                itl_ms = (finished - first_token) * 1000.0 / (seq_out_len - 1)

        e2e_ms = (t1 - t0) * 1000.0

        if idx >= args.warmup:
            energy_total_j += sampler.energy_joules(t0, t1)
            duration_total_s += (t1 - t0)

        m = RequestMetric(request_idx=idx,
                          question_id=item["id"],
                          seq_in_len=seq_in_len,
                          seq_out_len=seq_out_len,
                          e2e_ms=e2e_ms,
                          ttft_ms=ttft_ms,
                          itl_ms=itl_ms)
        metrics.append(m)
        marker = "warmup" if idx < args.warmup else "measure"
        print("[{}] idx={} qid={} in={} out={} e2e={:.2f}ms ttft={} itl={}".format(
            marker, idx, item["id"], seq_in_len, seq_out_len, e2e_ms,
            "{:.2f}ms".format(ttft_ms) if ttft_ms is not None else "n/a",
            "{:.3f}ms".format(itl_ms) if itl_ms is not None else "n/a"),
              flush=True)

    sampler.stop()
    sampler.join(timeout=2)

    measured = [asdict(m) for m in metrics[args.warmup:]]

    summary = {
        "config": {
            "model": args.model,
            "tp": args.tp,
            "lout": args.lout,
            "num_samples": args.num_samples,
            "warmup": args.warmup,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "image_size_max": args.image_size_max,
            "dataset": "MMMU/MMMU_Pro standard (4 options) split=test",
        },
        "raw_warmup": [asdict(m) for m in metrics[:args.warmup]],
        "raw_measured": measured,
        "stats": {
            "e2e_ms": stats([r["e2e_ms"] for r in measured]),
            "ttft_ms": stats([r["ttft_ms"] for r in measured if r["ttft_ms"] is not None]),
            "itl_ms": stats([r["itl_ms"] for r in measured if r["itl_ms"] is not None]),
            "seq_in_len": stats([r["seq_in_len"] for r in measured]),
            "seq_out_len": stats([r["seq_out_len"] for r in measured]),
        },
        "energy": {
            "total_joules": energy_total_j,
            "duration_s": duration_total_s,
            "average_watts": energy_total_j / duration_total_s if duration_total_s > 0 else 0,
            "joules_per_request": energy_total_j / max(1, len(measured)),
            "joules_per_token": (energy_total_j /
                                 max(1, sum(r["seq_out_len"] for r in measured))),
            "samples_collected": len(sampler.samples),
        },
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary["stats"], indent=2, default=str))
    print("\nEnergy:", json.dumps(summary["energy"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--lout", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--image_size_max", type=int, default=672,
                        help="Max image dimension; larger images get downscaled")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_cudnn", action="store_true", default=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
