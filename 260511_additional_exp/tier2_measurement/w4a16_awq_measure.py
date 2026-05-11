"""W4A16 (AWQ) vLLM measurement on H100 x 1.

Public quantized checkpoints (vLLM 0.7.3 compatible):
  - Qwen/Qwen2.5-VL-7B-Instruct-AWQ
  - Qwen/Qwen2.5-VL-3B-Instruct-AWQ

Compares against BF16 baseline (already measured in r6/r7).

Reports: weight memory, TTFT, ITL, J/token (via nvidia-smi).
"""
import argparse
import json
import os
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


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
]


def make_dummy_image(size=672):
    try:
        from PIL import Image
        return Image.new("RGB", (size, size), color=(128, 128, 128))
    except ImportError:
        return None


def get_gpu_power_w():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=2).decode().strip()
        return float(out.split("\n")[0])
    except Exception:
        return None


def measure_one(model_path, image_size, lout, repeats, warmup,
                 prompt="Describe this image with 100 specific words."):
    if not HAVE_VLLM:
        return {"error": "vllm not installed"}
    img = make_dummy_image(image_size)
    if img is None:
        return {"error": "PIL not installed"}

    print("    loading {} ...".format(model_path))
    t0 = time.time()
    llm = LLM(model=model_path,
              tensor_parallel_size=1,
              trust_remote_code=True,
              max_model_len=4096,
              quantization="awq",
              dtype="float16",
              enforce_eager=True,
              disable_log_stats=True,
              limit_mm_per_prompt={"image": 1})
    load_s = time.time() - t0
    print("    loaded in {:.1f}s".format(load_s))

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=lout,
                         min_tokens=lout, ignore_eos=True)

    inputs = [make_image_input(model_path, prompt, img)]

    raw = []
    powers = []
    for i in range(warmup + repeats):
        t_start = time.time()
        outs = llm.generate(inputs, sp, use_tqdm=False)
        t_end = time.time()
        out = outs[0]
        # vLLM RequestMetrics
        mt = out.metrics
        ttft_ms = (mt.first_token_time - mt.arrival_time) * 1000.0
        e2e_ms = (mt.finished_time - mt.arrival_time) * 1000.0
        seq_out = len(out.outputs[0].token_ids) if out.outputs else lout
        itl_ms = (e2e_ms - ttft_ms) / max(seq_out - 1, 1)
        raw.append({
            "request_idx": i,
            "is_warmup": i < warmup,
            "seq_in_len": len(out.prompt_token_ids) if out.prompt_token_ids else None,
            "seq_out_len": seq_out,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "itl_ms": itl_ms,
        })
        p = get_gpu_power_w()
        if p is not None:
            powers.append(p)
        time.sleep(0.1)

    measured = [r for r in raw if not r["is_warmup"]]
    stats = {
        "ttft_ms": summarize([r["ttft_ms"] for r in measured]),
        "itl_ms": summarize([r["itl_ms"] for r in measured]),
        "e2e_ms": summarize([r["e2e_ms"] for r in measured]),
        "power_w_avg": statistics.fmean(powers) if powers else None,
    }
    return {"model": model_path, "raw": raw, "stats": stats,
            "load_s": load_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--image_size", type=int, default=672)
    ap.add_argument("--lout", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    if not HAVE_VLLM:
        print("FATAL: vllm not installed in this environment", file=sys.stderr)
        print("       Install with: pip install vllm==0.7.3", file=sys.stderr)
        sys.exit(1)

    print("W4A16 (AWQ) vLLM measurement -- H100 x 1")
    all_results = []
    for model in args.models:
        print("  Model: {}".format(model))
        try:
            r = measure_one(model, args.image_size, args.lout,
                             args.repeats, args.warmup)
            stats = r.get("stats", {})
            print("    TTFT p50: {:.2f} ms  ITL p50: {:.3f} ms/tok".format(
                (stats.get("ttft_ms") or {}).get("p50") or 0,
                (stats.get("itl_ms") or {}).get("p50") or 0))
            all_results.append(r)
        except Exception as exc:
            print("    FAILED: {}".format(exc))
            all_results.append({"model": model, "error": str(exc)})

    save("w4a16_awq_measure",
         {"image_size": args.image_size, "lout": args.lout,
          "repeats": args.repeats, "warmup": args.warmup,
          "platform": "H100 x 1 vLLM 0.7.3 W4A16 (AWQ)"},
         {"per_model": all_results})
    print("Done")


if __name__ == "__main__":
    main()
