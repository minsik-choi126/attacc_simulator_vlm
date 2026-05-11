"""vLLM BF16 baseline aligned to multi_vlm_full_sim simulator matrix.

Runs the EXACT same (model, image_size, lin, lout, batch) configurations
as tier1_simulator/multi_vlm_full_sim.py to produce measured TTFT / ITL /
E2E. Lets paper Fig.3 overlay simulator-predicted vs measured speedup on
identical workload.

Models: Qwen2.5-VL-7B, LLaVA-1.5-7B, LLaVA-Next-Mistral-7B
        (driver 535 / vLLM 0.7.3 compatible)

Skipped here (vLLM 0.7.3 incompatible -- defer to driver 545+ node):
  - Qwen3-VL-4B
  - InternVL3-8B-hf

Output schema mirrors r6_vllm_measurement / w4a16_awq_measure so
downstream analysis can compare simulator vs measured side-by-side.
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


# Aligned to tier1_simulator/multi_vlm_full_sim.py VLM_CONFIGS + BATCHES.
# Qwen3-VL-4B and InternVL3-8B-hf are skipped because vLLM 0.7.3 does not
# load them (defer to driver 545+ stack).
ALIGNED_CONFIGS = [
    {"model": "Qwen/Qwen2.5-VL-7B-Instruct",
     "label": "Qwen2.5-VL-7B",          "image_size": 672, "lin": 704},
    {"model": "llava-hf/llava-1.5-7b-hf",
     "label": "LLaVA-1.5-7B",           "image_size": 336, "lin": 704},
    {"model": "llava-hf/llava-v1.6-mistral-7b-hf",
     "label": "LLaVA-Next-Mistral-7B",  "image_size": 672, "lin": 3008},
]
LOUT = 128
BATCHES = [1, 4, 8]


def make_dummy_image(size):
    try:
        from PIL import Image
        return Image.new("RGB", (size, size), color=(128, 128, 128))
    except ImportError:
        return None


def get_gpu_power_w():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"], timeout=2).decode().strip()
        return float(out.split("\n")[0])
    except Exception:
        return None


def measure_batch(llm, sp, model_id, prompt, image, batch_size):
    """Run a single batched request, return TTFT / E2E / ITL."""
    inputs = [make_image_input(model_id, prompt, image) for _ in range(batch_size)]
    outs = llm.generate(inputs, sp, use_tqdm=False)
    # Aggregate per-request metrics; report worst-case TTFT (slowest req).
    ttft_list, e2e_list, itl_list, seq_in_list, seq_out_list = [], [], [], [], []
    for out in outs:
        mt = out.metrics
        if mt is None or mt.first_token_time is None:
            continue
        ttft_ms = (mt.first_token_time - mt.arrival_time) * 1000.0
        e2e_ms = (mt.finished_time - mt.arrival_time) * 1000.0
        seq_in = len(out.prompt_token_ids) if out.prompt_token_ids else None
        seq_out = len(out.outputs[0].token_ids) if out.outputs else None
        itl_ms = ((e2e_ms - ttft_ms) / max(seq_out - 1, 1)
                    if seq_out and seq_out > 1 else None)
        ttft_list.append(ttft_ms)
        e2e_list.append(e2e_ms)
        if itl_ms is not None:
            itl_list.append(itl_ms)
        seq_in_list.append(seq_in)
        seq_out_list.append(seq_out)
    if not e2e_list:
        return None
    return {
        "ttft_ms_max": max(ttft_list) if ttft_list else None,
        "ttft_ms_mean": statistics.fmean(ttft_list) if ttft_list else None,
        "itl_ms_mean": statistics.fmean(itl_list) if itl_list else None,
        "e2e_ms_max": max(e2e_list),
        "e2e_ms_mean": statistics.fmean(e2e_list),
        "n_completed": len(e2e_list),
        "seq_in_first": seq_in_list[0],
        "seq_out_first": seq_out_list[0],
    }


def measure_one_model(cfg, lout, batches, repeats, warmup, prompt):
    if not HAVE_VLLM:
        return {"error": "vllm not installed"}
    img = make_dummy_image(cfg["image_size"])
    if img is None:
        return {"error": "PIL not installed"}

    print("    loading {} ...".format(cfg["model"]))
    t0 = time.time()
    llm = LLM(model=cfg["model"], tensor_parallel_size=1,
               trust_remote_code=True, max_model_len=8192,
               dtype="bfloat16", enforce_eager=True,
               disable_log_stats=True,
               limit_mm_per_prompt={"image": 1})
    load_s = time.time() - t0
    print("    loaded in {:.1f}s".format(load_s))

    sp = SamplingParams(temperature=0.0, max_tokens=lout,
                         min_tokens=lout, ignore_eos=True)

    per_batch = []
    for batch in batches:
        raw, powers = [], []
        for i in range(warmup + repeats):
            r = measure_batch(llm, sp, cfg["model"], prompt, img, batch)
            if r is None:
                continue
            r["iter"] = i
            r["is_warmup"] = i < warmup
            raw.append(r)
            p = get_gpu_power_w()
            if p is not None:
                powers.append(p)
            time.sleep(0.05)
        measured = [r for r in raw if not r["is_warmup"]]
        stats = {
            "ttft_ms_max":   summarize([r["ttft_ms_max"]   for r in measured if r["ttft_ms_max"]   is not None]),
            "ttft_ms_mean":  summarize([r["ttft_ms_mean"]  for r in measured if r["ttft_ms_mean"]  is not None]),
            "itl_ms_mean":   summarize([r["itl_ms_mean"]   for r in measured if r["itl_ms_mean"]   is not None]),
            "e2e_ms_max":    summarize([r["e2e_ms_max"]    for r in measured if r["e2e_ms_max"]    is not None]),
            "e2e_ms_mean":   summarize([r["e2e_ms_mean"]   for r in measured if r["e2e_ms_mean"]   is not None]),
            "power_w_avg":   statistics.fmean(powers) if powers else None,
        }
        per_batch.append({"batch": batch, "raw": raw, "stats": stats})
        ttft_p50 = (stats["ttft_ms_max"] or {}).get("p50") or 0
        itl_p50  = (stats["itl_ms_mean"] or {}).get("p50") or 0
        print("    batch={:>2d}  TTFT_max p50={:.1f}ms  ITL_mean p50={:.3f} ms/tok".format(
            batch, ttft_p50, itl_p50))
    return {"model": cfg["model"], "label": cfg["label"],
             "image_size": cfg["image_size"], "lin": cfg["lin"],
             "lout": lout, "batches": per_batch, "load_s": load_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                     help="HF paths to override ALIGNED_CONFIGS")
    ap.add_argument("--batches", nargs="+", type=int, default=BATCHES)
    ap.add_argument("--lout", type=int, default=LOUT)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prompt", default="Describe this image with 100 specific words.")
    args = ap.parse_args()

    if not HAVE_VLLM:
        print("FATAL: vllm not installed", file=sys.stderr)
        sys.exit(1)

    configs = ALIGNED_CONFIGS
    if args.models:
        configs = [c for c in ALIGNED_CONFIGS if c["model"] in args.models]

    print("vLLM BF16 baseline aligned to multi_vlm_full_sim -- H100 x 1")
    results = []
    for cfg in configs:
        print("  Model: {} (label {})".format(cfg["model"], cfg["label"]))
        try:
            r = measure_one_model(cfg, args.lout, args.batches,
                                    args.repeats, args.warmup, args.prompt)
            results.append(r)
        except Exception as exc:
            print("    FAILED: {}".format(exc))
            results.append({"model": cfg["model"], "label": cfg["label"],
                             "error": str(exc)})

    save("vllm_bf16_baseline_aligned",
         {"batches": args.batches, "lout": args.lout,
          "repeats": args.repeats, "warmup": args.warmup,
          "platform": "H100 x 1 vLLM 0.7.3 bf16",
          "note": "Skips Qwen3-VL / InternVL3 (vLLM 0.7.3 incompatible)"},
         {"per_model": results})
    print("Done -- pair with tier1_simulator/multi_vlm_full_sim.py for Fig.3 overlay")


if __name__ == "__main__":
    main()
