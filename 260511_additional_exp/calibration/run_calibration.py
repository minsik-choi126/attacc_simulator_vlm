"""Unified calibration runner -- simulator + vLLM, one shot.

Detects the GPU (A6000 / H100 / A100), runs the simulator and vLLM with
identical (model, image_size, lin, lout, batch) configurations, and emits
a JSON with per-cell s_corr / g_corr / breakdown.

Usage on each HW:
    cd attacc_simulator
    python 260511_additional_exp/calibration/run_calibration.py --mode both
    # → results/calibration_<hw>.json

Modes:
    --mode sim    only simulator (no GPU needed beyond running this code)
    --mode vllm   only vLLM measurement (model loading + decode)
    --mode both   (default) both, side-by-side comparison

Filters:
    --models Qwen2.5-VL-7B,LLaVA-1.5-7B   only these labels
    --batches 1,4                          only these batch sizes

Cross-HW comparison is done separately by cross_hw_compare.py after
running this on each HW.
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

from configs import VLM_CONFIGS, BATCHES as DEFAULT_BATCHES, LOUT, HW_TO_SIM

try:
    from vllm import LLM, SamplingParams
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False


# ---------------------------------------------------------------------
# HW detection
# ---------------------------------------------------------------------
def detect_hw():
    """Return one of 'A6000', 'H100', 'A100', or 'unknown:<name>'."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=4).decode().strip()
        name = out.split("\n")[0].upper()
    except Exception:
        return "unknown:no_nvidia_smi"
    for tag in ("A6000", "H100", "A100"):
        if tag in name:
            return tag
    return f"unknown:{name}"


def get_gpu_power_w():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"], timeout=2).decode().strip()
        return float(out.split("\n")[0])
    except Exception:
        return None


# ---------------------------------------------------------------------
# Simulator side
# ---------------------------------------------------------------------
def run_simulator(sim_model, image_size, lin, batch, hw):
    sim_cfg = HW_TO_SIM.get(hw, HW_TO_SIM["A6000"])
    try:
        m = sr.run(
            model=sim_model, system="dgx",
            gpu=sim_cfg["gpu"], ngpu=1, tp=1,
            num_attacc=1, num_hbm=5,
            interface=sim_cfg["interface"], pim="bank",
            lin=lin, lout=LOUT, batch=batch,
            image_size=image_size,
            prefill_chunk=512, prefill_samples=8,
            max_L=4096,
            powerlimit=False, ffopt=True, pipeopt=False,
            word=2,
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}
    if m is None:
        return {"status": "no_output"}
    s_ms = m.get("s_time")
    g_ms = m.get("g_time")
    return {
        "status": "ok",
        "s_ms": s_ms,
        "g_ms_per_tok": g_ms,
        "e2e_ms": s_ms + g_ms * (LOUT - 1) if (s_ms and g_ms) else None,
        "required_cap_per_gpu": m.get("required_cap"),
        "s_flops": m.get("s_flops"),
        "g_flops": m.get("g_flops"),
    }


# ---------------------------------------------------------------------
# vLLM side
# ---------------------------------------------------------------------
def _dummy_image(size):
    try:
        from PIL import Image
        return Image.new("RGB", (size, size), color=(128, 128, 128))
    except ImportError:
        return None


def _make_prompt(label, lin_text_tokens=64):
    """Generate a text prompt approximately lin_text_tokens long."""
    return ("Describe the image in detail. " * max(1, lin_text_tokens // 6))[:lin_text_tokens * 6]


def _load_llm(hf_id, tp=1):
    """Try to load a vLLM model. Returns (LLM, SamplingParams) or (None, None)."""
    if not HAVE_VLLM:
        return None, None
    try:
        llm = LLM(model=hf_id, tensor_parallel_size=tp,
                   max_model_len=4096, gpu_memory_utilization=0.85,
                   trust_remote_code=True, enforce_eager=False)
    except Exception as e:
        print(f"  [vllm load failed] {hf_id}: {e}")
        return None, None
    sp = SamplingParams(temperature=0.0, max_tokens=LOUT, ignore_eos=True)
    return llm, sp


def _measure_vllm(llm, sp, hf_id, image_size, batch, repeats=3):
    """Run vLLM batch=batch with dummy image, return per-request statistics.

    Catches CUDA OOM and other RuntimeErrors to mark cell rather than crash
    the whole sweep -- higher batch sizes can fail on capacity-bound VLMs.
    """
    img = _dummy_image(image_size)
    prompt = _make_prompt(hf_id)
    if img is None:
        return {"status": "no_pil"}
    inputs = []
    for _ in range(batch):
        inputs.append({"prompt": prompt,
                       "multi_modal_data": {"image": img}})
    ttfts, e2es, itls = [], [], []
    powers = []
    for _ in range(repeats):
        try:
            p0 = get_gpu_power_w()
            outs = llm.generate(inputs, sp, use_tqdm=False)
            p1 = get_gpu_power_w()
        except (RuntimeError, ValueError) as e:
            msg = str(e)[:200]
            err = "oom" if ("out of memory" in msg.lower() or
                            "cuda" in msg.lower()) else "runtime_error"
            # Try to free residual state so the next config can proceed.
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return {"status": err, "error": msg}
        if p0 is not None and p1 is not None:
            powers.append((p0 + p1) / 2.0)
        for out in outs:
            mt = getattr(out, "metrics", None)
            if mt is None or mt.first_token_time is None:
                continue
            ttft_ms = (mt.first_token_time - mt.arrival_time) * 1000.0
            e2e_ms = (mt.finished_time - mt.arrival_time) * 1000.0
            seq_out = len(out.outputs[0].token_ids) if out.outputs else 0
            itl_ms = ((e2e_ms - ttft_ms) / max(seq_out - 1, 1)
                      if seq_out > 1 else None)
            ttfts.append(ttft_ms)
            e2es.append(e2e_ms)
            if itl_ms is not None:
                itls.append(itl_ms)
    if not ttfts:
        return {"status": "no_metrics"}
    return {
        "status": "ok",
        "ttft_ms_p50": statistics.median(ttfts),
        "ttft_ms_mean": statistics.fmean(ttfts),
        "ttft_ms_max": max(ttfts),
        "itl_ms_p50": statistics.median(itls) if itls else None,
        "itl_ms_mean": statistics.fmean(itls) if itls else None,
        "e2e_ms_p50": statistics.median(e2es),
        "power_w_mean": statistics.fmean(powers) if powers else None,
        "n_requests": len(ttfts),
    }


def run_vllm(hf_id, image_size, lin, batch, llm_cache):
    """Run vLLM measurement, caching the loaded model across configurations."""
    if not HAVE_VLLM:
        return {"status": "no_vllm"}
    if hf_id not in llm_cache:
        print(f"  [vllm load] {hf_id} ...", flush=True)
        t0 = time.time()
        llm, sp = _load_llm(hf_id)
        print(f"  [vllm load] {hf_id} done in {time.time()-t0:.1f}s",
              flush=True)
        llm_cache[hf_id] = (llm, sp)
    llm, sp = llm_cache[hf_id]
    if llm is None:
        return {"status": "load_failed"}
    return _measure_vllm(llm, sp, hf_id, image_size, batch)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def _select(arg_list, default):
    if arg_list is None:
        return default
    items = [s.strip() for s in arg_list.split(",") if s.strip()]
    return items or default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "vllm", "both"], default="both")
    ap.add_argument("--models", default=None,
                    help="comma-separated model labels to include")
    ap.add_argument("--batches", default=None,
                    help="comma-separated batch sizes (e.g. 1,4)")
    args = ap.parse_args()

    hw = detect_hw()
    print(f"[calibration] HW detected = {hw}")
    print(f"[calibration] mode = {args.mode}")

    batches = [int(b) for b in _select(args.batches,
                                         [str(b) for b in DEFAULT_BATCHES])]
    sel_models = _select(args.models, None)

    cells = []
    llm_cache = {}

    for cfg in VLM_CONFIGS:
        sim_model, hf_id, label, image_size, lin = cfg
        if sel_models and label not in sel_models:
            continue
        for batch in batches:
            print(f"\n[{label} img={image_size} lin={lin} batch={batch}]",
                  flush=True)
            entry = {
                "sim_model": sim_model, "hf_id": hf_id, "label": label,
                "image_size": image_size, "lin": lin, "batch": batch,
            }
            if args.mode in ("sim", "both"):
                t0 = time.time()
                entry["sim"] = run_simulator(sim_model, image_size, lin,
                                              batch, hw)
                entry["sim_walltime_s"] = time.time() - t0
                if entry["sim"].get("status") == "ok":
                    print(f"  sim    s={entry['sim']['s_ms']:>7.2f}ms  "
                          f"g={entry['sim']['g_ms_per_tok']:>6.3f}ms/tok  "
                          f"({entry['sim_walltime_s']:.1f}s)")
                else:
                    print(f"  sim    {entry['sim'].get('status')}")
            if args.mode in ("vllm", "both"):
                t0 = time.time()
                entry["vllm"] = run_vllm(hf_id, image_size, lin, batch,
                                          llm_cache)
                entry["vllm_walltime_s"] = time.time() - t0
                if entry["vllm"].get("status") == "ok":
                    print(f"  vllm   TTFT_p50={entry['vllm']['ttft_ms_p50']:>7.2f}ms  "
                          f"ITL_p50={entry['vllm']['itl_ms_p50']:>6.3f}ms/tok  "
                          f"({entry['vllm_walltime_s']:.1f}s)")
                else:
                    print(f"  vllm   {entry['vllm'].get('status')}")
            # Correction factors
            sim_ok = entry.get("sim", {}).get("status") == "ok"
            vllm_ok = entry.get("vllm", {}).get("status") == "ok"
            if sim_ok and vllm_ok:
                s_sim = entry["sim"]["s_ms"]
                g_sim = entry["sim"]["g_ms_per_tok"]
                s_meas = entry["vllm"]["ttft_ms_p50"]
                g_meas = entry["vllm"]["itl_ms_p50"]
                entry["s_corr"] = s_meas / s_sim if s_sim else None
                entry["g_corr"] = (g_meas / g_sim) if (g_sim and g_meas) else None
                if entry["s_corr"]:
                    print(f"  →  s_corr = {entry['s_corr']:.2f}x  "
                          f"g_corr = {entry['g_corr']:.2f}x"
                          if entry["g_corr"] else
                          f"  →  s_corr = {entry['s_corr']:.2f}x")
            cells.append(entry)

    # Summary
    print("\n=== Summary ===")
    print(f"{'Model':24s} {'img':>4s} {'b':>2s} "
          f"{'sim_s':>8s} {'meas_ttft':>10s} {'s_corr':>7s} "
          f"{'sim_g':>7s} {'meas_itl':>9s} {'g_corr':>7s}")
    for e in cells:
        if not (e.get("sim", {}).get("status") == "ok"
                and e.get("vllm", {}).get("status") == "ok"):
            continue
        print(f"{e['label']:24s} {e['image_size']:>4d} {e['batch']:>2d} "
              f"{e['sim']['s_ms']:>8.2f} {e['vllm']['ttft_ms_p50']:>10.2f} "
              f"{e.get('s_corr', 0):>6.2f}x "
              f"{e['sim']['g_ms_per_tok']:>7.3f} "
              f"{e['vllm']['itl_ms_p50']:>9.3f} "
              f"{e.get('g_corr', 0):>6.2f}x"
              if e.get("g_corr") else "")

    hw_tag = hw.lower().replace(":", "_").replace(" ", "_")
    save(f"calibration_{hw_tag}",
         {"hw": hw, "mode": args.mode, "lout": LOUT, "batches": batches,
          "models_filter": sel_models},
         {"cells": cells})
    print(f"\nSaved -> results/calibration_{hw_tag}.json")


if __name__ == "__main__":
    main()
