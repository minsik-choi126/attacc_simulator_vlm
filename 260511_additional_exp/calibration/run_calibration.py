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
from vllm_helpers import make_image_input, detect_template_text

from configs import VLM_CONFIGS, BATCHES as DEFAULT_BATCHES, LOUT, HW_TO_SIM

try:
    from transformers import AutoTokenizer
    HAVE_TOKENIZER = True
except ImportError:
    HAVE_TOKENIZER = False

# Cache one tokenizer per HF repo so iterative prompt sizing doesn't
# re-load the tokenizer for every cell.
_TOKENIZER_CACHE = {}

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


def _approx_visual_tokens(sim_model, image_size):
    """Approximate visual token count fed into the LLM decoder.

    Uses Transformer.compute_visual_tokens (the post-projector count),
    falling back to 0 on failure so the text length defaults to lin.
    """
    try:
        from src.config import make_model_config
        from src.model import Transformer
        from src.type import DataType
        cfg = make_model_config(sim_model, DataType.W16A16)
        t = Transformer(cfg, tensor_parallel=1)
        return int(t.compute_visual_tokens(image_size))
    except Exception:
        return 0


def _get_tokenizer(hf_id):
    if not HAVE_TOKENIZER:
        return None
    if hf_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[hf_id]
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:
        print(f"  [tokenizer] {hf_id} load failed: {e}")
        tok = None
    _TOKENIZER_CACHE[hf_id] = tok
    return tok


def _count_template_tokens(tok, hf_id, text):
    """Token count of (chat-template + image placeholder + user text).

    vLLM later expands the image placeholder token into `visual_tokens`
    worth of multimodal embeddings, so the LLM decoder ultimately sees
    `count - 1 + visual_tokens` tokens (1 placeholder slot displaced).
    """
    full = detect_template_text(hf_id, text)
    return len(tok.encode(full, add_special_tokens=True))


def _make_prompt_for_lin(target_lin, visual_tokens, hf_id=None):
    """Build a prompt sized so that the LLM decoder sees ~target_lin tokens.

    If a HuggingFace tokenizer is available for `hf_id`, iteratively grow
    or trim the prompt until tokenize(template + text) + visual_tokens - 1
    is within 2 tokens of target_lin.  Otherwise fall back to a 6-char /
    token heuristic.  The fallback was tolerable for OOM avoidance only;
    paper-grade "Lin = X" claims require the tokenizer path.
    """
    text_tokens_target = max(8, target_lin - max(0, visual_tokens))
    base_phrase = "Describe the image in detail. "

    tok = _get_tokenizer(hf_id) if hf_id else None
    if tok is None:
        # Heuristic fallback: ~6 chars per English token.
        return (base_phrase * max(1, text_tokens_target // 6))[
            : text_tokens_target * 6]

    # template_tokens after substitution: target_lin == template_tokens - 1 + visual_tokens
    template_target = max(4, target_lin + 1 - max(0, visual_tokens))
    n_phrases = max(1, template_target // 5)
    text = base_phrase * n_phrases
    last_count = None
    for _ in range(15):
        n = _count_template_tokens(tok, hf_id, text)
        last_count = n
        delta = template_target - n
        if abs(delta) <= 2:
            break
        if delta > 0:
            text += base_phrase * max(1, delta // 5)
        else:
            words = text.split()
            cut = max(4, len(words) + delta)
            if cut >= len(words):
                break
            text = " ".join(words[:cut])
    return text


def _actual_lin_from_prompt_ids(prompt_token_ids):
    """vLLM's reported prompt_token_ids length is the *post-multimodal*
    decoder input length (includes expanded image tokens).  This is the
    number that should equal the simulator's `lin`."""
    if not prompt_token_ids:
        return None
    return len(prompt_token_ids)


def _unload_llm():
    """Drop any in-memory vLLM model + free CUDA cache so the next model
    can be loaded without OOM."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


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


def _measure_vllm(llm, sp, hf_id, sim_model, image_size, lin, batch, repeats=3):
    """Run vLLM batch=batch with dummy image, return per-request statistics.

    Catches CUDA OOM and other RuntimeErrors to mark cell rather than crash
    the whole sweep -- higher batch sizes can fail on capacity-bound VLMs.
    Prompt length is sized to match the simulator's `lin` (total prefill
    sequence length = visual_tokens + text_tokens).
    """
    img = _dummy_image(image_size)
    if img is None:
        return {"status": "no_pil"}
    vis_tok = _approx_visual_tokens(sim_model, image_size)
    # R9 guard: lin must leave room for the image placeholder + at least
    # one text token, otherwise the calibration cell is meaningless.
    if vis_tok and lin <= vis_tok:
        return {"status": "lin_below_visual_tokens",
                "visual_tokens": vis_tok, "lin": lin}
    # R15: tokenizer-based prompt sizing (falls back to char heuristic).
    prompt = _make_prompt_for_lin(lin, vis_tok, hf_id=hf_id)
    # R8: use per-model chat template with image placeholder so vLLM
    # routes multi_modal_data into the prompt correctly.  Without this,
    # some models (Qwen-VL, InternVL) silently drop the image.
    inputs = [make_image_input(hf_id, prompt, img) for _ in range(batch)]
    ttfts, e2es, itls = [], [], []
    actual_lin_tokens = []  # R15: track what vLLM actually saw
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
            # R15: actual decoder input length from vLLM (includes
            # expanded image placeholder tokens).
            n = _actual_lin_from_prompt_ids(
                getattr(out, "prompt_token_ids", None))
            if n is not None:
                actual_lin_tokens.append(n)
    if not ttfts:
        return {"status": "no_metrics"}
    # R15: report actual decoder input length so callers can audit
    # how close to target_lin the prompt actually landed.
    actual_lin_median = (statistics.median(actual_lin_tokens)
                          if actual_lin_tokens else None)
    actual_lin_delta = ((actual_lin_median - lin)
                         if actual_lin_median is not None else None)
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
        "actual_lin_tokens_p50": actual_lin_median,
        "actual_lin_delta_vs_target": actual_lin_delta,
        "visual_tokens_estimated": vis_tok,
    }


def run_vllm(llm, sp, hf_id, sim_model, image_size, lin, batch):
    """Run a single vLLM measurement against an already-loaded model."""
    if not HAVE_VLLM:
        return {"status": "no_vllm"}
    if llm is None:
        return {"status": "load_failed"}
    return _measure_vllm(llm, sp, hf_id, sim_model, image_size, lin, batch)


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

    # Group configs by hf_id so each vLLM model is loaded once, then unloaded
    # before the next model -- prevents 5 VLMs accumulating in VRAM.
    by_hf = {}
    for cfg in VLM_CONFIGS:
        sim_model, hf_id, label, image_size, lin = cfg
        if sel_models and label not in sel_models:
            continue
        by_hf.setdefault(hf_id, []).append(cfg)

    for hf_id, cfg_list in by_hf.items():
        llm = sp = None
        if args.mode in ("vllm", "both") and HAVE_VLLM:
            print(f"\n[vllm load] {hf_id} ...", flush=True)
            t0 = time.time()
            llm, sp = _load_llm(hf_id)
            print(f"[vllm load] {hf_id} done in {time.time()-t0:.1f}s",
                  flush=True)

        try:
            for cfg in cfg_list:
                sim_model, _, label, image_size, lin = cfg
                for batch in batches:
                    print(f"\n[{label} img={image_size} lin={lin} batch={batch}]",
                          flush=True)
                    entry = {
                        "sim_model": sim_model, "hf_id": hf_id,
                        "label": label, "image_size": image_size,
                        "lin": lin, "batch": batch,
                    }
                    if args.mode in ("sim", "both"):
                        t0 = time.time()
                        entry["sim"] = run_simulator(sim_model, image_size,
                                                      lin, batch, hw)
                        entry["sim_walltime_s"] = time.time() - t0
                        if entry["sim"].get("status") == "ok":
                            print(f"  sim    s={entry['sim']['s_ms']:>7.2f}ms  "
                                  f"g={entry['sim']['g_ms_per_tok']:>6.3f}ms/tok  "
                                  f"({entry['sim_walltime_s']:.1f}s)")
                        else:
                            print(f"  sim    {entry['sim'].get('status')}")
                    if args.mode in ("vllm", "both"):
                        t0 = time.time()
                        entry["vllm"] = run_vllm(llm, sp, hf_id, sim_model,
                                                  image_size, lin, batch)
                        entry["vllm_walltime_s"] = time.time() - t0
                        if entry["vllm"].get("status") == "ok":
                            print(f"  vllm   TTFT_p50={entry['vllm']['ttft_ms_p50']:>7.2f}ms  "
                                  f"ITL_p50={entry['vllm']['itl_ms_p50']:>6.3f}ms/tok  "
                                  f"({entry['vllm_walltime_s']:.1f}s)")
                        else:
                            print(f"  vllm   {entry['vllm'].get('status')}")
                    # Correction factors.
                    sim_ok = entry.get("sim", {}).get("status") == "ok"
                    vllm_ok = entry.get("vllm", {}).get("status") == "ok"
                    if sim_ok and vllm_ok:
                        s_sim = entry["sim"]["s_ms"]
                        g_sim = entry["sim"]["g_ms_per_tok"]
                        s_meas = entry["vllm"]["ttft_ms_p50"]
                        g_meas = entry["vllm"]["itl_ms_p50"]
                        entry["s_corr"] = s_meas / s_sim if s_sim else None
                        entry["g_corr"] = ((g_meas / g_sim)
                                            if (g_sim and g_meas) else None)
                        if entry["s_corr"]:
                            if entry.get("g_corr"):
                                print(f"  ->  s_corr = {entry['s_corr']:.2f}x  "
                                      f"g_corr = {entry['g_corr']:.2f}x")
                            else:
                                print(f"  ->  s_corr = {entry['s_corr']:.2f}x")
                    cells.append(entry)
        finally:
            # Unload before moving to the next HF model.
            if llm is not None:
                try:
                    del llm
                except Exception:
                    pass
            llm = sp = None
            _unload_llm()

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
