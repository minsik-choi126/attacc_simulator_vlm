"""Phase 1.3 — Phase-wise latency breakdown of Cosmos 3 generation.

Uses the official `Cosmos3OmniPipeline` from diffusers + UniPCMultistepScheduler
+ official negative_prompt so the timing reflects the same workload as
`measurement/pytorch/runner.py` (and thus the same workload NVIDIA
benchmarked).

Captures:
  model_load_s    : from_pretrained + device move + scheduler swap
  warmup_call_s   : first pipe() call (text encode + denoise + decode)
  measured_call_s : second pipe() call (steady state)
  step_times_s    : per-step latency via callback_on_step_end
  first_denoise_s, subseq_denoise_mean_s : derived from step_times_s
  text_encode_plus_decode_s : measured_call_s - sum(step_times_s)
                              (residual of text encode + VAE decode +
                              scheduler overhead)

Output: results/cosmos_phase_breakdown.json
"""
import argparse
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))

from hw_detect import detect_host
from result_aggregator import save
from cosmos_facts import (
    HF_REPOS, NEGATIVE_PROMPT_T2V, DEFAULT_FLOW_SHIFT, DEFAULT_FPS,
    RESOLUTIONS,
)

PROMPT = (
    "The video opens with a view of a well-lit indoor space featuring a "
    "wooden display case with compartments filled with various fruits. "
    "Two robotic arms with grippers are positioned at the bottom of the "
    "frame; the right arm extends toward the case, picks up a pear, and "
    "places it into a plastic bag in a shopping cart, then retracts."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano",
                    choices=list(HF_REPOS))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--resolution", default="720p",
                    choices=list(RESOLUTIONS))
    ap.add_argument("--frames", type=int, default=189)
    ap.add_argument("--denoise-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--flow-shift", type=float, default=DEFAULT_FLOW_SHIFT)
    args = ap.parse_args()

    host = detect_host()
    print(f"[Phase 1.3] phase_breakdown on host={host}")
    try:
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )

        repo = HF_REPOS[args.model]
        t0 = time.perf_counter()
        if args.tp > 1:
            pipe = Cosmos3OmniPipeline.from_pretrained(
                repo, torch_dtype=torch.bfloat16,
                device_map="balanced",
                enable_safety_checker=False)
        else:
            pipe = Cosmos3OmniPipeline.from_pretrained(
                repo, torch_dtype=torch.bfloat16,
                device_map="cuda",
                enable_safety_checker=False)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=args.flow_shift)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        t_model_load = time.perf_counter() - t0

        W, H = RESOLUTIONS[args.resolution]

        step_times = []
        last_t = [None]

        def cb(pipe_self, step_idx, t_step, callback_kwargs):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if last_t[0] is not None:
                step_times.append(now - last_t[0])
            last_t[0] = now
            return callback_kwargs

        def one_call():
            last_t[0] = time.perf_counter()
            kw = dict(
                prompt=[PROMPT] * args.batch,
                negative_prompt=[NEGATIVE_PROMPT_T2V] * args.batch,
                num_frames=args.frames,
                height=H, width=W,
                num_inference_steps=args.denoise_steps,
                guidance_scale=args.guidance,
                fps=DEFAULT_FPS,
                generator=torch.Generator(device="cuda").manual_seed(123),
            )
            try:
                return pipe(callback_on_step_end=cb, **kw)
            except TypeError:
                return pipe(**kw)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = one_call()
        torch.cuda.synchronize()
        t_warmup = time.perf_counter() - t0

        step_times.clear()
        last_t[0] = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = one_call()
        torch.cuda.synchronize()
        t_measured = time.perf_counter() - t0

        first = step_times[0] if step_times else None
        subseq_mean = (sum(step_times[1:]) / max(1, len(step_times) - 1)
                        if len(step_times) > 1 else None)
        text_encode_plus_decode = (t_measured - sum(step_times)
                                     if step_times else None)

        breakdown = {
            "model_load_s": t_model_load,
            "warmup_call_s": t_warmup,
            "measured_call_s": t_measured,
            "step_times_s": step_times,
            "first_denoise_s": first,
            "subseq_denoise_mean_s": subseq_mean,
            "text_encode_plus_decode_s": text_encode_plus_decode,
            "n_steps_captured": len(step_times),
            "pipeline_path": "diffusers.Cosmos3OmniPipeline",
            "scheduler": "UniPCMultistepScheduler",
            "flow_shift": args.flow_shift,
        }
        for k, v in breakdown.items():
            if isinstance(v, (int, float)):
                print(f"  {k:32s} = {v:.3f}")

        save("cosmos_phase_breakdown",
              {"phase": "1.3", "host": host, "platform": host,
               **vars(args)},
              breakdown)
    except Exception as e:
        traceback.print_exc()
        save("cosmos_phase_breakdown",
              {"phase": "1.3", "host": host, "platform": host,
               **vars(args)},
              {"status": "fail", "error": str(e)[:500]})


if __name__ == "__main__":
    main()
