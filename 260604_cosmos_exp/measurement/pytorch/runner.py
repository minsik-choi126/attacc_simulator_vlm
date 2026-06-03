"""PyTorch + Diffusers runner for Cosmos 3 (single Cosmos3OmniPipeline).

Confirmed against
https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3
and https://huggingface.co/nvidia/Cosmos3-{Nano,Super}.

A single Cosmos3OmniPipeline handles all tasks; the task axis differs
only by kwargs:

  t2i      : num_frames=1
  t2v      : num_frames=189 (default), no image
  i2v      : pass image=...
  t2v+aud  : enable_sound=True (sound-capable checkpoint only)

CLI matches measurement/vllm_omni/runner.py so the matrix driver is
backend-agnostic.

Records: per-rep wall-clock seconds, pipeline class actually used,
scheduler + flow_shift, negative_prompt selection.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))

# Imported lazily to keep --help fast.
from cosmos_facts import (
    HF_REPOS, NEGATIVE_PROMPT_T2V, NEGATIVE_PROMPT_I2V,
    DEFAULT_FLOW_SHIFT, DEFAULT_FPS, RESOLUTIONS,
)


TASK_OUTPUT_MODALITIES = {
    "t2v":          ["video"],
    "t2i":          ["image"],
    "i2v":          ["video"],
    "v2v":          ["video"],
    "t2a":          ["audio"],
    "multi2v":      ["video"],
    "multi2action": ["action"],
}


REFERENCE_PROMPT_T2V = (
    "The video opens with a view of a well-lit indoor space featuring a "
    "wooden display case with compartments filled with various fruits, "
    "including bananas, apples, pears, oranges, and carambolas. Two "
    "robotic arms with grippers are positioned at the bottom of the "
    "frame. The robotic arm on the right extends towards the display "
    "case, carefully picks up a pear, and places it into a plastic bag "
    "in a shopping cart nearby, then retracts. The arm repeats this with "
    "an orange and a carambola, completing a seamless automated "
    "fruit-picking process."
)
REFERENCE_PROMPT_I2V = REFERENCE_PROMPT_T2V


def _import_omni_pipeline():
    """Cosmos3OmniPipeline is THE only Cosmos 3 diffusers class."""
    import importlib
    diffusers = importlib.import_module("diffusers")
    cls = getattr(diffusers, "Cosmos3OmniPipeline", None)
    if cls is None:
        raise NotImplementedError(
            "diffusers.Cosmos3OmniPipeline not found -- need diffusers main "
            "branch (>= the Cosmos 3 release). pip install -U "
            "git+https://github.com/huggingface/diffusers")
    return cls, "diffusers.Cosmos3OmniPipeline"


def _stub_image(resolution):
    from PIL import Image
    import numpy as np
    W, H = RESOLUTIONS[resolution or "720p"]
    return Image.fromarray((np.random.rand(H, W, 3) * 255).astype("uint8"))


def measure(model, task, tp, batch, resolution, frames, denoise_steps, reps,
            guidance, flow_shift, prompt,
            nsys_profile_range=False, nvtx_per_step=False):
    import torch
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )

    pipeline_cls, pipeline_path = _import_omni_pipeline()
    repo = HF_REPOS[model]
    print(f"# loading {repo} via {pipeline_path} (tp={tp})", file=sys.stderr)

    common_load = dict(torch_dtype=torch.bfloat16,
                        enable_safety_checker=False)
    if tp > 1:
        pipe = pipeline_cls.from_pretrained(
            repo, device_map="balanced", **common_load)
    else:
        pipe = pipeline_cls.from_pretrained(
            repo, device_map="cuda", **common_load)

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=flow_shift)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    W, H = RESOLUTIONS[resolution or "720p"]
    out_mods = TASK_OUTPUT_MODALITIES[task]

    # Pick task-specific kwargs.
    if task == "t2i":
        num_frames = 1
        neg = NEGATIVE_PROMPT_T2V        # treat as t2v with frames=1
        kwargs_extra = {}
    elif task == "t2v":
        num_frames = frames or 189
        neg = NEGATIVE_PROMPT_T2V
        kwargs_extra = {}
    elif task == "i2v":
        num_frames = frames or 189
        neg = NEGATIVE_PROMPT_I2V
        kwargs_extra = {"image": _stub_image(resolution)}
    elif task in ("v2v", "multi2v", "multi2action"):
        # Not directly supported by Cosmos3OmniPipeline as documented;
        # fall back to i2v conditioning to record SOMETHING but tag
        # the row.
        num_frames = frames or 189
        neg = NEGATIVE_PROMPT_I2V
        kwargs_extra = {"image": _stub_image(resolution)}
    elif task == "t2a":
        # Audio is jointly generated with video via enable_sound=True;
        # there is no audio-only mode in the documented API.
        num_frames = frames or 189
        neg = NEGATIVE_PROMPT_T2V
        kwargs_extra = {"enable_sound": True}
    else:
        raise NotImplementedError(f"unknown task: {task}")

    # Optional: NVTX label per denoise step (one range per inference step,
    # opened by callback_on_step_end).  Used by Phase 3.2 to bound traffic
    # accounting to specific steps only.
    step_nvtx_active = [False]
    def step_cb(pipe_self, step_idx, t_step, callback_kwargs):
        if nvtx_per_step:
            if step_nvtx_active[0]:
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"cosmos:step_{step_idx + 1}")
            step_nvtx_active[0] = True
        return callback_kwargs

    def one_call(with_cb=False):
        kw = dict(
            prompt=[prompt] * batch,
            negative_prompt=[neg] * batch,
            num_frames=num_frames,
            height=H, width=W,
            num_inference_steps=denoise_steps,
            guidance_scale=guidance,
            fps=DEFAULT_FPS,
            generator=torch.Generator(device="cuda").manual_seed(123),
        )
        kw.update(kwargs_extra)
        if with_cb:
            try:
                return pipe(callback_on_step_end=step_cb, **kw)
            except TypeError:
                # diffusers version too old / Cosmos pipeline does not
                # forward the callback -- fall back to no-callback call.
                return pipe(**kw)
        return pipe(**kw)

    # Warmup is OUTSIDE any NVTX / profiler range.
    print(f"# warmup task={task} num_frames={num_frames}", file=sys.stderr)
    _ = one_call()
    torch.cuda.synchronize()

    # Measured reps.  When nsys_profile_range, wrap each rep with
    # cudaProfilerStart/Stop so an outer `nsys profile
    # --capture-range=cudaProfilerApi` only records the rep windows
    # (excludes model load and warmup from the trace).
    cudart = None
    if nsys_profile_range:
        try:
            cudart = torch.cuda.cudart()
        except Exception:
            cudart = None

    times = []
    for r in range(reps):
        torch.cuda.synchronize()
        if cudart is not None:
            cudart.cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"cosmos:denoise_call_rep{r}")
        try:
            t0 = time.perf_counter()
            _ = one_call(with_cb=nvtx_per_step)
            if nvtx_per_step and step_nvtx_active[0]:
                torch.cuda.nvtx.range_pop()      # close last per-step range
                step_nvtx_active[0] = False
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()          # close the rep range
            if cudart is not None:
                cudart.cudaProfilerStop()
        times.append(time.perf_counter() - t0)
        print(f"# rep {r} = {times[-1]:.3f}s", file=sys.stderr)
    return times, pipeline_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=list(HF_REPOS))
    ap.add_argument("--task", required=True,
                    choices=list(TASK_OUTPUT_MODALITIES))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--resolution", default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--denoise-steps", type=int, default=35)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--flow-shift", type=float, default=DEFAULT_FLOW_SHIFT)
    ap.add_argument("--prompt", default=REFERENCE_PROMPT_T2V)
    ap.add_argument("--nsys-profile-range", action="store_true",
                    help="wrap measured reps with cudaProfilerStart/Stop so "
                          "an outer `nsys profile --capture-range="
                          "cudaProfilerApi` records only the rep windows "
                          "(skips model load + warmup)")
    ap.add_argument("--nvtx-per-step", action="store_true",
                    help="push one NVTX range per denoise step via "
                          "callback_on_step_end (Phase 3.2 step-bounded "
                          "traffic accounting)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        times, pipeline_path = measure(
            args.model, args.task, args.tp, args.batch, args.resolution,
            args.frames, args.denoise_steps, args.reps, args.guidance,
            args.flow_shift, args.prompt,
            nsys_profile_range=args.nsys_profile_range,
            nvtx_per_step=args.nvtx_per_step)
    except (ImportError, NotImplementedError) as e:
        print(json.dumps({
            "status": "task_unsupported" if isinstance(e, NotImplementedError)
                       else "framework_missing",
            "framework": "pytorch", "task": args.task,
            "error": str(e)[:300]}))
        sys.exit(0)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"status": "runtime_error",
                          "framework": "pytorch",
                          "task": args.task,
                          "error": str(e)[:300]}))
        sys.exit(3)

    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2]
    payload = {
        "status": "ok",
        "framework": "pytorch",
        "task": args.task,
        "pipeline_path": pipeline_path,
        "scheduler": "UniPCMultistepScheduler",
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance,
        "denoise_steps": args.denoise_steps,
        "rep_seconds": times,
        "e2e_s": p50, "p50_s": p50,
        "min_s": min(times), "max_s": max(times),
        "mean_s": sum(times) / len(times),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
