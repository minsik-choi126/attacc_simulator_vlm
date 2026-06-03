"""vLLM-Omni runner — invoked as subprocess by e2e_latency_matrix.py.

CLI (omni task aware):
    python runner.py --model Cosmos3-Nano --task t2v --tp 1 --batch 1 \
        --resolution 720p --frames 189 --denoise-steps 35 \
        --reps 3 --guidance 6.0 --prompt "..." --json

Supported tasks: t2v, t2i, i2v, v2v, t2a, multi2v, multi2action.
Tasks whose modalities the framework does not support print a
{"status": "task_unsupported"} payload (still rc 0 so the matrix
driver records the gap).

Prints exactly one JSON line on stdout with the timing payload, then exits
with rc 0 on success.
"""
import argparse
import json
import pathlib
import sys
import time
import traceback


HF_REPOS = {
    "Cosmos3-Nano":  "nvidia/Cosmos3-Nano",
    "Cosmos3-Super": "nvidia/Cosmos3-Super",
}


TASK_OUTPUT_MODALITIES = {
    "t2v":          ["video"],
    "t2i":          ["image"],
    "i2v":          ["video"],
    "v2v":          ["video"],
    "t2a":          ["audio"],
    "multi2v":      ["video"],
    "multi2action": ["action"],
}


def _stub_image(resolution):
    """Generate a dummy reference image for i2v / multi2v on-the-fly."""
    try:
        from PIL import Image
        import numpy as np
        sz = {"256p": (448, 256), "480p": (832, 480),
              "720p": (1280, 720)}[resolution or "720p"]
        return Image.fromarray((np.random.rand(sz[1], sz[0], 3) * 255)
                                 .astype("uint8"))
    except Exception:
        return None


def _stub_audio(seconds=2.0):
    try:
        import numpy as np
        return np.zeros(int(48000 * seconds), dtype="float32")
    except Exception:
        return None


def _stub_video(resolution, n_frames=8):
    try:
        from PIL import Image
        import numpy as np
        sz = {"256p": (448, 256), "480p": (832, 480),
              "720p": (1280, 720)}[resolution or "720p"]
        return [Image.fromarray(
            (np.random.rand(sz[1], sz[0], 3) * 255).astype("uint8"))
                 for _ in range(n_frames)]
    except Exception:
        return None


def measure(model, task, tp, batch, resolution, frames, denoise_steps, reps,
            guidance, prompt):
    """Run vLLM-Omni generation for the given omni task and return per-rep
    wall-clock seconds.  Returns None if the framework cannot serve the
    requested task."""
    from vllm_omni import OmniLLM      # type: ignore
    from vllm_omni.sampling import OmniSamplingParams  # type: ignore

    repo = HF_REPOS[model]
    print(f"# loading {repo} (tp={tp}) via vllm_omni", file=sys.stderr)
    llm = OmniLLM(
        model=repo,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        max_num_seqs=batch,
        enforce_eager=False,
    )

    output_modalities = TASK_OUTPUT_MODALITIES[task]
    sampling_kwargs = dict(
        modalities=output_modalities,
        denoise_steps=denoise_steps,
        guidance_scale=guidance,
        max_tokens=256,
    )
    if "video" in output_modalities:
        sampling_kwargs.update(video_resolution=resolution or "720p",
                                video_frames=frames or 189)
    elif "image" in output_modalities:
        sampling_kwargs.update(image_resolution=resolution or "720p")
    elif "audio" in output_modalities:
        sampling_kwargs.update(audio_seconds=4.0)
    elif "action" in output_modalities:
        sampling_kwargs.update(action_steps=64)

    sampling = OmniSamplingParams(**sampling_kwargs)

    extra_inputs = {}
    req = task.split("2")[0]
    if "i" in req or task == "multi2v" or task == "multi2action":
        img = _stub_image(resolution)
        if img is not None:
            extra_inputs["images"] = [img] * batch
    if task == "v2v" or task == "multi2action":
        vid = _stub_video(resolution)
        if vid is not None:
            extra_inputs["videos"] = [vid] * batch
    if task == "multi2v":
        au = _stub_audio()
        if au is not None:
            extra_inputs["audios"] = [au] * batch

    prompts = [prompt] * batch
    print(f"# warmup task={task}", file=sys.stderr)
    _ = llm.generate(prompts, sampling, **extra_inputs)

    times = []
    for r in range(reps):
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sampling, **extra_inputs)
        times.append(time.perf_counter() - t0)
        print(f"# rep {r} = {times[-1]:.3f}s", file=sys.stderr)

    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True,
                    choices=list(TASK_OUTPUT_MODALITIES))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--resolution", default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--denoise-steps", type=int, default=35)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--prompt", default="a self-driving car turning right")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        times = measure(args.model, args.task, args.tp, args.batch,
                         args.resolution, args.frames, args.denoise_steps,
                         args.reps, args.guidance, args.prompt)
    except ImportError as e:
        print(json.dumps({"status": "framework_missing",
                          "framework": "vllm_omni",
                          "task": args.task,
                          "error": str(e)[:300]}))
        sys.exit(0)
    except NotImplementedError as e:
        print(json.dumps({"status": "task_unsupported",
                          "framework": "vllm_omni",
                          "task": args.task,
                          "error": str(e)[:300]}))
        sys.exit(0)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"status": "runtime_error",
                          "framework": "vllm_omni",
                          "task": args.task,
                          "error": str(e)[:300]}))
        sys.exit(3)

    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2]
    payload = {
        "status": "ok",
        "framework": "vllm_omni",
        "task": args.task,
        "rep_seconds": times,
        "e2e_s": p50, "p50_s": p50,
        "min_s": min(times), "max_s": max(times),
        "mean_s": sum(times) / len(times),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
