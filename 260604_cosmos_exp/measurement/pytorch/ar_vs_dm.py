"""Phase 1.5 — AR-only vs DM-only timing decomposition.

Cosmos 3 is a MoT (mixture-of-transformers): one AR reasoner tower and
one DM (diffusion) generator tower share KV but have separate FFN.  We
isolate each by:

    1. AR-only (text reasoning only) — run the *reasoner* path; should
       expose Qwen3-VL backbone latency without any denoising loop.
    2. DM-only (image / video generation given a fixed text embedding) —
       run the *generator* with text embedding frozen; isolates the
       35-step denoising cost.
    3. AR+DM (default omni call) — full path.

Output: results/cosmos_ar_vs_dm.json
"""
import argparse
import json
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))

from hw_detect import detect_host
from result_aggregator import save


HF_REPOS = {
    "Cosmos3-Nano-Reasoner": "nvidia/Cosmos-Reason1-7B",
    "Cosmos3-Nano-Generator": "nvidia/Cosmos-Predict2",
    "Cosmos3-Nano":  "nvidia/Cosmos3-Nano",
}


def _ar_only(prompt, batch, reps):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_REPOS["Cosmos3-Nano-Reasoner"])
    mdl = AutoModelForCausalLM.from_pretrained(
        HF_REPOS["Cosmos3-Nano-Reasoner"],
        torch_dtype=torch.bfloat16).to("cuda")
    mdl.eval()
    ids = tok([prompt] * batch, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        for _ in range(2):
            _ = mdl.generate(ids, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = mdl.generate(ids, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    del mdl
    torch.cuda.empty_cache()
    return ts


def _dm_only(prompt, batch, reps, denoise_steps, resolution, frames):
    """Run the Cosmos generator alone given a frozen text embedding.

    Falls back to "task_unsupported" if diffusers does not expose a
    standalone generator pipeline.
    """
    import torch
    try:
        import importlib
        mod = importlib.import_module("diffusers")
        try:
            cls = getattr(mod, "CosmosVideoGeneratorPipeline")
        except AttributeError:
            cls = getattr(mod, "CosmosPredict2Pipeline")
    except Exception as e:
        return {"status": "task_unsupported", "error": str(e)}
    pipe = cls.from_pretrained(HF_REPOS["Cosmos3-Nano-Generator"],
                                torch_dtype=torch.bfloat16).to("cuda")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    W, H = {"256p": (448, 256), "480p": (832, 480),
            "720p": (1280, 720)}[resolution]
    gen = torch.Generator(device="cuda").manual_seed(42)
    text_emb = pipe.encode_prompt([prompt] * batch) \
        if hasattr(pipe, "encode_prompt") else None
    kwargs = dict(num_inference_steps=denoise_steps,
                   height=H, width=W, num_frames=frames, generator=gen)
    if text_emb is not None:
        kwargs["prompt_embeds"] = text_emb
    else:
        kwargs["prompt"] = [prompt] * batch
    # warmup + measure
    _ = pipe(**kwargs); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = pipe(**kwargs); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    del pipe
    torch.cuda.empty_cache()
    return ts


def _full_pipeline(prompt, batch, reps, denoise_steps, resolution, frames):
    """Use the e2e pytorch runner as subprocess (already validated)."""
    import subprocess
    runner = HERE / "runner.py"
    cmd = [sys.executable, str(runner),
           "--model", "Cosmos3-Nano", "--task", "t2v",
           "--tp", "1", "--batch", str(batch),
           "--resolution", resolution, "--frames", str(frames),
           "--denoise-steps", str(denoise_steps),
           "--reps", str(reps), "--prompt", prompt, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=3600)
    last_line = None
    for line in (proc.stdout or "").splitlines():
        if line.strip().startswith("{"):
            last_line = line.strip()
    return json.loads(last_line) if last_line else {"status": "no_json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a self-driving car turning right")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--frames", type=int, default=189)
    ap.add_argument("--denoise-steps", type=int, default=35)
    args = ap.parse_args()

    host = detect_host()
    print(f"[Phase 1.5] AR vs DM decomposition on host={host}")
    payload = {"args": vars(args)}

    print("  AR-only ...")
    try:
        payload["ar_only_seconds"] = _ar_only(args.prompt, args.batch,
                                               args.reps)
        print(f"    -> {payload['ar_only_seconds']}")
    except Exception as e:
        traceback.print_exc()
        payload["ar_only_error"] = str(e)

    print("  DM-only ...")
    try:
        payload["dm_only_seconds"] = _dm_only(
            args.prompt, args.batch, args.reps, args.denoise_steps,
            args.resolution, args.frames)
        print(f"    -> {payload['dm_only_seconds']}")
    except Exception as e:
        traceback.print_exc()
        payload["dm_only_error"] = str(e)

    print("  AR+DM (full) ...")
    payload["full_pipeline"] = _full_pipeline(
        args.prompt, args.batch, args.reps, args.denoise_steps,
        args.resolution, args.frames)
    print(f"    -> {payload['full_pipeline'].get('rep_seconds')}")

    config = {"phase": "1.5", "host": host, "platform": host, **vars(args)}
    save("cosmos_ar_vs_dm", config, payload)


if __name__ == "__main__":
    main()
