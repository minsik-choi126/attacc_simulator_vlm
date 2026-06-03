"""Phase 1.1 — E2E latency matrix sweep on Cosmos 3 (omni-modal).

Axes (per user 2026-06-04 confirm):
  model       : Cosmos3-Nano | Cosmos3-Super
  tp          : 1 | 2          (Super requires TP>=2)
  batch       : 1, 2, 4, 8     (real-time scenario coverage)
  framework   : vllm_omni | pytorch
  task        : t2v | t2i | i2v | v2v | t2a | multi2v | multi2action
                (omni input/output -- not just t2v)
  resolution  : 720p (default; video-output tasks only)
  frames      : 189 default    (video-output tasks only)

For each cell we measure wall-clock latency for a representative prompt,
with N_REPS warm + measured runs (default 3 measured).  Results land in
results/cosmos_e2e_latency.json.

Skips infeasible cells (Super TP=1, batch>1 if OOM, framework can't run
the task).
"""
import argparse
import gc
import json
import pathlib
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host, gpu_count, peak_bw_tbs
from result_aggregator import save, summarize
from cosmos_facts import (
    NANO, SUPER, ALL_MODELS, RESOLUTIONS, DEFAULT_FRAMES,
    DEFAULT_DENOISE_STEPS, DEFAULT_GUIDANCE_SCALE, TASKS, TASK_REQUIRES,
)


HOST = detect_host()
N_GPU = gpu_count()


def matrix_cells(args):
    """Yield (model, tp, batch, framework, task, resolution, frames) tuples.

    Cells where the task does not produce video have resolution/frames
    fields set to None where irrelevant (single-image / audio / action).
    """
    for model in args.models:
        for tp in args.tps:
            if model == "Cosmos3-Super" and tp < 2:
                continue
            if tp > N_GPU:
                continue
            for batch in args.batches:
                for framework in args.frameworks:
                    for task in args.tasks:
                        req = TASK_REQUIRES[task]
                        if "video" in req["outputs"]:
                            for resolution in args.resolutions:
                                yield (model, tp, batch, framework, task,
                                       resolution, args.frames)
                        elif "image" in req["outputs"]:
                            for resolution in args.resolutions:
                                yield (model, tp, batch, framework, task,
                                       resolution, 1)
                        else:
                            # audio / action: no res/frames axis
                            yield (model, tp, batch, framework, task,
                                   None, None)


def _maybe_subprocess(cmd, env=None, timeout=None):
    """Run the per-cell measurement in a fresh process so torch / vLLM state
    leaks (KV cache, CUDA context) cannot pollute the next cell."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                               env=env, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", "TIMEOUT after %s s" % timeout
    except Exception as e:
        return -1, "", f"subprocess error: {e}"


def run_cell(model, tp, batch, framework, task, resolution, frames,
             reps=3, timeout_s=900,
             denoise_steps=DEFAULT_DENOISE_STEPS,
             prompt=None):
    """Call the per-framework runner.

    Each cell is invoked as:
        python measurement/<framework>/runner.py --task T --model M ... --json
    The runner prints a single-line JSON with timings; we parse it back.
    """
    runner = HERE / framework / "runner.py"
    if not runner.exists():
        return {"status": "missing_runner",
                "error": f"runner not found at {runner}"}
    cmd = [
        sys.executable, str(runner),
        "--model", model,
        "--task", task,
        "--tp", str(tp),
        "--batch", str(batch),
        "--denoise-steps", str(denoise_steps),
        "--reps", str(reps),
        "--guidance", str(DEFAULT_GUIDANCE_SCALE),
        "--prompt", prompt or "a self-driving car turning right at sunset",
        "--json",
    ]
    if resolution is not None:
        cmd.extend(["--resolution", resolution])
    if frames is not None:
        cmd.extend(["--frames", str(frames)])
    rc, out, err = _maybe_subprocess(cmd, timeout=timeout_s)
    if rc != 0:
        return {"status": "runner_fail", "rc": rc,
                "stderr_tail": (err or "")[-800:],
                "stdout_tail": (out or "")[-800:]}
    # Parse the last JSON line
    payload = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
    if payload is None:
        return {"status": "no_json", "stdout_tail": (out or "")[-800:]}
    payload.setdefault("status", "ok")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+",
                    default=["Cosmos3-Nano", "Cosmos3-Super"],
                    choices=list(ALL_MODELS.keys()))
    ap.add_argument("--tps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--frameworks", nargs="+",
                    default=["pytorch", "vllm_omni"])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS),
                    choices=list(TASKS))
    ap.add_argument("--resolutions", nargs="+",
                    default=["720p"],
                    choices=list(RESOLUTIONS.keys()))
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout-s", type=int, default=900,
                    help="per-cell timeout (default 900 s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only enumerate cells, don't execute")
    args = ap.parse_args()

    print(f"[Phase 1.1] e2e_latency_matrix on host={HOST} ngpu={N_GPU}")
    print(f"  models={args.models} tps={args.tps} "
          f"batches={args.batches} frameworks={args.frameworks}")
    print(f"  tasks={args.tasks}")
    print(f"  resolutions={args.resolutions} frames={args.frames} "
          f"reps={args.reps}")

    cells = list(matrix_cells(args))
    print(f"  -> {len(cells)} cells\n")

    if args.dry_run:
        for c in cells:
            print(" ", c)
        return

    rows = []
    for i, (model, tp, batch, framework, task, resolution, frames) in (
            enumerate(cells, 1)):
        print(f"[{i:>3d}/{len(cells)}] {model:18s} tp={tp} b={batch:>2d} "
              f"fw={framework:9s} task={task:14s} "
              f"res={resolution!s:>5s} frames={frames!s:>4s}", end=" ")
        t0 = time.time()
        res = run_cell(model, tp, batch, framework, task, resolution,
                        frames, reps=args.reps, timeout_s=args.timeout_s)
        elapsed = time.time() - t0
        res.update({
            "model": model, "tp": tp, "batch": batch,
            "framework": framework, "task": task,
            "resolution": resolution, "frames": frames,
            "wall_s": round(elapsed, 1),
        })
        rows.append(res)
        if res.get("status") == "ok":
            e2e = res.get("e2e_s", res.get("p50_s"))
            e2e_s = f"{e2e:.2f}s" if isinstance(e2e, (int, float)) else "—"
            print(f"-> e2e={e2e_s}  (wall {elapsed:.0f}s)")
        else:
            print(f"-> [{res.get('status')}]")
        gc.collect()

    config = {
        "phase": "1.1", "host": HOST, "platform": HOST,
        "n_gpu": N_GPU, "peak_bw_tbs": peak_bw_tbs(HOST),
        "axes": {
            "models": args.models, "tps": args.tps,
            "batches": args.batches, "frameworks": args.frameworks,
            "tasks": args.tasks,
            "resolutions": args.resolutions, "frames": args.frames,
            "reps": args.reps,
        },
    }
    paths = save("cosmos_e2e_latency", config, {"rows": rows})
    print(f"\nSaved -> {[str(p) for p in paths]}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(ok_rows)}/{len(rows)} cells succeeded.")


if __name__ == "__main__":
    main()
