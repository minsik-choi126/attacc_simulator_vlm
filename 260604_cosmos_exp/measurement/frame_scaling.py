"""Phase 1.7 — frame count scaling sweep (independent of Phase 1.1 output).

Invokes runner.py directly per cell.  Does NOT clobber
cosmos_e2e_latency.json.  Own output: cosmos_frame_scaling.json.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host, gpu_count
from result_aggregator import save


HOST = detect_host()
N_GPU = gpu_count()


def _run(framework, model, task, tp, batch, resolution, frames, denoise_steps,
         reps, timeout_s):
    runner = HERE / framework / "runner.py"
    if not runner.exists():
        return {"status": "missing_runner"}
    cmd = [sys.executable, str(runner),
           "--model", model, "--task", task,
           "--tp", str(tp), "--batch", str(batch),
           "--resolution", resolution, "--frames", str(frames),
           "--denoise-steps", str(denoise_steps),
           "--reps", str(reps), "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    last = None
    for line in (proc.stdout or "").splitlines():
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                last = json.loads(line.strip())
            except Exception:
                continue
    if last is None:
        return {"status": "no_json", "rc": proc.returncode,
                "stderr_tail": (proc.stderr or "")[-300:]}
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Cosmos3-Nano", "Cosmos3-Super"])
    ap.add_argument("--tps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--frameworks", nargs="+",
                    default=["pytorch", "vllm_omni"])
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--frame-grid", nargs="+", type=int,
                    default=[24, 96, 189, 300])
    ap.add_argument("--denoise-steps", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout-s", type=int, default=900)
    args = ap.parse_args()

    print(f"[Phase 1.7] frame_scaling on host={HOST} ngpu={N_GPU}")
    rows = []
    for model in args.models:
        for tp in args.tps:
            if model == "Cosmos3-Super" and tp < 2:
                continue
            if tp > N_GPU:
                continue
            for batch in args.batches:
                for fw in args.frameworks:
                    for f in args.frame_grid:
                        t0 = time.time()
                        p = _run(fw, model, "t2v", tp, batch,
                                  args.resolution, f, args.denoise_steps,
                                  args.reps, args.timeout_s)
                        p.update({"model": model, "task": "t2v",
                                   "framework": fw, "tp": tp,
                                   "batch": batch,
                                   "resolution": args.resolution,
                                   "frames": f,
                                   "wall_s": round(time.time() - t0, 1)})
                        rows.append(p)
                        print(f"  {model:14s} fw={fw:9s} tp={tp} b={batch} "
                              f"frames={f:>4d}  -> {p.get('status')} "
                              f"e2e={p.get('e2e_s')}")
    save("cosmos_frame_scaling",
          {"phase": "1.7", "host": HOST, "platform": HOST, **vars(args)},
          {"rows": rows})


if __name__ == "__main__":
    main()
