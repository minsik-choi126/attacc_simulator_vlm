"""Phase 3.1 — Denoise step count sweep.

Verifies whether latency scales linearly with num_inference_steps (the
"each step costs ~same" assumption that gives Topic B its 35x leverage
ceiling).

Output: results/cosmos_denoise_step_sweep.json
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano")
    ap.add_argument("--task", default="t2v")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--framework", default="pytorch",
                    choices=["pytorch", "vllm_omni"])
    ap.add_argument("--resolution", default="480p")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--step-grid", nargs="+", type=int,
                    default=[10, 20, 35, 50, 100])
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    host = detect_host()
    print(f"[Phase 3.1] denoise_step_sweep on host={host}")
    rows = []
    runner = HERE / args.framework / "runner.py"
    for steps in args.step_grid:
        for b in args.batches:
            cmd = [sys.executable, str(runner),
                   "--model", args.model, "--task", args.task,
                   "--tp", str(args.tp), "--batch", str(b),
                   "--resolution", args.resolution,
                   "--frames", str(args.frames),
                   "--denoise-steps", str(steps),
                   "--reps", str(args.reps), "--json"]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=3600)
            last = None
            for line in (proc.stdout or "").splitlines():
                if line.strip().startswith("{"):
                    last = line.strip()
            try:
                payload = json.loads(last) if last else {}
            except Exception:
                payload = {"status": "no_json"}
            payload.update({"model": args.model, "task": args.task,
                             "tp": args.tp, "batch": b,
                             "denoise_steps": steps,
                             "resolution": args.resolution,
                             "frames": args.frames})
            rows.append(payload)
            print(f"  steps={steps:>3d} b={b:>2d}  e2e_s={payload.get('e2e_s')}")

    save("cosmos_denoise_step_sweep",
          {"phase": "3.1", "host": host, "platform": host, **vars(args)},
          {"rows": rows})


if __name__ == "__main__":
    main()
