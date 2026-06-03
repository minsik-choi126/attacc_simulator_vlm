"""Phase 3.6 — guidance_scale + flow_shift influence on latency.

CFG (classifier-free guidance) doubles the denoising compute (two
forwards per step).  Sweep guidance_scale to verify the cost model.

Also sweep flow_shift (UniPCMultistepScheduler param) -- default 10.0
per nvidia/Cosmos3-Nano model card -- to see whether step count or
scheduler curvature dominates latency.

Output: results/cosmos_guidance_sweep.json
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
    ap.add_argument("--framework", default="pytorch")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--resolution", default="480p")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--denoise-steps", type=int, default=20)
    ap.add_argument("--guidance-grid", nargs="+", type=float,
                    default=[1.0, 6.0, 12.0])
    ap.add_argument("--flow-shift-grid", nargs="+", type=float,
                    default=[3.0, 10.0])  # 10.0 = official default
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    host = detect_host()
    print(f"[Phase 3.6] guidance_sweep on host={host}")
    rows = []
    runner = HERE / args.framework / "runner.py"
    for g in args.guidance_grid:
        for fs in args.flow_shift_grid:
            for b in args.batches:
                cmd = [sys.executable, str(runner),
                       "--model", args.model, "--task", "t2v",
                       "--tp", str(args.tp), "--batch", str(b),
                       "--resolution", args.resolution,
                       "--frames", str(args.frames),
                       "--denoise-steps", str(args.denoise_steps),
                       "--guidance", str(g),
                       "--flow-shift", str(fs),
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
                payload.update({"guidance": g, "flow_shift": fs,
                                 "batch": b})
                rows.append(payload)
                print(f"  cfg={g}  flow_shift={fs}  b={b}  "
                      f"e2e_s={payload.get('e2e_s')}")
    save("cosmos_guidance_sweep",
          {"phase": "3.6", "host": host, "platform": host, **vars(args)},
          {"rows": rows})


if __name__ == "__main__":
    main()
