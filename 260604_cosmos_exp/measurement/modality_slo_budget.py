"""Phase 4.2 — Per-modality latency budget (P50/P95/P99).

For each output modality {video, image, audio, action}, run N samples
of a representative single-modality generation under (model, TP, batch)
and report distribution.

Output: results/cosmos_modality_slo_budget.json
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import save, summarize


MODALITY_TASKS = {
    "video":  "t2v",
    "image":  "t2i",
    "audio":  "t2a",
    "action": "multi2action",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano")
    ap.add_argument("--framework", default="pytorch",
                    choices=["pytorch", "vllm_omni"])
    ap.add_argument("--tps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--resolution", default="480p")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--denoise-steps", type=int, default=20)
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    host = detect_host()
    runner = HERE / args.framework / "runner.py"
    rows = []
    for mod, task in MODALITY_TASKS.items():
        for tp in args.tps:
            for b in args.batches:
                cmd = [sys.executable, str(runner),
                       "--model", args.model, "--task", task,
                       "--tp", str(tp), "--batch", str(b),
                       "--resolution", args.resolution,
                       "--frames", str(args.frames),
                       "--denoise-steps", str(args.denoise_steps),
                       "--reps", str(args.samples), "--json"]
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=3600)
                last = None
                for line in (proc.stdout or "").splitlines():
                    if line.strip().startswith("{"):
                        last = line.strip()
                payload = (json.loads(last) if last
                            else {"status": "no_json"})
                summary = summarize(payload.get("rep_seconds", []))
                row = {"modality": mod, "task": task,
                       "tp": tp, "batch": b,
                       "status": payload.get("status"),
                       "summary": summary}
                rows.append(row)
                if summary:
                    print(f"  {mod:7s} tp={tp} b={b}  "
                          f"P50={summary.get('p50'):.2f}s  "
                          f"P95={summary.get('p95'):.2f}s  "
                          f"P99={summary.get('p99'):.2f}s")
                else:
                    print(f"  {mod:7s} tp={tp} b={b}  [{row['status']}]")
    save("cosmos_modality_slo_budget",
          {"phase": "4.2", "host": host, "platform": host, **vars(args)},
          {"rows": rows})


if __name__ == "__main__":
    main()
