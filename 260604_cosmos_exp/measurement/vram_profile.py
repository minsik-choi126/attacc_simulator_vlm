"""Phase 1.2 — VRAM profile during Cosmos 3 generation.

Samples nvidia-smi memory.used at fixed interval while one generation
job runs in a background subprocess; reports peak / time-series.

Output: results/cosmos_vram_profile.json
"""
import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host, gpu_count
from result_aggregator import save


def _sample_vram(stop_event, samples, interval_s):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=3).decode().strip()
            ts = time.perf_counter()
            for line in out.splitlines():
                idx, used, total = [int(x.strip()) for x in line.split(",")]
                samples.append({"t_s": round(ts, 3), "gpu": idx,
                                "used_mib": used, "total_mib": total})
        except Exception:
            pass
        stop_event.wait(interval_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano")
    ap.add_argument("--task", default="t2v")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--framework", default="pytorch",
                    choices=["pytorch", "vllm_omni"])
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--frames", type=int, default=189)
    ap.add_argument("--interval-s", type=float, default=0.2)
    args = ap.parse_args()

    host = detect_host()
    print(f"[Phase 1.2] VRAM profile on host={host}")
    runner = HERE / args.framework / "runner.py"
    cmd = [sys.executable, str(runner),
           "--model", args.model, "--task", args.task,
           "--tp", str(args.tp), "--batch", str(args.batch),
           "--resolution", args.resolution, "--frames", str(args.frames),
           "--reps", "1", "--json"]

    samples = []
    stop = threading.Event()
    t = threading.Thread(target=_sample_vram,
                          args=(stop, samples, args.interval_s),
                          daemon=True)
    t.start()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=1800)
    elapsed = time.perf_counter() - t0
    stop.set()
    t.join(timeout=2)

    last_line = None
    for line in (proc.stdout or "").splitlines():
        if line.strip().startswith("{"):
            last_line = line.strip()
    runner_payload = json.loads(last_line) if last_line else {}

    by_gpu = {}
    for s in samples:
        by_gpu.setdefault(s["gpu"], []).append(s)
    peaks = {gpu: max(rec["used_mib"] for rec in recs)
             for gpu, recs in by_gpu.items()}
    print(f"  wall {elapsed:.1f}s  peaks(MiB) = {peaks}")

    config = {"phase": "1.2", "host": host, "platform": host,
              **vars(args)}
    paths = save("cosmos_vram_profile", config,
                  {"samples": samples, "peaks_mib": peaks,
                   "runner": runner_payload,
                   "wall_s": elapsed})
    print(f"Saved -> {[str(p) for p in paths]}")


if __name__ == "__main__":
    main()
