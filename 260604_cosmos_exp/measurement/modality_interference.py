"""Phase 4.3 — Mixed-modality interference (3 streams in-flight).

Submits 3 concurrent generation tasks of different output modalities
to vLLM-Omni and measures how each's P99 dilates compared with running
alone (single-stream P99 from Phase 4.2).

If vLLM-Omni is missing or only supports 1-shot generation, the script
records {"status": "concurrency_unsupported"} and Topic C path is
gated on R-CO3.

Output: results/cosmos_modality_interference.json
"""
import argparse
import json
import pathlib
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import save, load, summarize


STREAMS = [
    {"name": "video",  "task": "t2v", "prompt": "a sunrise over the ocean"},
    {"name": "image",  "task": "t2i", "prompt": "a single photo of a cat"},
    {"name": "audio",  "task": "t2a", "prompt": "soft piano music"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano")
    ap.add_argument("--samples-per-stream", type=int, default=8)
    ap.add_argument("--resolution", default="480p")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--denoise-steps", type=int, default=20)
    args = ap.parse_args()
    host = detect_host()
    print(f"[Phase 4.3] modality_interference on host={host}")

    try:
        from vllm_omni import OmniLLM            # type: ignore
        from vllm_omni.sampling import OmniSamplingParams  # type: ignore
    except ImportError as e:
        save("cosmos_modality_interference",
              {"phase": "4.3", "host": host, "platform": host, **vars(args)},
              {"status": "concurrency_unsupported",
               "error": str(e)})
        print("  vllm_omni missing -> recording status only")
        return

    try:
        llm = OmniLLM(model="nvidia/Cosmos3-Nano",
                       tensor_parallel_size=1,
                       dtype="bfloat16",
                       max_num_seqs=8)
    except Exception as e:
        save("cosmos_modality_interference",
              {"phase": "4.3", "host": host, "platform": host, **vars(args)},
              {"status": "load_fail", "error": str(e)})
        return

    per_stream_times = {s["name"]: [] for s in STREAMS}

    def worker(stream):
        sampling = OmniSamplingParams(
            modalities=[{"t2v": "video", "t2i": "image", "t2a": "audio"}
                         [stream["task"]]],
            denoise_steps=args.denoise_steps,
            video_resolution=args.resolution,
            video_frames=args.frames,
            max_tokens=128,
        )
        for _ in range(args.samples_per_stream):
            t0 = time.perf_counter()
            try:
                _ = llm.generate([stream["prompt"]], sampling)
                per_stream_times[stream["name"]].append(
                    time.perf_counter() - t0)
            except Exception:
                per_stream_times[stream["name"]].append(None)

    threads = [threading.Thread(target=worker, args=(s,)) for s in STREAMS]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3600)
    wall = time.perf_counter() - t_start

    # Compare with Phase 4.2 (single-stream) if present
    baseline = load("cosmos_modality_slo_budget") or {}
    baseline_by = {}
    for r in baseline.get("results", {}).get("rows", []):
        if r["tp"] == 1 and r["batch"] == 1 and r["status"] == "ok":
            baseline_by[r["modality"]] = r["summary"]

    rows = []
    for s in STREAMS:
        times = [t for t in per_stream_times[s["name"]] if t is not None]
        summ = summarize(times)
        base = baseline_by.get(s["name"], {})
        dilation = (None if not summ or not base.get("p99")
                    else summ.get("p99", 0) / base["p99"])
        rows.append({"stream": s["name"], "task": s["task"],
                      "concurrent_summary": summ,
                      "baseline_summary": base,
                      "p99_dilation_x": dilation})
        if summ:
            print(f"  {s['name']:7s} concurrent P99={summ.get('p99'):.2f}s "
                  f"baseline P99={base.get('p99'):.2f}s "
                  f"dilation={dilation}")

    save("cosmos_modality_interference",
          {"phase": "4.3", "host": host, "platform": host, **vars(args)},
          {"status": "ok", "rows": rows, "wall_s": wall})


if __name__ == "__main__":
    main()
