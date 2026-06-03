"""Phase 4.1 — Streaming arrival feasibility check.

For Topic C (multi-modal Streaming SLO) to be a viable paper angle,
Cosmos 3 has to support *asynchronous* arrival of inputs across
modalities (text query interrupts video, audio chunks streamed in,
etc.) within at least one of vLLM-Omni / PyTorch.

This script tries:
    1. vLLM-Omni: open an OmniLLM, then submit 3 staggered requests
       (text-only @ t=0, image @ t=2s, audio @ t=4s) and check that
       they overlap in execution (not serialized).
    2. PyTorch: only single-shot generation; record as "unsupported".

If both fall back to 1-shot generation only, Topic C is dropped per
R-CO3 and we move on to Topic A/B only.

Output: results/cosmos_streaming_arrival.json
"""
import argparse
import json
import pathlib
import sys
import threading
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import save


def _try_vllm_streaming():
    try:
        from vllm_omni import OmniLLM        # type: ignore
        from vllm_omni.sampling import OmniSamplingParams  # type: ignore
    except ImportError as e:
        return {"status": "framework_missing", "error": str(e)}

    try:
        llm = OmniLLM(model="nvidia/Cosmos3-Nano",
                       tensor_parallel_size=1,
                       dtype="bfloat16",
                       max_num_seqs=4)
    except Exception as e:
        return {"status": "load_fail", "error": str(e)}

    timeline = []
    samp_text = OmniSamplingParams(modalities=["text"], max_tokens=128)
    samp_img = OmniSamplingParams(modalities=["image"], max_tokens=128)
    samp_aud = OmniSamplingParams(modalities=["audio"], max_tokens=128)

    def submit(name, delay, sampling, prompt):
        time.sleep(delay)
        t0 = time.perf_counter()
        try:
            _ = llm.generate([prompt], sampling)
            timeline.append({"name": name, "start_s": t0,
                             "end_s": time.perf_counter()})
        except Exception as e:
            timeline.append({"name": name, "error": str(e)})

    threads = [
        threading.Thread(target=submit, args=(
            "text", 0.0, samp_text, "describe the scene")),
        threading.Thread(target=submit, args=(
            "image", 2.0, samp_img, "render the next frame")),
        threading.Thread(target=submit, args=(
            "audio", 4.0, samp_aud, "say hello")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)

    overlapping = False
    if len(timeline) >= 2:
        ends = sorted([(t["start_s"], t["end_s"]) for t in timeline
                       if "error" not in t])
        for i in range(1, len(ends)):
            if ends[i][0] < ends[i - 1][1]:
                overlapping = True
                break
    return {"status": "tested", "timeline": timeline,
            "overlapping": overlapping}


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    host = detect_host()
    print(f"[Phase 4.1] streaming_arrival on host={host}")
    payload = {"vllm_omni": _try_vllm_streaming(),
               "pytorch": {"status": "unsupported",
                           "note": "PyTorch Diffusers pipelines are "
                                    "1-shot only by design."}}
    print(json.dumps(payload, indent=2, default=str))
    save("cosmos_streaming_arrival",
          {"phase": "4.1", "host": host, "platform": host},
          payload)


if __name__ == "__main__":
    main()
