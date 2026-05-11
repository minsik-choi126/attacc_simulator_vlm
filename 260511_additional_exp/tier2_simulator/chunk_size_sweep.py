"""Chunk size sweep -- find optimal prefill_chunk per VLM.

C ∈ {4, 16, 64, 128, 256, 512, 1024, lin (=full prefill)}.
Each VLM x each chunk -> s_time (prefill latency) measured.
Output: optimal C, latency curve.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

MODELS = [
    ("Qwen3-VL-4B",           672,  569),
    ("Qwen2.5-VL-7B",         672,  704),
    ("LLaVA-1.5-7B",          336,  704),
    ("LLaVA-Next-Mistral-7B", 672, 3008),
]
CHUNKS = [4, 16, 64, 128, 256, 512, 1024]


def main():
    print("Chunk size sweep -- H100 x 1 S1 dgx-attacc")
    all_results = []
    for model, image_size, lin in MODELS:
        full_chunk = max(CHUNKS + [lin])
        chunks = CHUNKS + [lin]   # last one = full prefill
        per_model = {"model": model, "image_size": image_size, "lin": lin,
                     "points": []}
        for c in chunks:
            m = sr.run(
                model=model, system="dgx-attacc", gpu="H100",
                ngpu=1, tp=1, num_attacc=1, num_hbm=5, interface="NVLINK4",
                pim="bank", lin=lin, lout=128, batch=1,
                image_size=image_size,
                prefill_chunk=c, prefill_samples=8, max_L=4096,
                powerlimit=True, ffopt=True, pipeopt=True, word=2,
            )
            s = m.get("s_time") if m else None
            g = m.get("g_time") if m else None
            tag = "full" if c == lin else "C={}".format(c)
            per_model["points"].append({
                "chunk": c, "tag": tag,
                "s_time_ms": s, "g_time_ms": g,
            })
            print("  {:25s}  C={:>5d}  s={:>7.2f}ms".format(
                model, c, s or -1))
        # Find optimum
        valid = [p for p in per_model["points"] if p["s_time_ms"]]
        if valid:
            opt = min(valid, key=lambda p: p["s_time_ms"])
            per_model["optimal"] = opt
            print("  -> optimal: {} ({:.2f}ms)\n".format(
                opt["tag"], opt["s_time_ms"]))
        all_results.append(per_model)

    save("chunk_size_sweep",
         {"chunks": CHUNKS, "platform": "H100 x 1 S1 dgx-attacc"},
         {"models": all_results})
    print("Done")


if __name__ == "__main__":
    main()
