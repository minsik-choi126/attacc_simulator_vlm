"""R4 sensitivity sweep -- batch x L x chunk x pim_layers full grid.

Validates simulator monotonicity + identifies regime boundaries.

Default grid (Qwen3-VL-4B):
  batch  ∈ {1, 4, 8, 16, 32}
  L      ∈ {569, 1024, 2048}
  chunk  ∈ {16, 64, 256, 512}
  pim_layers (count) ∈ {0, 11, 22, 36}   # uses contiguous indices [0..count]

= 5 x 3 x 4 x 4 = 240 sim runs (manageable in ~15-30 min).
Output: long-form JSON for heatmap/plot in paper.
"""
import sys
import pathlib
import itertools

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

MODEL = "Qwen3-VL-4B"
IMG = 672
LOUT = 128

BATCHES = [1, 4, 8, 16, 32]
LS = [569, 1024, 2048]
CHUNKS = [16, 64, 256, 512]
PIM_LAYER_COUNTS = [0, 11, 22, 36]
NDEC = 36


def pim_layers_for_count(n):
    if n <= 0:
        return ""
    if n >= NDEC:
        return ",".join(str(i) for i in range(NDEC))
    # Evenly spaced
    step = NDEC / n
    return ",".join(str(int(i * step)) for i in range(n))


def main():
    total = len(BATCHES) * len(LS) * len(CHUNKS) * len(PIM_LAYER_COUNTS)
    print("R4 sensitivity sweep -- {} configs (Qwen3-VL-4B)".format(total))
    grid = []
    i = 0
    for batch, L, chunk, count in itertools.product(
            BATCHES, LS, CHUNKS, PIM_LAYER_COUNTS):
        i += 1
        layer_arg = pim_layers_for_count(count)
        m = sr.run(
            model=MODEL, system="dgx-attacc", gpu="A6000",
            ngpu=1, tp=1, num_attacc=1, num_hbm=5, interface="NVLINK_BRIDGE",
            pim="bank", lin=L, lout=LOUT, batch=batch,
            image_size=IMG,
            prefill_chunk=chunk, prefill_samples=8,
            max_L=max(2048, L + LOUT),
            powerlimit=True, ffopt=True, pipeopt=True, word=2,
            routing="list" if count > 0 else "conservative",
            pim_layers=layer_arg,
        )
        s = m.get("s_time") if m else None
        g = m.get("g_time") if m else None
        grid.append({"batch": batch, "L": L, "chunk": chunk,
                      "pim_layers_count": count,
                      "s_ms": s, "g_ms": g,
                      "total_ms": sr.e2e_ms(m, LOUT)})
        if i % 30 == 0 or i == total:
            print("  [{:>3d}/{:>3d}] batch={} L={} chunk={} pim_n={} -> "
                  "{:.2f}ms".format(i, total, batch, L, chunk, count,
                                       (s or 0) + (g or 0)))

    save("sensitivity_sweep",
         {"model": MODEL, "image_size": IMG, "lout": LOUT,
          "batches": BATCHES, "Ls": LS, "chunks": CHUNKS,
          "pim_layer_counts": PIM_LAYER_COUNTS,
          "platform": "A6000 x 1 A1 dgx-attacc"},
         {"grid": grid})
    print("\nDone -- {} configs in results/sensitivity_sweep.json".format(total))


if __name__ == "__main__":
    main()
