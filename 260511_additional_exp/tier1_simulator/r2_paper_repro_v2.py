"""R2 paper repro — Method A (hardcoded paper Fig.14 max batches).

Paper AttAcc ASPLOS'24 Fig.14 (GPT-3 175B, Lin=2048, Lout=128, no SLO):
  DGX_Base    max batch =  54   (640 GB HBM, capacity-bound)
  DGX_Large   max batch = 101   (1280 GB HBM, hypothetical)   ← path not yet modeled
  DGX+AttAcc  max batch = 854   (1280 GB on AttAcc side, KV offloaded)

Target speedup ratios:
  DGX+AttAcc vs DGX_Base   =  4.84×  (must-pass)
  DGX+AttAcc vs DGX_Large  =  2.48×  (should-pass, skipped — no DGX_Large)
  INT8 base                =  3.47×
  INT8 large               =  2.59×

Method A — hardcode batches, compute throughput-at-max-batch ratio:
  throughput = batch * lout / e2e_s
  ratio      = throughput(dgx-attacc) / throughput(dgx)

Caveats (see docs/r2_paper_repro_root_cause.md):
  1. Simulator's capacity model places KV on GPU side even for dgx-attacc, so
     batch=854 runs without OOM but is physically inconsistent with paper's
     AttAcc-side-KV assumption. Latency is still PIM-routed correctly.
  2. Per-system max batch (54 vs 854) is hardcoded from paper Fig.14, not
     back-solved from simulator capacity.
  3. DGX_Large path absent → INT8/FP16 large skipped.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save


MODEL = "GPT-175B"
LIN   = 2048
LOUT  = 128

# Paper Fig.14 max batches
BATCH_DGX_BASE  = 54
BATCH_DGX_ATTAC = 854

# Paper targets
TARGET_FP16_BASE  = 4.84
TARGET_FP16_LARGE = 2.48
TARGET_INT8_BASE  = 3.47
TARGET_INT8_LARGE = 2.59

# Tolerance for PASS (±15% — Method A is a rough match, not bit-precise)
TOLERANCE = 0.15


def throughput(metrics, batch, lout):
    """tokens per second."""
    e2e_s = sr.e2e_ms(metrics, lout) / 1000.0
    return batch * lout / e2e_s


def measure_pair(word, label):
    """Run dgx (batch=54) and dgx-attacc (batch=854) at given word size."""
    print(f"\n=== {label}  (word={word}) ===")
    print(f"  dgx        batch={BATCH_DGX_BASE} ...", flush=True)
    r_base = sr.run(
        model=MODEL, system="dgx", gpu="A100a",
        ngpu=8, tp=8, num_attacc=8, num_hbm=5,
        interface="NVLINK3",
        batch=BATCH_DGX_BASE, lin=LIN, lout=LOUT, max_L=4096,
        word=word, powerlimit=True, ffopt=True, pipeopt=True,
    )
    print(f"  dgx-attacc batch={BATCH_DGX_ATTAC} ...", flush=True)
    r_atta = sr.run(
        model=MODEL, system="dgx-attacc", gpu="A100a",
        ngpu=8, tp=8, num_attacc=8, num_hbm=5,
        interface="NVLINK3", pim="bank",
        batch=BATCH_DGX_ATTAC, lin=LIN, lout=LOUT, max_L=4096,
        word=word, powerlimit=True, ffopt=True, pipeopt=True,
    )
    if r_base is None or r_atta is None:
        print("  !! one or both runs failed")
        return None

    tput_base = throughput(r_base, BATCH_DGX_BASE,  LOUT)
    tput_atta = throughput(r_atta, BATCH_DGX_ATTAC, LOUT)
    ratio = tput_atta / tput_base

    print(f"  dgx        e2e {sr.e2e_ms(r_base, LOUT):>10.1f} ms   "
          f"tput {tput_base:>10.1f} tok/s")
    print(f"  dgx-attacc e2e {sr.e2e_ms(r_atta, LOUT):>10.1f} ms   "
          f"tput {tput_atta:>10.1f} tok/s")
    print(f"  ratio = {ratio:.2f}×")

    return {
        "label": label,
        "word": word,
        "base": {"batch": BATCH_DGX_BASE,
                  "e2e_ms": sr.e2e_ms(r_base, LOUT),
                  "throughput_tok_per_sec": tput_base},
        "attacc": {"batch": BATCH_DGX_ATTAC,
                    "e2e_ms": sr.e2e_ms(r_atta, LOUT),
                    "throughput_tok_per_sec": tput_atta},
        "ratio": ratio,
    }


def gate(ratio, target, label):
    if ratio is None:
        return f"  {label:25s}  ratio=—       target={target:.2f}×  SKIP"
    delta = abs(ratio - target) / target
    status = "PASS" if delta <= TOLERANCE else "FAIL"
    return (f"  {label:25s}  ratio={ratio:>5.2f}×  target={target:.2f}×  "
            f"Δ={delta*100:+.1f}%  {status}")


def main():
    print("R2 paper repro v2 — Method A (hardcoded max batches)")
    print(f"  model = {MODEL}, Lin={LIN}, Lout={LOUT}")
    print(f"  dgx batch={BATCH_DGX_BASE}  vs  dgx-attacc batch={BATCH_DGX_ATTAC}")
    print(f"  paper targets: FP16 {TARGET_FP16_BASE}×  INT8 {TARGET_INT8_BASE}×")

    fp16 = measure_pair(word=2, label="FP16 (W16A16)")
    int8 = measure_pair(word=1, label="INT8 (W8A8)")

    print("\n=== Gate summary ===")
    print(gate(fp16["ratio"] if fp16 else None, TARGET_FP16_BASE,  "FP16  vs DGX_Base"))
    print(gate(int8["ratio"] if int8 else None, TARGET_INT8_BASE,  "INT8  vs DGX_Base"))
    print(f"  {'FP16  vs DGX_Large':25s}  ratio=—       target={TARGET_FP16_LARGE:.2f}×  SKIP (no DGX_Large)")
    print(f"  {'INT8  vs DGX_Large':25s}  ratio=—       target={TARGET_INT8_LARGE:.2f}×  SKIP (no DGX_Large)")

    save("r2_paper_repro_v2",
         {"model": MODEL, "lin": LIN, "lout": LOUT,
          "batch_dgx": BATCH_DGX_BASE, "batch_dgx_attacc": BATCH_DGX_ATTAC,
          "targets": {"fp16_base": TARGET_FP16_BASE,
                       "fp16_large": TARGET_FP16_LARGE,
                       "int8_base": TARGET_INT8_BASE,
                       "int8_large": TARGET_INT8_LARGE},
          "method": "hardcoded paper Fig.14 max batches, throughput ratio",
          "tolerance": TOLERANCE},
         {"fp16": fp16, "int8": int8})


if __name__ == "__main__":
    main()
