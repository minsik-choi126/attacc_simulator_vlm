"""R2 paper repro v3 — decode-only throughput (matches paper definition).

Paper Section 3.2 (line 378 of pdftotext extract):
    "throughput (generated tokens per second)"

i.e. Fig.14's throughput is decode-token throughput, *not* prefill+decode.
v2 used e2e_ms = prefill + decode, which is dominated by prefill at
batch=854 (94% of e2e), making the ratio collapse to ~1.1x.

Throughput definition used here:
    throughput = batch * Lout / decode_total_seconds
    decode_total_ms = g_time_ms * (Lout - 1)   # all generated tokens

Same paper Fig.14 setup:
    GPT-175B, Lin=2048, Lout=128, A100a x8, NVLink3, PIM bank
    DGX_Base batch = 54
    DGX+AttAcc batch = 854
    Targets: FP16 4.84x  INT8 3.47x   (DGX_Large skipped — path absent)
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save


MODEL = "GPT-175B"
LIN, LOUT = 2048, 128
B_DGX, B_ATT = 54, 854
TARGET_FP16 = 4.84
TARGET_INT8 = 3.47
TOLERANCE = 0.15


def go(system, batch, word):
    return sr.run(
        model=MODEL, system=system, gpu="A100a",
        ngpu=8, tp=8, num_attacc=8, num_hbm=5,
        interface="NVLINK3", pim="bank",
        batch=batch, lin=LIN, lout=LOUT, max_L=4096,
        word=word, powerlimit=True, ffopt=True, pipeopt=True,
    )


def decode_tput(metrics, batch):
    g = metrics["g_time"]
    decode_total_s = g * (LOUT - 1) / 1000.0
    return batch * LOUT / decode_total_s


def pair(word, label):
    print(f"\n=== {label} (word={word}) ===")
    print(f"  dgx        batch={B_DGX} ...", flush=True)
    rb = go("dgx", B_DGX, word)
    print(f"  dgx-attacc batch={B_ATT} ...", flush=True)
    ra = go("dgx-attacc", B_ATT, word)
    if rb is None or ra is None:
        return None

    tb = decode_tput(rb, B_DGX)
    ta = decode_tput(ra, B_ATT)
    ratio = ta / tb

    print(f"  dgx        g={rb['g_time']:>6.2f}ms/tok  "
          f"decode_tot={rb['g_time']*(LOUT-1):>8.1f}ms  "
          f"tput={tb:>7.1f} tok/s")
    print(f"  dgx-attacc g={ra['g_time']:>6.2f}ms/tok  "
          f"decode_tot={ra['g_time']*(LOUT-1):>8.1f}ms  "
          f"tput={ta:>7.1f} tok/s")
    print(f"  ratio = {ratio:.2f}x")

    return {
        "label": label, "word": word,
        "base":   {"batch": B_DGX, "g_time_ms": rb["g_time"],
                   "decode_total_ms": rb["g_time"] * (LOUT - 1),
                   "decode_throughput_tok_per_sec": tb,
                   "s_time_ms": rb["s_time"]},
        "attacc": {"batch": B_ATT, "g_time_ms": ra["g_time"],
                   "decode_total_ms": ra["g_time"] * (LOUT - 1),
                   "decode_throughput_tok_per_sec": ta,
                   "s_time_ms": ra["s_time"]},
        "ratio": ratio,
    }


def gate(label, ratio, target):
    if ratio is None:
        return f"  {label:22s}  ratio=—       target={target:.2f}x  SKIP"
    delta = (ratio - target) / target
    status = "PASS" if abs(delta) <= TOLERANCE else "FAIL"
    return (f"  {label:22s}  ratio={ratio:>5.2f}x  target={target:.2f}x  "
            f"Δ={delta*100:+.1f}%  {status}")


def main():
    print("R2 paper repro v3 — decode-only throughput")
    print(f"  model={MODEL}, Lin={LIN}, Lout={LOUT}")
    print(f"  batches: dgx={B_DGX}  dgx-attacc={B_ATT}")
    print(f"  paper targets: FP16 {TARGET_FP16}x  INT8 {TARGET_INT8}x")
    print(f"  throughput = batch * Lout / (g_time * (Lout-1))")

    fp16 = pair(2, "FP16 (W16A16)")
    int8 = pair(1, "INT8 (W8A8)")

    print("\n=== Gate summary ===")
    print(gate("FP16 vs DGX_Base", fp16["ratio"] if fp16 else None, TARGET_FP16))
    print(gate("INT8 vs DGX_Base", int8["ratio"] if int8 else None, TARGET_INT8))

    save("r2_paper_repro_v3",
         {"model": MODEL, "lin": LIN, "lout": LOUT,
          "batch_dgx": B_DGX, "batch_dgx_attacc": B_ATT,
          "targets": {"fp16_base": TARGET_FP16, "int8_base": TARGET_INT8},
          "method": "decode-only throughput (paper §3.2: generated tok/s)",
          "tolerance": TOLERANCE},
         {"fp16": fp16, "int8": int8})


if __name__ == "__main__":
    main()
