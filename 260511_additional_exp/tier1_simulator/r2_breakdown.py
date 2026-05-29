"""Capture full breakdown of dgx / dgx-attacc at fp16 and int8 to analyze why
ratio is ~1.1x instead of paper's 4.84x / 3.47x. Tries multiple throughput
definitions matching Fig.14 (10000-request batched throughput, decode-only,
prefill-amortized)."""
import json
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr

MODEL = "GPT-175B"
LIN = 2048
LOUT = 128
B_DGX = 54
B_ATT = 854

def go(system, batch, word):
    return sr.run(
        model=MODEL, system=system, gpu="A100a",
        ngpu=8, tp=8, num_attacc=8, num_hbm=5,
        interface="NVLINK3", pim="bank",
        batch=batch, lin=LIN, lout=LOUT, max_L=4096,
        word=word, powerlimit=True, ffopt=True, pipeopt=True,
    )

def fmt(label, r):
    s = r.get("s_time")
    g = r.get("g_time")
    decode = g * (LOUT - 1)
    e2e = s + decode
    print(f"  {label}: s_time={s:>10.1f}ms  g_time={g:>7.2f}ms/tok  "
          f"decode_total={decode:>10.1f}ms  e2e={e2e:>10.1f}ms  "
          f"prefill_pct={100*s/e2e:.1f}%")
    return {"s_time_ms": s, "g_time_ms_per_tok": g,
            "decode_total_ms": decode, "e2e_ms": e2e,
            "qkv_time": r.get("g_qkv_time"),
            "prj_time": r.get("g_prj_time"),
            "ff_time": r.get("g_ff_time"),
            "softmax": r.get("g_softmax")}

def main():
    results = {}
    for word, dtype_label in [(2, "fp16"), (1, "int8")]:
        print(f"\n=== {dtype_label.upper()} (word={word}) ===")
        print(f"  running dgx batch={B_DGX} ...", flush=True)
        r_dgx = go("dgx", B_DGX, word)
        print(f"  running dgx-attacc batch={B_ATT} (cache hit expected) ...", flush=True)
        r_att = go("dgx-attacc", B_ATT, word)
        d_dgx = fmt(f"dgx        b={B_DGX:>3}", r_dgx)
        d_att = fmt(f"dgx-attacc b={B_ATT:>3}", r_att)
        results[dtype_label] = {"dgx": d_dgx, "attacc": d_att}

    print("\n=== Throughput definitions and ratios ===")
    for dt in ("fp16", "int8"):
        d = results[dt]["dgx"]
        a = results[dt]["attacc"]
        # def 1: e2e per single batch (what current script does)
        tput_dgx_1 = B_DGX * LOUT * 1000 / d["e2e_ms"]
        tput_att_1 = B_ATT * LOUT * 1000 / a["e2e_ms"]
        # def 2: decode-only
        tput_dgx_2 = B_DGX * LOUT * 1000 / d["decode_total_ms"]
        tput_att_2 = B_ATT * LOUT * 1000 / a["decode_total_ms"]
        # def 3: 10000 requests (prefill+decode * num_batches)
        # this is mathematically equal to def 1, but let's spell it out
        T_dgx = (10000 / B_DGX) * d["e2e_ms"]
        T_att = (10000 / B_ATT) * a["e2e_ms"]
        tput_dgx_3 = 10000 * LOUT * 1000 / T_dgx
        tput_att_3 = 10000 * LOUT * 1000 / T_att

        print(f"\n  {dt.upper()}:")
        print(f"    [1] e2e(prefill+decode) ratio    "
              f"= {tput_att_1/tput_dgx_1:.2f}x  (curr method)")
        print(f"    [2] decode-only ratio            "
              f"= {tput_att_2/tput_dgx_2:.2f}x")
        print(f"    [3] 10000-req e2e ratio (=[1])   "
              f"= {tput_att_3/tput_dgx_3:.2f}x")

    out = HERE.parent / "results" / "r2_breakdown.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  saved -> {out}")

if __name__ == "__main__":
    main()
