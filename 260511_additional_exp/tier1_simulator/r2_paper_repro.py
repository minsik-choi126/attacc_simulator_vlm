"""R2 -- AttAcc paper sec.7.2 reproduction.

Simulated DGX-A100 x8 + AttAcc 8. GPT-175B at L=2048, batch=64.
Targets:
  DGXxAttAccs vs DGX_Base (FP16):  4.84x  +/-20%  (must-pass)
  DGXxAttAccs vs DGX_Large (FP16): 2.48x  +/-20%  (should)
  DGXxAttAccs vs DGX_Base (W8A8):  3.47x  +/-20%  (should)
  DGXxAttAccs vs DGX_Large (W8A8): 2.59x  +/-20%  (should)
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

import sim_runner as sr
from result_aggregator import save

LIN, LOUT, BATCH = 2048, 128, 64


def total_time(metrics):
    """Return prefill + all decode steps in ms."""
    return sr.e2e_ms(metrics, LOUT)


def run_config(model, system, word, label):
    metrics = sr.run(
        model=model,
        system=system,
        gpu="A100a",
        ngpu=8,
        tp=8,
        num_attacc=8,
        num_hbm=5,
        interface="NVLINK3",
        pim="bank",
        lin=LIN,
        lout=LOUT,
        batch=BATCH,
        image_size=672,
        prefill_chunk=512,
        prefill_samples=8,
        max_L=2048,
        powerlimit=True,
        ffopt=True,
        pipeopt=True,
        word=word,
    )
    t = total_time(metrics)
    return {"label": label, "system": system, "word": word, "total_ms": t,
            "s_time": metrics["s_time"] if metrics else None,
            "g_time": metrics["g_time"] if metrics else None}


def main():
    model = "GPT-175B"
    print("R2: AttAcc paper sec.7.2 reproduction -- GPT-175B @ L=2048 batch=64")
    configs = []
    for word, prec in [(2, "FP16"), (1, "W8A8")]:
        base = run_config(model, "dgx", word, "DGX_Base_{}".format(prec))
        attacc = run_config(model, "dgx-attacc", word,
                            "DGX_AttAcc_{}".format(prec))
        configs += [base, attacc]
        print("  {}: DGX_Base={:.1f}ms  DGXxAttAccs={:.1f}ms".format(
            prec, base["total_ms"] or -1, attacc["total_ms"] or -1))

    targets = [
        ("FP16_base",  4.84, 0.20, "DGX_Base_FP16",   "DGX_AttAcc_FP16", "must"),
        ("FP16_large", 2.48, 0.20, "DGX_Large_FP16",  "DGX_AttAcc_FP16", "should"),
        ("INT8_base",  3.47, 0.20, "DGX_Base_W8A8",   "DGX_AttAcc_W8A8", "should"),
        ("INT8_large", 2.59, 0.20, "DGX_Large_W8A8",  "DGX_AttAcc_W8A8", "should"),
    ]
    by_label = {c["label"]: c for c in configs}

    summary = []
    for tag, target, tol, base_lbl, acc_lbl, kind in targets:
        b = by_label.get(base_lbl, {}).get("total_ms")
        a = by_label.get(acc_lbl, {}).get("total_ms")
        if not b or not a:
            summary.append({"tag": tag, "kind": kind, "status": "skip",
                            "target_x": target, "baseline": base_lbl,
                            "attacc": acc_lbl,
                            "reason": "baseline not modeled by current CLI"})
            continue
        speedup = b / a
        status = "PASS" if abs(speedup - target) / target <= tol else "FAIL"
        summary.append({"tag": tag, "kind": kind, "target_x": target,
                        "speedup_x": round(speedup, 3),
                        "tolerance": tol, "status": status})
        print("  {}: speedup={:.2f}x  target={:.2f}x +/-{:.0f}% -> {}".format(
            tag, speedup, target, tol * 100, status))

    save("r2_paper_repro",
         {"model": model, "lin": LIN, "lout": LOUT, "batch": BATCH,
          "platform": "simulated DGX-A100 x8"},
         {"configs": configs, "targets": summary})
    must_fail = any(s["kind"] == "must" and s["status"] != "PASS"
                    for s in summary)
    sys.exit(1 if must_fail else 0)


if __name__ == "__main__":
    main()
