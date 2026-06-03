"""B4 -- Capacity-argument framing: reframe existing slo_throughput.json
into batch_ratio x ITL_ratio decomposition.

This is data reuse (no new simulator runs).  For each (model, SLO),
read the existing slo_throughput result and report:

    batch_ratio   = max_batch_attacc / max_batch_dgx
    itl_ratio     = ITL_dgx / ITL_attacc
    speedup_total = throughput_attacc / throughput_dgx
    sanity        = batch_ratio * itl_ratio
                   ~= speedup_total  (should be within 1-2%)

Goal: paper Table cell that explains why LLaVA-1.5's 19.9x SLO speedup
is bigger than the GPT-175B 4.84x -- the VLM's MHA + larger KV/req
inflates batch headroom enormously, while ITL also shrinks (decode
attention PIM gain).

Output: results/capacity_framing.json
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from result_aggregator import save
from hw_detect import detect_host

HOST = detect_host()
ROOT = pathlib.Path(__file__).resolve().parents[2]
# Per-host slo_throughput is REQUIRED so we never accidentally consume
# another host's SLO numbers (e.g. running capacity_framing on H100 and
# silently picking up an A6000-era `slo_throughput.json`).  If the
# per-host file is missing, fail-fast in main().
_RESULTS_DIR = ROOT / "260511_additional_exp" / "results"
SLO_JSON = _RESULTS_DIR / f"slo_throughput_{HOST.lower()}.json"


def frame_one(model_entry):
    """For one model in slo_throughput.json, build per-SLO decomposition."""
    label = model_entry["model"]
    systems = {s["label"]: s for s in model_entry["systems"]}
    gpu = systems.get("GPU only")
    att = systems.get("AttAcc A1")
    if not gpu or not att:
        return None

    # SLO points share the same SLO list order.
    rows = []
    for slo_gpu, slo_att in zip(gpu["slo_points"], att["slo_points"]):
        if slo_gpu["slo_per_token_ms"] != slo_att["slo_per_token_ms"]:
            continue
        slo = slo_gpu["slo_per_token_ms"]
        b_gpu = slo_gpu.get("best")
        b_att = slo_att.get("best")
        if not b_gpu or not b_att:
            continue
        batch_gpu = b_gpu["batch"]
        batch_att = b_att["batch"]
        itl_gpu = b_gpu["g_ms_per_tok"]
        itl_att = b_att["g_ms_per_tok"]
        tput_gpu = b_gpu["throughput_tok_per_sec"]
        tput_att = b_att["throughput_tok_per_sec"]

        batch_ratio = batch_att / batch_gpu
        itl_ratio = itl_gpu / itl_att
        speedup_total = tput_att / tput_gpu
        expected = batch_ratio * itl_ratio
        delta = (expected - speedup_total) / speedup_total * 100.0

        rows.append({
            "slo_per_token_ms": slo,
            "batch_gpu": batch_gpu, "batch_attacc": batch_att,
            "itl_gpu_ms": itl_gpu, "itl_attacc_ms": itl_att,
            "throughput_gpu": tput_gpu, "throughput_attacc": tput_att,
            "batch_ratio": batch_ratio,
            "itl_ratio": itl_ratio,
            "speedup_total": speedup_total,
            "decomposition_expected": expected,
            "decomposition_delta_pct": delta,
        })
    return {"model": label, "rows": rows}


def main():
    print(f"B4 -- Capacity argument framing on host={HOST} "
          f"(data reuse from slo_throughput)")
    if not SLO_JSON.exists():
        # Fail-fast.  Do NOT fall back to host-agnostic slo_throughput.json
        # -- that file may contain another host's results, which would
        # silently contaminate the capacity-framing output labelled as
        # this host.  See 260601_experiment.md, v5 finding High-1.
        print(f"  MISSING per-host input: {SLO_JSON}")
        print(f"  Run tier2_simulator/slo_throughput.py on host={HOST} "
               f"first to produce slo_throughput_{HOST.lower()}.json.")
        sys.exit(2)
    data = json.loads(SLO_JSON.read_text(encoding="utf-8"))
    models = data["results"]["models"]
    out = []
    for entry in models:
        framed = frame_one(entry)
        if framed is None:
            continue
        out.append(framed)
        print(f"\n=== {framed['model']} ===")
        print(f"  {'SLO':>4s} {'b_gpu':>6s} {'b_att':>6s} "
              f"{'ITL_gpu':>8s} {'ITL_att':>8s} "
              f"{'b_ratio':>8s} {'itl_ratio':>10s} "
              f"{'total':>7s} {'check':>7s}")
        for r in framed["rows"]:
            print(f"  {r['slo_per_token_ms']:>4d} {r['batch_gpu']:>6d} "
                  f"{r['batch_attacc']:>6d} "
                  f"{r['itl_gpu_ms']:>8.2f} {r['itl_attacc_ms']:>8.2f} "
                  f"{r['batch_ratio']:>7.2f}x {r['itl_ratio']:>9.2f}x "
                  f"{r['speedup_total']:>6.2f}x "
                  f"{r['decomposition_delta_pct']:>+5.1f}%")

    meta = {"source": str(SLO_JSON.name),
            "host_detected": HOST,
            "platform": f"{HOST} simulator A1 (capacity-framing decomp)",
            "decomposition": "speedup_total = batch_ratio x itl_ratio"}
    payload = {"models": out}
    save("capacity_framing", meta, payload)
    save(f"capacity_framing_{HOST.lower()}", meta, payload)
    print(f"\nSaved -> results/capacity_framing{{,_{HOST.lower()}}}.json")


if __name__ == "__main__":
    main()
