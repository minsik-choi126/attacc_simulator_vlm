"""Cross-HW comparison -- merge calibration_a6000.json + calibration_h100.json
into a single table showing s_corr / g_corr per HW for each (model, img, batch).

Goal: confirm that simulator modeling is HW-independent. After Fix A
(spatial_merge) + Fix B (floor overhead), s_corr should be in [1.0, 1.3]
range *consistently on both A6000 and H100*. Inconsistency between HW
points to HW-specific calibration bug (bad), consistency points to clean
modeling (good).

Usage:
    cd attacc_simulator
    python 260511_additional_exp/calibration/cross_hw_compare.py
    # → prints table; optional --save writes results/calibration_cross_hw.json
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from result_aggregator import save

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "260511_additional_exp" / "results"


def load_hw(hw_label):
    path = RESULTS / f"calibration_{hw_label.lower()}.json"
    if not path.exists():
        print(f"[warn] missing {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def index_cells(data):
    """Map (label, image_size, lin, batch) -> cell dict.

    Including `lin` in the key prevents merging cells whose configs.py
    `lin` differs across HWs (e.g. LLaVA-Next 336 was 704 before R9 and
    is 1856 now; an old A6000 JSON with lin=704 must NOT match a new
    H100 JSON with lin=1856 just because (label, image_size, batch)
    happens to align).
    """
    if data is None:
        return {}
    return {
        (c["label"], c["image_size"], c.get("lin"), c["batch"]): c
        for c in data["results"]["cells"]
    }


def summarize(label, cell):
    if cell is None:
        return ("—", "—", "—", "—", "—", "—")
    sim = cell.get("sim", {})
    vllm = cell.get("vllm", {})
    if sim.get("status") != "ok" or vllm.get("status") != "ok":
        return ("—", "—", "—", "—", "—", "—")
    return (
        f"{sim['s_ms']:>7.1f}",
        f"{vllm['ttft_ms_p50']:>7.1f}",
        f"{cell.get('s_corr', 0):>5.2f}x",
        f"{sim['g_ms_per_tok']:>6.2f}",
        f"{vllm['itl_ms_p50']:>6.2f}" if vllm.get("itl_ms_p50") else "—",
        f"{cell.get('g_corr', 0):>5.2f}x" if cell.get("g_corr") else "—",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="save merged JSON to results/")
    args = ap.parse_args()

    a6000 = load_hw("A6000")
    h100 = load_hw("H100")

    if a6000 is None and h100 is None:
        print("No calibration files found; run run_calibration.py on each HW first.")
        return

    a_cells = index_cells(a6000)
    h_cells = index_cells(h100)
    all_keys = sorted(set(a_cells) | set(h_cells))

    print(f"\n{'Model':24s} {'img':>4s} {'lin':>5s} {'b':>2s}  "
          f"{'A6000_sim':>9s} {'A6000_meas':>10s} {'A6000_corr':>10s}  "
          f"{'H100_sim':>9s} {'H100_meas':>10s} {'H100_corr':>10s}  "
          f"{'consistency':>11s}")
    print("-" * 138)
    merged = []
    for key in all_keys:
        label, img, lin, batch = key
        a_cell = a_cells.get(key)
        h_cell = h_cells.get(key)
        a_sim, a_meas, a_corr, a_g_sim, a_g_meas, a_g_corr = \
            summarize(label, a_cell)
        h_sim, h_meas, h_corr, h_g_sim, h_g_meas, h_g_corr = \
            summarize(label, h_cell)
        # Consistency check on s_corr — HW-independent modeling target.
        s_a = a_cell.get("s_corr") if a_cell else None
        s_h = h_cell.get("s_corr") if h_cell else None
        if s_a is not None and s_h is not None:
            diff = abs(s_a - s_h) / max(s_a, s_h)
            cons = f"Δ {diff*100:>4.1f}%"
        else:
            cons = "—"
        lin_s = f"{lin:>5d}" if lin is not None else "  —  "
        print(f"{label:24s} {img:>4d} {lin_s} {batch:>2d}  "
              f"{a_sim:>9s} {a_meas:>10s} {a_corr:>10s}  "
              f"{h_sim:>9s} {h_meas:>10s} {h_corr:>10s}  "
              f"{cons:>11s}")
        merged.append({
            "label": label, "image_size": img, "lin": lin, "batch": batch,
            "a6000": a_cell, "h100": h_cell,
            "consistency_delta_pct": (
                abs(s_a - s_h) / max(s_a, s_h) * 100
                if (s_a is not None and s_h is not None) else None),
        })

    print()
    # Aggregate: mean / max s_corr per HW
    def collect(cells, key):
        vals = [c.get(key) for c in cells.values()
                if c and c.get(key) is not None]
        return vals
    a_s = collect(a_cells, "s_corr")
    h_s = collect(h_cells, "s_corr")
    if a_s:
        print(f"A6000 s_corr  min={min(a_s):.2f}x  max={max(a_s):.2f}x  "
              f"mean={sum(a_s)/len(a_s):.2f}x  (n={len(a_s)})")
    if h_s:
        print(f"H100  s_corr  min={min(h_s):.2f}x  max={max(h_s):.2f}x  "
              f"mean={sum(h_s)/len(h_s):.2f}x  (n={len(h_s)})")

    if args.save:
        save("calibration_cross_hw",
             {"hw_compared": ["A6000", "H100"]},
             {"cells": merged})
        print("\nSaved -> results/calibration_cross_hw.json")


if __name__ == "__main__":
    main()
