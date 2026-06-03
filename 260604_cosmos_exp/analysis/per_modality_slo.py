"""Phase 4.5 — Per-modality SLO formalization (analysis only).

Combines Phase 4.2 (single-stream P50/P95/P99) + 4.3 (concurrent
dilation) to produce a single SLO table per modality:

    SLO_relax = max(P99_concurrent, max_allowed_user_latency_modality)

Recommended user-facing SLO targets (paper figure anchor):
    video : 1 s for 1-frame (interactive), 10 s for 8 s clip
    image : 200 ms
    audio : 500 ms
    action: 50 ms

Output: results/cosmos_per_modality_slo.json
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import load, save


USER_SLO_TARGETS_MS = {
    "video":  10000,
    "image":  200,
    "audio":  500,
    "action": 50,
}


def main():
    host = detect_host()
    print(f"[Phase 4.5] per_modality_slo_formalization on host={host}")

    budget = load("cosmos_modality_slo_budget") or {}
    interference = load("cosmos_modality_interference") or {}

    by_mod = {}
    for r in budget.get("results", {}).get("rows", []):
        if r["tp"] == 1 and r["batch"] == 1 and r["status"] == "ok":
            by_mod.setdefault(r["modality"], {})["single"] = r["summary"]
    for r in interference.get("results", {}).get("rows", []):
        if "concurrent_summary" in r:
            by_mod.setdefault(r["stream"], {})["concurrent"] = (
                r["concurrent_summary"])
            by_mod[r["stream"]]["p99_dilation"] = r.get("p99_dilation_x")

    rows = []
    for mod, target_ms in USER_SLO_TARGETS_MS.items():
        rec = by_mod.get(mod, {})
        single = rec.get("single", {})
        concur = rec.get("concurrent", {})
        single_p99 = (single.get("p99") or 0) * 1000
        concur_p99 = (concur.get("p99") or 0) * 1000
        slo_ok_single = single_p99 <= target_ms
        slo_ok_concur = concur_p99 <= target_ms if concur_p99 else None
        rows.append({"modality": mod,
                      "user_target_ms": target_ms,
                      "single_stream_p99_ms": round(single_p99, 1),
                      "concurrent_p99_ms": round(concur_p99, 1),
                      "p99_dilation_x": rec.get("p99_dilation"),
                      "slo_met_single_stream": slo_ok_single,
                      "slo_met_concurrent": slo_ok_concur})
        print(f"  {mod:7s} target={target_ms} ms  single P99={single_p99:.0f} ms"
              f"  concurrent={concur_p99:.0f} ms  ok_single={slo_ok_single}"
              f"  ok_concurrent={slo_ok_concur}")

    save("cosmos_per_modality_slo",
          {"phase": "4.5", "host": host, "platform": host,
           "user_slo_targets_ms": USER_SLO_TARGETS_MS},
          {"rows": rows})


if __name__ == "__main__":
    main()
