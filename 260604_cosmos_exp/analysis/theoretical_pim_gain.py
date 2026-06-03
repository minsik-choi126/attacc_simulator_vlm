"""Phase 3.5 — Core deliverable: theoretical PIM gain on Cosmos 3.

KEY INSIGHT (post-review):
    Each denoise step generates traffic = weight_read + kv_read where
    *PIM accelerates only the KV/attention portion*, not weights.

    weight_read_per_step = model_weight_bytes (fixed)
    kv_read_per_step     = KV/tok * context * batch

    t_H100_per_step   = (weight + kv) / H100_BW
    t_AttAcc_per_step = max(weight / H100_BW, kv / ATTACC_BW)
                        # weights still served by H100; PIM handles KV
                        # pipelined; max() approximates overlap

    realized_gain = t_H100 / t_AttAcc

The naive ratio ATTACC_BW / H100_BW = 72x is only the *KV-dominated
ceiling* (where kv_read >> weight_read).  At realistic
(context, batch) we report:

    upper_bound_gain   = 72x   (KV-only assumption)
    realized_gain      = actually achievable given mixed traffic
    kv_dominance_pct   = kv / (kv + weight) * 100  -- closer to 100%
                        means closer to upper bound
    crossover_ctx      = context where kv == weight (b=1)

Phase 3.2 measurements (status=='measured' rows) are NOT injected into
this analytic grid -- the grid's `ctx` values are hypothetical and
typically do not coincide with the actual workload context the
measurement was taken at.  Instead, measured rows are reported in a
separate `measured_anchors` section in the same JSON, each carrying
its own (model, batch, resolution, frames, denoise_steps, engine_tag,
actual_context_tokens).

Output: results/cosmos_theoretical_pim_gain.json (+ per-host copy)
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host, peak_bw_tbs
from result_aggregator import load, save
from cosmos_facts import (
    NANO, SUPER, ATTACC_BW_TBS, H100_BW_TBS, H100_NVL_BW_TBS,
    DEFAULT_DENOISE_STEPS, kv_bytes_per_token, kv_bytes_per_request,
)


CONTEXT_GRID = [16_384, 32_768, 65_536, 131_072, 262_144]
BATCH_GRID = [1, 2, 4, 8]
DENOISE_STEPS_GRID = [10, 20, 35, 50, 100]
MODELS = [NANO, SUPER]


def compute_step_gain(model, context, batch, kv_bytes_override=None,
                      kv_source="analytic"):
    """Per-step BW-bound times for H100 alone vs AttAcc + H100 hybrid.

    If `kv_bytes_override` is provided (from Phase 3.2 measurement), it
    REPLACES the analytic `KV/req * batch` term in *both* the H100 and
    AttAcc time calculations.  `kv_source` is just bookkeeping so
    downstream JSON shows which anchor was used.
    """
    weight_b = model["weight_bytes_bf16"]
    if kv_bytes_override is not None:
        kv_b = kv_bytes_override
    else:
        kv_b = kv_bytes_per_request(model, context) * batch

    t_h100_step_ms = (weight_b + kv_b) / (H100_BW_TBS * 1e12) * 1000
    t_weight_only_ms = weight_b / (H100_BW_TBS * 1e12) * 1000
    t_kv_pim_ms = kv_b / (ATTACC_BW_TBS * 1e12) * 1000
    t_attacc_step_ms = max(t_weight_only_ms, t_kv_pim_ms)

    realized_gain = (t_h100_step_ms / t_attacc_step_ms
                     if t_attacc_step_ms > 0 else None)
    upper_bound_gain = ATTACC_BW_TBS / H100_BW_TBS
    kv_dominance = (kv_b / (kv_b + weight_b) * 100
                     if (kv_b + weight_b) > 0 else 0)

    return {
        "weight_bytes": weight_b,
        "kv_bytes_per_step": kv_b,
        "kv_source": kv_source,
        "kv_dominance_pct": round(kv_dominance, 2),
        "t_H100_per_step_ms": round(t_h100_step_ms, 3),
        "t_AttAcc_per_step_ms": round(t_attacc_step_ms, 3),
        "t_weight_alone_ms": round(t_weight_only_ms, 3),
        "t_kv_via_attacc_ms": round(t_kv_pim_ms, 4),
        "realized_gain_x": round(realized_gain, 2) if realized_gain else None,
        "upper_bound_gain_x": round(upper_bound_gain, 2),
    }


def crossover_context(model, batch=1):
    """Context where kv_bytes == weight_bytes for given batch."""
    kvpt = kv_bytes_per_token(model)
    return int(model["weight_bytes_bf16"] / (kvpt * batch))


def load_measured_anchors():
    """Return all status='measured' rows from Phase 3.2 as a list.

    Each anchor has its own (model, batch, resolution, frames,
    denoise_steps, engine_tag, actual_context_tokens, kv_per_step_bytes).
    Phase 3.5 reports gains at these specific cells AS-IS rather than
    trying to inject them into an analytic (model x ctx x batch) grid.
    """
    p = load("cosmos_denoise_step_traffic")
    if p is None:
        return []
    anchors = []
    for row in p.get("results", {}).get("rows", []):
        if row.get("status") != "measured":
            continue
        kv = (row.get("kv_per_step_derived_bytes")
              # legacy schema (pre-2026-06-04 Fix 1)
              or row.get("dram_read_bytes_per_step"))
        if kv is None:
            continue
        anchors.append({
            "model": row.get("model"),
            "batch": row.get("batch"),
            "resolution": row.get("resolution"),
            "frames": row.get("frames"),
            "denoise_steps": row.get("denoise_steps"),
            "engine_tag": row.get("engine_tag", "PyTorch"),
            # IMPORTANT: actual context (visual+text), NOT analytic anchor
            "actual_context_tokens": row.get("actual_context_tokens"),
            "kv_per_step_bytes_measured": kv,
            "measured_per_step_bytes_in_capture":
                row.get("measured_per_step_bytes_in_capture"),
        })
    return anchors


def measured_kv_per_step(model_name, actual_context, batch,
                         resolution=None, frames=None,
                         denoise_steps=None, engine_tag=None):
    """Strict lookup: returns kv_per_step (bytes) only when ALL specified
    keys match a Phase 3.2 measured row exactly.  Without strict batch
    and resolution / frames matching, a single batch=1 measurement
    would silently masquerade as the truth for batch=8 etc."""
    for a in load_measured_anchors():
        if a["model"] != model_name: continue
        if a["batch"] != batch: continue
        if a["actual_context_tokens"] != actual_context: continue
        if resolution is not None and a["resolution"] != resolution:
            continue
        if frames is not None and a["frames"] != frames: continue
        if denoise_steps is not None and a["denoise_steps"] != denoise_steps:
            continue
        if engine_tag is not None and a["engine_tag"] != engine_tag:
            continue
        return a["kv_per_step_bytes_measured"]
    return None


def main():
    host = detect_host()
    print(f"[Phase 3.5] theoretical PIM gain on host={host}")
    print(f"  AttAcc BW       = {ATTACC_BW_TBS} TB/s  (4-stack paper)")
    print(f"  H100 BW         = {H100_BW_TBS} TB/s  (NVL {H100_NVL_BW_TBS})")
    print(f"  upper-bound gain (KV-only) = {ATTACC_BW_TBS/H100_BW_TBS:.1f}x")
    print()

    rows = []
    for model in MODELS:
        name = model["name"]
        kvpt = kv_bytes_per_token(model)
        cross_b1 = crossover_context(model, batch=1)
        cross_b8 = crossover_context(model, batch=8)
        print(f"=== {name} ===")
        print(f"  weights={model['weight_bytes_bf16']/1e9:.0f} GB  "
              f"KV/tok={kvpt/1024:.1f} KB")
        print(f"  crossover ctx (KV == weight): b=1 -> {cross_b1:,}  "
              f"b=8 -> {cross_b8:,}")
        for ctx in CONTEXT_GRID:
            for batch in BATCH_GRID:
                # Grid is PURELY ANALYTIC.  We do NOT inject measured
                # KV here -- Phase 3.2 measurements are at specific
                # (resolution, frames, batch, steps) cells whose actual
                # context typically does not coincide with this grid's
                # ctx.  Real measured anchors are reported separately in
                # the measured_anchors section below.
                g = compute_step_gain(model, ctx, batch,
                                       kv_source="analytic")
                for steps in DENOISE_STEPS_GRID:
                    row = {"model": name, "context_tokens": ctx,
                           "batch": batch, "denoise_steps": steps,
                           "per_step": g,
                           "video_total_H100_ms":
                               round(g["t_H100_per_step_ms"] * steps, 1),
                           "video_total_AttAcc_ms":
                               round(g["t_AttAcc_per_step_ms"] * steps, 1)}
                    rows.append(row)
                if batch in (1, 8):
                    print(f"  ctx={ctx:>7d} b={batch}  "
                          f"KV/step={g['kv_bytes_per_step']/1e9:6.1f} GB  "
                          f"KV%={g['kv_dominance_pct']:5.1f}  "
                          f"H100/step={g['t_H100_per_step_ms']:7.2f}ms  "
                          f"AttAcc/step={g['t_AttAcc_per_step_ms']:6.2f}ms  "
                          f"-> {g['realized_gain_x']}x  "
                          f"(ceiling {g['upper_bound_gain_x']}x)")
        print()

    # ---- Measured anchors (Phase 3.2 status=='measured' rows) ----
    measured_anchors = []
    model_by_name = {m["name"]: m for m in MODELS}
    for a in load_measured_anchors():
        mdl = model_by_name.get(a["model"])
        if mdl is None:
            continue
        ctx = a["actual_context_tokens"]
        batch = a["batch"]
        g = compute_step_gain(mdl, ctx, batch,
                               kv_bytes_override=a["kv_per_step_bytes_measured"],
                               kv_source="Phase3.2_measured")
        steps = a["denoise_steps"] or 35
        anchor_row = {
            "model": a["model"],
            "batch": batch,
            "resolution": a["resolution"],
            "frames": a["frames"],
            "engine_tag": a["engine_tag"],
            "denoise_steps": steps,
            "actual_context_tokens": ctx,
            "kv_per_step_measured_bytes": a["kv_per_step_bytes_measured"],
            "per_step": g,
            "video_total_H100_ms": round(g["t_H100_per_step_ms"] * steps, 1),
            "video_total_AttAcc_ms":
                round(g["t_AttAcc_per_step_ms"] * steps, 1),
        }
        measured_anchors.append(anchor_row)
        print(f"  [measured anchor] {a['model']:14s} "
              f"b={batch} {a['resolution']} f={a['frames']} "
              f"engine={a['engine_tag']} actual_ctx={ctx} "
              f"-> {g['realized_gain_x']}x")

    config = {
        "phase": "3.5", "host": host, "platform": host,
        "attacc_bw_tbs": ATTACC_BW_TBS,
        "h100_bw_tbs": H100_BW_TBS,
        "h100_nvl_bw_tbs": H100_NVL_BW_TBS,
        "context_grid": CONTEXT_GRID,
        "batch_grid": BATCH_GRID,
        "denoise_steps_grid": DENOISE_STEPS_GRID,
        "model_assumption": ("AttAcc accelerates KV/attention bandwidth "
                             "ONLY; weights still served by H100 at "
                             "H100_BW. Time per step is max(weight/H100, "
                             "KV/AttAcc) under perfect pipelining."),
        "schema_note": (
            "rows[] = pure analytic grid (model x ctx x batch x steps), "
            "no measured override.  measured_anchors[] = one row per "
            "Phase 3.2 status=='measured' cell.  Strict measured-lookup "
            "key is "
            "(model, batch, resolution, frames, denoise_steps, "
            "engine_tag, actual_context_tokens) -- ALL fields must "
            "match exactly; partial matches do not silently substitute."),
    }
    paths = save("cosmos_theoretical_pim_gain", config,
                  {"rows": rows, "measured_anchors": measured_anchors})
    print(f"Saved -> {[str(p) for p in paths]}")
    print("\n*** Paper-hook reading guide ***")
    print("  upper_bound_gain_x (72x) = KV-dominated ceiling")
    print("  realized_gain_x         = actually achievable at this (ctx,b)")
    print("  kv_dominance_pct        = closer to 100 -> realized -> upper")
    print("  Strong AttAcc story requires kv_dominance > ~80% --")
    print("  which is reached at LARGE batch x LONG context.")


if __name__ == "__main__":
    main()
