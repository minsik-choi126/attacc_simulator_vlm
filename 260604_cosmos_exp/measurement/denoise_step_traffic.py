"""Phase 3.2 — DRAM read traffic per denoise step (Topic B anchor).

HONEST measurement boundary (read this carefully before citing numbers):

Outer bound  (excluded by cudaProfilerStart/Stop in runner.py):
    + model load, scheduler swap, warmup rep, post-rep cleanup

Inner bound  (included in the nsys capture):
    + text encode (Qwen2 tokenizer + AR tower prefill of text tokens)
    + DENOISE LOOP  -- num_inference_steps forward passes
    + VAE / video decode (latents -> frames)

We DO NOT currently filter the CSV by NVTX per-step ranges (although
runner pushes `cosmos:step_{n}` markers; nsys-CSV-side filtering by
those ranges is version dependent and not implemented here -- see
TODO_NVTX_FILTER).  Therefore:

    measured_per_step_bytes_in_capture
        = total_bytes_in_capture / n_inference_steps
        = (text_encode + denoise_loop + vae_decode) / num_inference_steps
        --> approaches denoise per-step as num_inference_steps grows

    kv_per_step_derived_bytes
        = max(0, measured_per_step - weight_bytes)
        assumes weights are read once per denoise step at H100 BW.

Status semantics:
  'measured'        capture-range nsys + CSV parse succeeded -> the
                    derived KV is honest within the amortization
                    caveat above.  Phase 3.5 consumes this row.
  'analytic_only'   capture failed -- only analytic_upper is reliable.
                    Phase 3.5 ignores this row.

TODO_NVTX_FILTER: a stricter per-step measurement requires
nsys-version-agnostic NVTX-range-bounded gpummu query (or switching
to torch.profiler with PyTorch CUDA mem events) -- left for the
Phase 3.2 v2 once we know which nsys is installed on the H100 host.
"""
import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import save, _results_dir
from cosmos_facts import (
    ALL_MODELS, kv_bytes_per_request, actual_context_tokens,
    TEXT_TOKENS_APPROX,
)


def _have_nsys():
    try:
        subprocess.check_output(["nsys", "--version"], timeout=3)
        return True
    except Exception:
        return False


_BYTE_UNIT = {
    "bytes": 1, "byte": 1, "b": 1,
    "kib": 1024, "kb": 1024,
    "mib": 1024 ** 2, "mb": 1024 ** 2,
    "gib": 1024 ** 3, "gb": 1024 ** 3,
    "tib": 1024 ** 4, "tb": 1024 ** 4,
}
# regex alternation MUST list longer units first so "kb"/"gb" wins
# before "b" -- otherwise "Read KB" gets parsed as bytes (1x).
_UNIT_RE_ALTERNATION = "|".join(
    sorted(_BYTE_UNIT.keys(), key=lambda u: -len(u)))
_UNIT_RE_BRACKET = re.compile(
    r"[\[\(]\s*(" + _UNIT_RE_ALTERNATION + r")\s*[\]\)]")
_UNIT_RE_TRAILING = re.compile(
    r"\b(" + _UNIT_RE_ALTERNATION + r")\b\s*$")


def _unit_multiplier(col_label):
    """Detect byte-unit annotation in a CSV column header.

    Preference order:
      1. explicit bracket annotation: 'Read [KB]', 'Throughput (MB)'
      2. trailing unit token:         'Read KB', 'DRAM Read Bytes'
      3. default = 1 byte
    """
    s = col_label.lower()
    m = _UNIT_RE_BRACKET.search(s)
    if m:
        return _BYTE_UNIT[m.group(1)]
    m = _UNIT_RE_TRAILING.search(s)
    if m:
        return _BYTE_UNIT[m.group(1)]
    return 1


_WRITE_HINTS = ("write", "writes", "written", "store", "stores")


def _is_read_column(h):
    s = h.lower()
    if any(w in s for w in _WRITE_HINTS):
        return False
    return "read" in s


def _has_byte_unit(h):
    s = h.lower()
    return any(re.search(r"\b" + u + r"\b", s) for u in _BYTE_UNIT.keys())


_OPERATION_COL_CANDIDATES = ("operation", "op", "kind", "type")
_READ_ROW_VALUES = ("read", "ld", "load", "dram_read")


def _find_operation_col(headers):
    for h in headers:
        if h.lower().strip() in _OPERATION_COL_CANDIDATES:
            return h
    return None


def _parse_dram_read_total(csv_path):
    """Sum DRAM read bytes from an nsys stats CSV (gpummu / dramread).

    Strict, NO fallback to "non-write byte column".  A column counts as
    a read column only if EITHER:
      (a) the header itself mentions 'read' AND has a byte unit AND
          does NOT mention write/written/store, OR
      (b) there is an 'Operation' / 'Op' / 'Kind' / 'Type' column whose
          rows we can filter to read-only events, and a byte-unit
          column exists (in which case we sum that byte column only
          across rows where the operation row reads).

    Previously a "non-write byte column" fallback was used; that
    incorrectly classified e.g. `Total [MB]` (sum of read+write) as
    read-only, contaminating Phase 3.2's KV-only anchor.  Removed.
    """
    if not csv_path.exists():
        return None, "csv_missing"
    try:
        with csv_path.open() as fp:
            reader = csv.DictReader(fp)
            headers = reader.fieldnames or []
            # Path (a): explicit read column
            explicit_read = [h for h in headers
                              if _is_read_column(h) and _has_byte_unit(h)]
            if explicit_read:
                col = explicit_read[0]
                mult = _unit_multiplier(col)
                total = 0
                for row in reader:
                    v = (row.get(col, "") or "").strip().replace(",", "")
                    try:
                        total += int(float(v) * mult)
                    except (TypeError, ValueError):
                        continue
                if total == 0:
                    return None, f"col={col} parsed_total=0"
                return total, f"col={col} mult={mult}"

            # Path (b): operation-column row filtering
            op_col = _find_operation_col(headers)
            byte_cols = [h for h in headers
                          if _has_byte_unit(h)
                          and not any(w in h.lower() for w in _WRITE_HINTS)]
            if op_col is None or not byte_cols:
                return None, (f"no_read_col_path_a_and "
                               f"no_op_filter_path_b headers={headers[:6]}")
            byte_col = byte_cols[0]
            mult = _unit_multiplier(byte_col)
            total = 0
            n_read_rows = 0
            for row in reader:
                op_val = (row.get(op_col, "") or "").strip().lower()
                if not any(op_val == r or op_val.startswith(r + " ")
                            or op_val.startswith(r + "_")
                            or op_val == r.upper().lower()
                            for r in _READ_ROW_VALUES):
                    continue
                v = (row.get(byte_col, "") or "").strip().replace(",", "")
                try:
                    total += int(float(v) * mult)
                    n_read_rows += 1
                except (TypeError, ValueError):
                    continue
            if total == 0 or n_read_rows == 0:
                return None, (f"op_col={op_col} byte_col={byte_col} "
                               f"n_read_rows={n_read_rows}")
            return total, (f"op_col={op_col}({n_read_rows} read rows) "
                            f"byte_col={byte_col} mult={mult}")
    except Exception as e:
        return None, f"parse_exc={e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano")
    ap.add_argument("--framework", default="pytorch")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--resolution", default="480p")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--analytic-context-tokens", type=int, default=131072,
                    dest="analytic_context_tokens",
                    help="HYPOTHETICAL context length used only for the "
                          "analytic KV/req anchor printed in the same row. "
                          "This is NOT what the workload actually "
                          "generates -- the real context is derived from "
                          "(resolution, frames, text) and stored as "
                          "actual_context_tokens.  Phase 3.5 matches by "
                          "actual_context_tokens, never by this anchor.")
    # Back-compat alias
    ap.add_argument("--context-tokens", type=int, default=None,
                    dest="legacy_context_tokens",
                    help="DEPRECATED alias for --analytic-context-tokens")
    ap.add_argument("--text-tokens-approx", type=int,
                    default=TEXT_TOKENS_APPROX,
                    help="text-token component of actual_context_tokens "
                          "(visual tokens come from resolution+frames)")
    ap.add_argument("--engine-tag", default="PyTorch",
                    choices=["PyTorch", "vLLM-Omni", "Diffusers"],
                    help="affects 256p resolution override "
                          "(Diffusers 256p = 320x192, others = 448x256)")
    args = ap.parse_args()
    if args.legacy_context_tokens is not None:
        args.analytic_context_tokens = args.legacy_context_tokens

    host = detect_host()
    nsys_ok = _have_nsys()
    print(f"[Phase 3.2] denoise_step_traffic on host={host}  nsys={nsys_ok}")

    facts = ALL_MODELS[args.model]
    weight_b = facts["weight_bytes_bf16"]
    # actual = what the workload truly generates (visual + text)
    actual_ctx = actual_context_tokens(facts, args.resolution, args.frames,
                                          text_tokens=args.text_tokens_approx,
                                          engine=args.engine_tag)
    # analytic anchor is the hypothetical KV/req figure
    kv_req_b = kv_bytes_per_request(facts, args.analytic_context_tokens)
    analytic_per_step = kv_req_b * args.batch + weight_b

    measured_total_b = None
    measured_per_step_b = None
    kv_per_step_derived = None
    nsys_rep = None
    nsys_csv = None
    parse_diag = None
    nsys_rc = None

    if nsys_ok:
        rep_dir = _results_dir() / "nsys_reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        stem = (f"step_traffic_{host}_{args.model}_tp{args.tp}_b{args.batch}"
                f"_{args.resolution}_f{args.frames}_s{args.denoise_steps}"
                f"_{int(time.time())}")
        rep = rep_dir / f"{stem}.nsys-rep"

        runner = HERE / args.framework / "runner.py"
        cmd = [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            "--gpu-metrics-device=all",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--output", str(rep.with_suffix("")),
            sys.executable, str(runner),
            "--model", args.model, "--task", "t2v",
            "--tp", str(args.tp), "--batch", str(args.batch),
            "--resolution", args.resolution, "--frames", str(args.frames),
            "--denoise-steps", str(args.denoise_steps),
            "--reps", "1",
            "--nsys-profile-range",
            "--nvtx-per-step",
            "--json",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=3600)
            nsys_rep = str(rep)
            nsys_rc = proc.returncode

            # Try both gpummu (memory-management unit traffic) and
            # cuda_gpu_mem_size_sum (memory-op-level), whichever the nsys
            # version exposes.
            for report_name in ("gpummu", "cuda_gpu_mem_size_sum",
                                  "gpukernsum"):
                csv_path = rep.with_suffix(f".{report_name}.csv")
                try:
                    subprocess.run(["nsys", "stats", "--report",
                                     report_name, "--format", "csv",
                                     "--output", str(csv_path),
                                     str(rep)], timeout=300,
                                    capture_output=True)
                except Exception:
                    continue
                if csv_path.exists():
                    measured_total_b, parse_diag = _parse_dram_read_total(
                        csv_path)
                    if measured_total_b is not None:
                        nsys_csv = str(csv_path)
                        break
            if measured_total_b is not None:
                measured_per_step_b = (measured_total_b
                                         / args.denoise_steps)
                kv_per_step_derived = max(
                    0, measured_per_step_b - weight_b)
        except Exception as e:
            parse_diag = f"profile_exc={e}"

    status = "measured" if measured_per_step_b is not None else "analytic_only"

    row = {
        "model": args.model,
        # Phase 3.5 keys: model + batch + resolution + frames +
        # denoise_steps + engine_tag.  actual_context_tokens is the
        # workload's real context window (visual + text); the analytic
        # anchor is recorded separately and NEVER used as a measured
        # lookup key.
        "batch": args.batch,
        "resolution": args.resolution,
        "frames": args.frames,
        "denoise_steps": args.denoise_steps,
        "engine_tag": args.engine_tag,
        "actual_context_tokens": actual_ctx,
        "text_tokens_approx": args.text_tokens_approx,
        # Analytic anchor (label-only, do not match against)
        "analytic_context_anchor_tokens": args.analytic_context_tokens,
        "kv_bytes_per_request_analytic_at_anchor": kv_req_b,
        "analytic_upper_per_step_bytes": analytic_per_step,
        "weight_bytes": weight_b,
        # Measured anchors
        "measured_total_bytes_in_capture": measured_total_b,
        "measured_per_step_bytes_in_capture": measured_per_step_b,
        "kv_per_step_derived_bytes": kv_per_step_derived,
        "status": status,
    }

    print(f"  actual context (visual+text) = {actual_ctx} tokens "
          f"(engine={args.engine_tag})")
    print(f"  analytic anchor (label only) = "
          f"{args.analytic_context_tokens} tokens")
    print(f"  analytic upper per step = {analytic_per_step/1e9:6.2f} GB")
    print(f"    (= KV {kv_req_b*args.batch/1e9:.2f} + W {weight_b/1e9:.0f})")
    if measured_per_step_b is not None:
        print(f"  measured per step       = "
              f"{measured_per_step_b/1e9:6.2f} GB")
        print(f"  derived KV-only / step  = "
              f"{kv_per_step_derived/1e9:6.2f} GB  "
              f"(measured - weight)")
    else:
        print(f"  measured                = none ({parse_diag})")

    save("cosmos_denoise_step_traffic",
          {"phase": "3.2", "host": host, "platform": host, **vars(args),
           "nsys_available": nsys_ok},
          {"rows": [row],
           "nsys_rep": nsys_rep, "nsys_csv": nsys_csv,
           "nsys_rc": nsys_rc, "parse_diag": parse_diag,
           "schema_note": (
               "Match measured rows by "
               "(model, batch, resolution, frames, denoise_steps, "
               "engine_tag, actual_context_tokens).  Do NOT match on "
               "analytic_context_anchor_tokens -- that field is for "
               "label / anchor printing only."),
           "boundary_note": (
               "status='measured' iff capture-range bound nsys CSV "
               "parse succeeded.  per_step is total / n_inference_steps "
               "within the rep window (model load + warmup are excluded "
               "by cudaProfilerStart/Stop, but text encode + denoise "
               "loop + VAE decode are all included -- amortized).  "
               "kv_per_step_derived = per_step - weight_bytes "
               "(assumes weights are read once per step).")})


if __name__ == "__main__":
    main()
