"""Shared simulator wrapper: invokes main.py and parses output.csv.

Hides argparse glue + CSV column ordering for downstream scripts.
"""
import csv
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_CSV = ROOT / "output.csv"

# Aliases between friendly keys (caller-facing) and actual CSV column names.
# main.py write_csv() writes the right side; callers use the left side.
COL_ALIAS = {
    "g_time":       "g_time (ms)",
    "g_energy":     "g_energy (nJ)",
    "required_cap": "required_cap_per_gpu",
}


def run(
    model,
    *,
    system="dgx-attacc",
    gpu="A6000",            # Deployment default. Use gpu="A100a" for R2 paper repro.
    ngpu=1,
    tp=1,
    num_attacc=1,
    num_hbm=5,
    interface="NVLINK_BRIDGE",  # A6000 NVLink Bridge 112 GB/s.
    pim="bank",
    lin=569,
    lout=128,
    batch=1,
    image_size=672,
    prefill_chunk=512,
    prefill_samples=8,
    max_L=2048,
    routing="default",
    pim_layers="",
    powerlimit=False,
    ffopt=False,
    pipeopt=False,
    word=2,
    extra=None,
    strict=True,
    capture=("s_time", "g_time", "g_qkv_time", "g_prj_time", "g_ff_time",
             "g2g_comm", "c2g_comm", "g_softmax", "g_energy", "required_cap",
             "s_flops", "g_flops"),
):
    """Run main.py with given config, return parsed dict of captured columns.

    Returns None on failure.
    """
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable, "main.py",
        "--system", system,
        "--gpu", gpu,
        "--ngpu", str(ngpu),
        "--tp", str(tp),
        "--num_attacc", str(num_attacc),
        "--num_hbm", str(num_hbm),
        "--interface", interface,
        "--pim", pim,
        "--model", model,
        "--lin", str(lin),
        "--lout", str(lout),
        "--batch", str(batch),
        "--image_size", str(image_size),
        "--prefill_chunk", str(prefill_chunk),
        "--prefill_samples", str(prefill_samples),
        "--max_L", str(max_L),
        "--routing", routing,
        "--word", str(word),
    ]
    if pim_layers:
        cmd += ["--pim_layers", pim_layers]
    if powerlimit:
        cmd += ["--powerlimit"]
    if ffopt:
        cmd += ["--ffopt"]
    if pipeopt:
        cmd += ["--pipeopt"]
    if extra:
        cmd += list(extra)
    try:
        subprocess.run(
            cmd, cwd=str(ROOT), env=env, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        message = "[sim_runner] FAILED: {} | {}".format(
            " ".join(cmd), stderr[-2000:])
        if strict:
            raise RuntimeError(message) from None
        sys.stderr.write(message + "\n")
        return None

    if not OUTPUT_CSV.exists():
        if strict:
            raise RuntimeError("[sim_runner] output.csv was not produced")
        return None
    with OUTPUT_CSV.open() as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        if strict:
            raise RuntimeError("[sim_runner] output.csv contained no rows")
        return None
    row = rows[-1]
    out = {}
    for k in capture:
        csv_key = COL_ALIAS.get(k, k)
        v = row.get(csv_key)
        if v is None or v == "":
            out[k] = None
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    out["_raw"] = row
    return out


def s_g(model, **kwargs):
    """Convenience: return (s_time, g_time) tuple in ms."""
    r = run(model, **kwargs)
    if r is None:
        return None, None
    return r.get("s_time"), r.get("g_time")


def decode_total_ms(metrics, lout):
    """Return total decode latency from per-token g_time."""
    if metrics is None:
        return None
    g = metrics.get("g_time")
    if g is None:
        return None
    return g * max(0, int(lout) - 1)


def e2e_ms(metrics, lout):
    """Return prefill + all decode steps in ms."""
    if metrics is None:
        return None
    s = metrics.get("s_time")
    decode = decode_total_ms(metrics, lout)
    if s is None or decode is None:
        return None
    return s + decode
