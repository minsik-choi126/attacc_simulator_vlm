"""Aggregate JSON result files + provide common percentile helpers."""
import json
import pathlib
import statistics
import subprocess
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save(name, config, results, results_dir=None, metadata_extra=None):
    """Write standard-schema JSON to results dir."""
    if results_dir is None:
        results_dir = pathlib.Path(__file__).resolve().parents[1] / "results"
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "platform": "H100x1 driver 535-compatible",
    }
    if metadata_extra:
        meta.update(metadata_extra)
    payload = {
        "experiment": name,
        "config": config,
        "results": results,
        "metadata": meta,
    }
    out = results_dir / "{}.json".format(name)
    with out.open("w") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False, default=str)
    return out


def load(name, results_dir=None):
    if results_dir is None:
        results_dir = pathlib.Path(__file__).resolve().parents[1] / "results"
    path = pathlib.Path(results_dir) / "{}.json".format(name)
    if not path.exists():
        return None
    with path.open() as fp:
        return json.load(fp)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize(values):
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }
