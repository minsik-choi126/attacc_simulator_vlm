"""Save Cosmos experiment results as standard-schema JSON.

`save(name, config, results)` writes both `<name>.json` and
`<name>_<host>.json` to 260604_cosmos_exp/results/, mirroring the
per-host convention from 260511_additional_exp.
"""
import json
import pathlib
import statistics
import subprocess
from datetime import datetime

from hw_detect import detect_host

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _results_dir():
    return pathlib.Path(__file__).resolve().parents[1] / "results"


def save(name, config, results, also_per_host=True):
    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    host = detect_host()
    platform = config.get("platform") if isinstance(config, dict) else None
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "host": host,
        "platform": platform or host,
    }
    payload = {"experiment": name, "config": config,
               "results": results, "metadata": meta}
    targets = [out_dir / f"{name}.json"]
    if also_per_host:
        targets.append(out_dir / f"{name}_{host.lower()}.json")
    for path in targets:
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False, default=str)
    return targets


def load(name):
    path = _results_dir() / f"{name}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fp:
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
        "min": min(values), "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }
