"""Master orchestrator -- runs Phase 1 (calibration) + Phase 2 (gain)
+ Phase 3 (figure refresh) in order, with env sanity check at the start.

Stops on first failure unless --continue-on-error is passed.

Usage on each HW:
    cd attacc_simulator
    python 260511_additional_exp/calibration/run_all.py

Skip specific phases:
    --skip phase1            # accuracy
    --skip phase2            # vlm gain
    --skip phase3            # figure refresh
    --skip regression        # upstream_baseline check

See 260601_experiment.md for per-script details.
"""
import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]


# (script_path_relative_to_root, label, phase_tag)
STEPS = [
    # --- env sanity ---
    ("260511_additional_exp/tier1_simulator/upstream_baseline.py",
        "Step 0  upstream_baseline LLM regression",  "regression"),
    # --- Phase 1: accuracy ---
    ("260511_additional_exp/calibration/run_calibration.py",
        "Step 1  Sim vs vLLM calibration (batch 1-128)", "phase1"),
    ("260511_additional_exp/tier1_simulator/vit_recalibration.py",
        "Step 2  ViT recalibration (legacy comparison)", "phase1"),
    # --- Phase 2: VLM gain ---
    ("260511_additional_exp/tier1_simulator/vlm_vs_llm_pair.py",
        "Step 3  LLM <-> VLM pair speedup (B1)",        "phase2"),
    ("260511_additional_exp/tier1_simulator/prefill_decomp_vlm.py",
        "Step 4  Prefill vs decode decomposition (B2)", "phase2"),
    ("260511_additional_exp/tier1_simulator/visual_token_scaling.py",
        "Step 5  Visual token sensitivity (B3)",        "phase2"),
    ("260511_additional_exp/tier1_simulator/capacity_framing.py",
        "Step 6  Capacity argument framing (B4)",       "phase2"),
    # --- Phase 3: figure refresh ---
    ("260511_additional_exp/tier1_simulator/multi_vlm_full_sim.py",
        "Step 7  Multi-VLM speedup matrix refresh",     "phase3"),
    ("260511_additional_exp/tier2_simulator/slo_throughput.py",
        "Step 8  SLO throughput refresh",               "phase3"),
    ("260511_additional_exp/tier2_simulator/roofline_per_vlm.py",
        "Step 9  Roofline refresh",                     "phase3"),
    ("260511_additional_exp/tier2_simulator/capacity_regime.py",
        "Step 10 Capacity regime refresh",              "phase3"),
]


def check_env():
    """Quick environment sanity report."""
    print("=" * 70)
    print("Environment check")
    print("=" * 70)
    # GPU
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=4).decode().strip()
        print(f"  GPU:        {gpu.splitlines()[0]}")
    except Exception:
        print("  GPU:        nvidia-smi not available")
    # Ramulator2 binary
    ram_paths = [
        ROOT / "ramulator2" / "ramulator2",
        ROOT / "ramulator2" / "ramulator2.exe",
        ROOT / "ramulator2" / "build" / "ramulator2",
    ]
    ram_found = next((p for p in ram_paths if p.exists()), None)
    if ram_found:
        print(f"  Ramulator2: {ram_found}")
    else:
        print("  Ramulator2: NOT FOUND -- dgx-attacc sims will fail")
        print("              Build with: cd ramulator2 && mkdir build && "
              "cd build && cmake .. && make -j && cp ramulator2 ../ramulator2")
    # vLLM
    try:
        import vllm
        print(f"  vLLM:       {vllm.__version__}")
    except ImportError:
        print("  vLLM:       NOT INSTALLED -- calibration vllm mode will fail")
    print()


def run_step(script_rel, label):
    print("=" * 70)
    print(f"{label}")
    print(f"  -> {script_rel}")
    print("=" * 70)
    t0 = time.time()
    env = {"PYTHONIOENCODING": "utf-8"}
    import os
    env.update(os.environ)
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / script_rel)],
            cwd=str(ROOT), env=env,
            check=False,
        )
        dt = time.time() - t0
        ok = proc.returncode == 0
        print(f"\n  -> {'OK' if ok else 'FAIL'}  (exit {proc.returncode}, "
              f"{dt:.1f}s)\n")
        return ok
    except Exception as e:
        print(f"\n  -> EXCEPTION: {e}\n")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma-separated phase tags to skip "
                    "(regression / phase1 / phase2 / phase3)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going even if a step fails")
    args = ap.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    check_env()

    summary = []
    for rel, label, phase in STEPS:
        if phase in skip:
            print(f"-- SKIP ({phase}): {label}\n")
            summary.append((label, "SKIP"))
            continue
        ok = run_step(rel, label)
        summary.append((label, "OK" if ok else "FAIL"))
        if not ok and not args.continue_on_error:
            print(f"Stopping at first failure (use --continue-on-error to override).")
            break

    print("=" * 70)
    print("Run summary")
    print("=" * 70)
    for label, status in summary:
        print(f"  [{status}]  {label}")


if __name__ == "__main__":
    main()
