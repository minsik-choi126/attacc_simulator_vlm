#!/usr/bin/env bash
# Master driver — runs all 260511 additional experiments on H100 × 1.
#
# Usage:
#   bash run_all_h100x1.sh             # run everything
#   bash run_all_h100x1.sh --tier 1    # Tier 1 simulator only
#   bash run_all_h100x1.sh --tier 2sim # Tier 2 simulator
#   bash run_all_h100x1.sh --tier meas # Tier 2 measurement (needs GPU)
#
# Skips GPU-dependent measurement steps if `nvidia-smi` is unavailable.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

TIER="${1:-all}"
if [[ "$TIER" == "--tier" ]]; then
    TIER="${2:-all}"
fi

LOG_DIR="$SCRIPT_DIR/results/_logs"
mkdir -p "$LOG_DIR"
FAILURES=0

run_step () {
    local name="$1"; shift
    local cmd=("$@")
    echo
    echo "=============================================================="
    echo "  RUN: $name"
    echo "=============================================================="
    local log="$LOG_DIR/${name}.log"
    if "${cmd[@]}" 2>&1 | tee "$log"; then
        echo "  -> OK ($log)"
    else
        echo "  -> FAILED ($log)"
        FAILURES=$((FAILURES + 1))
    fi
}

have_gpu () {
    command -v nvidia-smi >/dev/null 2>&1
}

# ============================================================
# Tier 1 — simulator foundation (no GPU needed)
# ============================================================
if [[ "$TIER" == "all" || "$TIER" == "1" || "$TIER" == "1sim" ]]; then
    echo
    echo "###############  Tier 1 — Foundation  ###############"
    run_step r2_paper_repro          python "$SCRIPT_DIR/tier1_simulator/r2_paper_repro.py"
    run_step upstream_baseline       python "$SCRIPT_DIR/tier1_simulator/upstream_baseline.py"
    run_step multi_vlm_full_sim      python "$SCRIPT_DIR/tier1_simulator/multi_vlm_full_sim.py"
    run_step ablation_contribution   python "$SCRIPT_DIR/tier1_simulator/ablation_contribution.py"
    run_step vit_recalibration       python "$SCRIPT_DIR/tier1_simulator/vit_recalibration.py"
fi

# ============================================================
# Tier 2 — simulator sensitivity / architectural
# ============================================================
if [[ "$TIER" == "all" || "$TIER" == "2" || "$TIER" == "2sim" ]]; then
    echo
    echo "###############  Tier 2 — Simulator  ###############"
    run_step chunk_size_sweep        python "$SCRIPT_DIR/tier2_simulator/chunk_size_sweep.py"
    run_step routing_mode_compare    python "$SCRIPT_DIR/tier2_simulator/routing_mode_compare.py"
    run_step eff_lat_ablation        python "$SCRIPT_DIR/tier2_simulator/eff_lat_ablation.py"
    run_step nvlink_compare          python "$SCRIPT_DIR/tier2_simulator/nvlink_compare.py"
    run_step roofline_per_vlm        python "$SCRIPT_DIR/tier2_simulator/roofline_per_vlm.py"
    run_step capacity_regime         python "$SCRIPT_DIR/tier2_simulator/capacity_regime.py"
    run_step pim_mode_compare        python "$SCRIPT_DIR/tier2_simulator/pim_mode_compare.py"
    run_step slo_throughput          python "$SCRIPT_DIR/tier2_simulator/slo_throughput.py"
    # sensitivity_sweep is the long one — run last
    run_step sensitivity_sweep       python "$SCRIPT_DIR/tier2_simulator/sensitivity_sweep.py"
fi

# ============================================================
# Tier 2 — Real H100 measurement (vLLM 0.7.3 + driver 535 compatible)
# ============================================================
if [[ "$TIER" == "all" || "$TIER" == "2" || "$TIER" == "meas" ]]; then
    if have_gpu; then
        echo
        echo "###############  Tier 2 — Measurement (H100)  ###############"
        run_step w4a16_awq_measure       python "$SCRIPT_DIR/tier2_measurement/w4a16_awq_measure.py"
        run_step w8a16_gptq_measure      python "$SCRIPT_DIR/tier2_measurement/w8a16_gptq_measure.py"
        run_step quant_stability_test    python "$SCRIPT_DIR/tier2_measurement/quant_stability_test.py" --n_runs 50
        run_step image_size_sweep        python "$SCRIPT_DIR/tier2_measurement/image_size_sweep.py"
        run_step prompt_pattern_matrix   python "$SCRIPT_DIR/tier2_measurement/prompt_pattern_matrix.py"
    else
        echo
        echo "###############  Tier 2 measurement skipped (no GPU)  ###############"
    fi
fi

echo
echo "##############################################################"
echo "  All requested tiers complete."
echo "  Results: $SCRIPT_DIR/results/*.json"
echo "  Logs:    $LOG_DIR/"
echo "##############################################################"
if [[ "$FAILURES" -ne 0 ]]; then
    echo "  FAILED steps: $FAILURES"
    exit 1
fi
