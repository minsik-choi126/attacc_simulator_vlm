#!/usr/bin/env bash
# R7 — Paper-grade vLLM measurement runner.
#
# Each model is measured at TP=1 and TP=2 (real NCCL all-reduce). Output goes
# to results/r7_{safe_model}_tp{tp}_vllm.json with per-request raw data.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p results

declare -a CASES=(
    "Qwen/Qwen2.5-VL-7B-Instruct 672 2048"
    "OpenGVLab/InternVL3-8B-hf 448 2048"
    "llava-hf/llava-1.5-7b-hf 336 1024"
    "llava-hf/llava-v1.6-mistral-7b-hf 672 2048"
)

for case_spec in "${CASES[@]}"; do
    set -- $case_spec
    model="$1"
    img="$2"
    max_len="$3"
    safe=$(echo "$model" | tr '/' '_')
    for tp in 1 2; do
        out="results/r7_${safe}_tp${tp}_vllm.json"
        if [ -f "$out" ]; then
            echo "[skip] $out exists"
            continue
        fi
        echo "[R7-vllm] $model tp=$tp image=$img max_len=$max_len"
        HF_HOME=/home/elicer/.cache/huggingface python3 tests/r6_vllm_measurement.py \
            --model "$model" --tp "$tp" --image_size "$img" \
            --max_model_len "$max_len" --disable_cudnn \
            --repeats 8 --warmup 2 \
            --output "$out" \
            > "results/r7_${safe}_tp${tp}_vllm.log" 2>&1 \
            || echo "[FAIL] $model tp=$tp (check log)"
    done
done

echo "[R7-vllm] done at $(date '+%H:%M:%S')"
