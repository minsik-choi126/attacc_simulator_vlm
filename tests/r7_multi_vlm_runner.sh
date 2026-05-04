#!/usr/bin/env bash
# R7 — Multi-VLM real-hardware measurement runner.
#
# Iterates the 5 in-framework VLM models defined in plan §0.2 across
# {tp=1, tp=2}. Saves each result to results/r7_{model}_{tp}.json.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p results

declare -A MODEL_IMG=(
    ["Qwen/Qwen3-VL-4B-Instruct"]=672
    ["Qwen/Qwen2.5-VL-7B-Instruct"]=672
    ["OpenGVLab/InternVL3-8B-hf"]=448
    ["llava-hf/llava-1.5-7b-hf"]=336
    ["llava-hf/llava-v1.6-mistral-7b-hf"]=672
)

for model in "${!MODEL_IMG[@]}"; do
    img=${MODEL_IMG[$model]}
    safe=$(echo "$model" | tr '/' '_')
    for tp in 1 2; do
        out="results/r7_${safe}_tp${tp}.json"
        if [ -f "$out" ]; then
            echo "[skip] $out exists"
            continue
        fi
        echo "[R7] $model tp=$tp image=$img"
        python3 tests/r6_h100_measurement.py \
            --model "$model" --tp "$tp" --image_size "$img" \
            --output "$out" 2>&1 | tee -a "results/r7_${safe}_tp${tp}.log" || \
            echo "[FAIL] $model tp=$tp"
    done
done

echo "[R7] All done."
