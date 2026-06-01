# Calibration framework — simulator vs vLLM, cross-HW

One-shot script that runs the AttAcc simulator and vLLM measurement with
identical configurations (model × image_size × lin × batch) on a given
GPU, then emits per-cell `s_corr` (prefill correction) and `g_corr`
(decode correction). Run on each GPU (A6000 / H100) separately, then
merge with `cross_hw_compare.py`.

Goal: after Fix A (spatial_merge timing) + Fix B (floor overhead), `s_corr`
should be in `[1.0, 1.3]` consistently on both A6000 and H100 — i.e. the
simulator modeling is HW-independent, not calibrated to one GPU.

## Files

- `configs.py` — model × image_size × lin matrix (shared with both HW)
- `run_calibration.py` — main runner (auto-detects HW)
- `cross_hw_compare.py` — merges A6000 + H100 JSONs into one table

## Usage

On each HW:

```bash
cd attacc_simulator
python 260511_additional_exp/calibration/run_calibration.py --mode both
# -> results/calibration_a6000.json   (on A6000)
# -> results/calibration_h100.json    (on H100)
```

Options:

```bash
--mode {sim,vllm,both}        # default both
--models Qwen2.5-VL-7B        # filter by label (comma-separated)
--batches 1,4                 # filter batch sizes
```

After both HW have run, merge on any machine that has both JSONs:

```bash
python 260511_additional_exp/calibration/cross_hw_compare.py
python 260511_additional_exp/calibration/cross_hw_compare.py --save
# -> results/calibration_cross_hw.json
```

## Output schema

`calibration_<hw>.json`:

```json
{
  "experiment": "calibration_<hw>",
  "config": {"hw": "A6000", "mode": "both", "lout": 128, "batches": [1, 4, 8]},
  "results": {
    "cells": [
      {
        "label": "Qwen2.5-VL-7B", "image_size": 672, "lin": 704, "batch": 1,
        "sim": {"status": "ok", "s_ms": 13.6, "g_ms_per_tok": 8.7, ...},
        "vllm": {"status": "ok", "ttft_ms_p50": 107.4, "itl_ms_p50": 22.7, ...},
        "s_corr": 7.90,
        "g_corr": 2.61
      },
      ...
    ]
  }
}
```

## Calibration target

Baseline (before fixes):

| Model                  | s_corr (A6000) | g_corr (A6000) |
|------------------------|----------------|----------------|
| Qwen2.5-VL-7B          | **7.9x**       | 1.50x          |
| LLaVA-1.5-7B           | 2.2x           | 1.22x          |
| LLaVA-Next-Mistral-7B  | 1.09x          | 1.04x          |

After Fix A (spatial_merge timing in `compute_vit_attention_tokens`):

- Qwen2.5-VL `s_corr` predicted **7.9x -> 1.3x** (ViT attention compute no
  longer 16x under-modeled).

After Fix B (common floor overhead):

- All VLMs `s_corr` predicted to converge into `[1.0, 1.3]` on both A6000 and
  H100.

After Fix C (capacity model — KV on AttAcc side for `dgx-attacc`):

- Paper r2 reproduction becomes batch-derived (no need to hardcode 54/854).
- `capacity_regime` LLaVA max batch (88 / 91 on A6000) becomes more accurate.

## Known limitations

- `Qwen3-VL-4B` and `InternVL3-8B-hf` may fail vLLM 0.7.3 load (driver 545+
  required); will report `status: load_failed` in the JSON cell.
- vLLM measurement loads each model once and reuses across `(image_size, batch)`
  combinations to amortize model load time (~30s each).
- The simulator side does not need ramulator2 binary for `system="dgx"` (the
  default in this script); for `dgx-attacc` validation, ramulator2 must be
  built on the host.
