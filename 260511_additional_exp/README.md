# Additional Experiments for Paper Submission (260511)

This directory collects the additional experiments needed before writing the
VLM + PIM paper results section.  The scripts are split into simulator-only
experiments and real measurement experiments.

Target stack:

- Simulator: repo-root `main.py` + Ramulator2-backed AttAcc PIM path.
  Default deployment is **A6000 / NVLink Bridge**; A1 (TP=1) / A2 (TP=2)
  scenarios are both produced from the same matrix. Override via
  `--gpu` / `--interface` / `--ngpu` / `--tp` for other devices.
- Measurement: vLLM 0.7.3 stack (CUDA 12.x, driver 535-compatible).
  All scripts default to **`--tp 1`** (single GPU). For A2 measurements on
  A6000 x 2, pass `--tp 2` explicitly — currently treated as the test /
  validation path, not the default.
- Output: JSON files under `260511_additional_exp/results/`.

See [`RUNNING_ON_A6000x2.md`](RUNNING_ON_A6000x2.md) for the full A6000 x 2
deployment guide (hardware comparison, A1/A2 scenarios, pass criteria).

The scripts are intended to fail loudly.  If a simulator subprocess fails,
`shared/sim_runner.py` raises by default and the top-level script exits
non-zero.  `run_all_h100x1.sh` also returns non-zero if any step fails.

---

## Required Environment

### Simulator tiers

Run from the `attacc_simulator` repo root or from this directory.

Required:

- Python packages: `pandas`, `numpy`.
- Ramulator2 executable available at one of:
  - `ramulator2/ramulator2`
  - `ramulator2/ramulator2.exe`
  - `ramulator2/build/ramulator2`
  - `ramulator2/build/ramulator2.exe`
- Trace generators available under either:
  - `ramulator2/trace_gen/`
  - `pim_ramulator_src/trace_gen/`

Notes:

- `dgx` GPU-only runs do not need Ramulator2.
- `dgx-attacc` runs need Ramulator2 unless the exact PIM timing row is already
  present in `ramulator.out`.
- The wrapper now treats missing trace generators, missing Ramulator2, or
  zero-cycle Ramulator output as hard failures.

### Measurement tier

Required:

- GPU node with `nvidia-smi` (default deployment: A6000 x 1 or A6000 x 2;
  H100 / A100 also works — only the platform metadata label changes).
- `vllm==0.7.3`, `torch==2.5.1`, `transformers==4.49.0`, `pillow`.
- Hugging Face cache/access for the selected model checkpoints.

The master script skips measurement steps when `nvidia-smi` is unavailable.
It does not install vLLM or download models.

### Shell

`run_all_h100x1.sh` requires bash.  On Windows, run it through WSL/Git Bash or
run the Python scripts directly from PowerShell.

---

## Folder Structure

```text
260511_additional_exp/
├── shared/
│   ├── sim_runner.py            # main.py subprocess wrapper + CSV parser
│   ├── result_aggregator.py     # standard JSON writer + stats helpers
│   ├── plot_helpers.py          # matplotlib helpers
│   └── vllm_helpers.py          # VLM prompt templates with image placeholders
│
├── tier1_simulator/
│   ├── r2_paper_repro.py        # AttAcc GPT-175B paper-repro gate
│   ├── upstream_baseline.py     # Legacy LLM simulator regression sweep
│   ├── multi_vlm_full_sim.py    # VLM GPU-only vs AttAcc S1 matrix
│   ├── ablation_contribution.py # Main modification ablation table
│   └── vit_recalibration.py     # Measured TTFT vs simulated ViT fit
│
├── tier2_simulator/
│   ├── chunk_size_sweep.py
│   ├── routing_mode_compare.py
│   ├── eff_lat_ablation.py
│   ├── sensitivity_sweep.py
│   ├── nvlink_compare.py
│   ├── roofline_per_vlm.py
│   ├── slo_throughput.py
│   ├── capacity_regime.py
│   ├── pim_mode_compare.py
│   └── w4a16_pim_sim.py
│
├── tier2_measurement/
│   ├── w4a16_awq_measure.py
│   ├── w8a16_gptq_measure.py
│   ├── quant_stability_test.py
│   ├── image_size_sweep.py
│   ├── prompt_pattern_matrix.py
│   └── vllm_bf16_baseline_aligned.py
│
├── results/
└── run_all_h100x1.sh
```

---

## Quick Start

From the repo root:

```bash
cd 260511_additional_exp
```

Simulator foundation:

```bash
python tier1_simulator/r2_paper_repro.py
python tier1_simulator/upstream_baseline.py
python tier1_simulator/multi_vlm_full_sim.py
python tier1_simulator/ablation_contribution.py
python tier1_simulator/vit_recalibration.py
```

Tier 2 simulator sweep:

```bash
bash run_all_h100x1.sh --tier 2sim
```

Real-GPU measurements (default A6000 x 1 with `--tp 1`; add `--tp 2`
for A6000 x 2 / TP=2 runs):

```bash
HF_HOME=/your/cache python tier2_measurement/w4a16_awq_measure.py
HF_HOME=/your/cache python tier2_measurement/w8a16_gptq_measure.py
HF_HOME=/your/cache python tier2_measurement/image_size_sweep.py
HF_HOME=/your/cache python tier2_measurement/prompt_pattern_matrix.py
HF_HOME=/your/cache python tier2_measurement/quant_stability_test.py --n_runs 50
HF_HOME=/your/cache python tier2_measurement/vllm_bf16_baseline_aligned.py
```

Everything:

```bash
bash run_all_h100x1.sh
```

Do not treat the experiment set as complete unless `run_all_h100x1.sh` exits
with code 0 and every expected JSON has non-null results.

Tier selection:

```bash
bash run_all_h100x1.sh --tier 1      # Tier 1 simulator foundation only
bash run_all_h100x1.sh --tier 1sim   # Same as --tier 1
bash run_all_h100x1.sh --tier 2sim   # Tier 2 simulator only
bash run_all_h100x1.sh --tier meas   # Real-GPU measurement only (A6000 default)
bash run_all_h100x1.sh --tier 2      # Tier 2 simulator + measurement
```

---

## Output Schema

Each script writes one standard JSON file:

```text
results/<experiment_name>.json
```

Standard shape:

```json
{
  "experiment": "experiment_name",
  "config": {},
  "results": {},
  "metadata": {
    "timestamp": "2026-05-11T...",
    "git_commit": "...",
    "platform": "copied from config.platform"
  }
}
```

Simulator latency convention:

- `s_time`: prefill latency in ms.
- `g_time`: decode latency per generated token in ms/token.
- E2E latency for `lout` tokens is:

```text
e2e_ms = s_time + g_time * max(0, lout - 1)
```

Use `shared/sim_runner.py::e2e_ms()` for this calculation.

---

## Exact Experiment List

This is the exact set currently run by `run_all_h100x1.sh` (legacy filename;
current simulator defaults are A6000 / NVLink Bridge unless a script overrides
them for A100a paper reproduction).

### Tier 1: simulator foundation

| Step / script | Output JSON | What it runs | Success / use |
|---|---|---|---|
| `tier1_simulator/r2_paper_repro.py` | `results/r2_paper_repro.json` | GPT-175B, A100a x8, `lin=2048`, `lout=128`, `batch=64`, FP16 and W8A8, `dgx` vs `dgx-attacc` | Must-pass gate for trusting AttAcc path. Current CLI does not model `DGX_Large`; those targets are skipped. |
| `tier1_simulator/upstream_baseline.py` | `results/upstream_baseline.json` | GPT-175B, GPT-89B, GPT-13B, LLAMA-65B, LLAMA-7B, MT-530B, OPT-66B on A100a x8 `dgx-attacc`, `batch=8` | Regression check that legacy LLM path still runs. |
| `tier1_simulator/multi_vlm_full_sim.py` | `results/multi_vlm_full_sim.json` | Qwen3-VL-4B, Qwen2.5-VL-7B, InternVL3-8B-hf, LLaVA-1.5-7B, LLaVA-Next-Mistral-7B; batches 1/4/8; A6000 x1 A1 `dgx` vs `dgx-attacc` | Main VLM GPU-only vs PIM speedup table. |
| `tier1_simulator/ablation_contribution.py` | `results/ablation_contribution.json` | Qwen3-VL-4B A1 (A6000 x1); `A_no_pim`, `A_no_chunked`, `A_no_routing`, `A_full` | Main ablation table. `A_no_efflat` and `A_no_deepstack` require code-patch reruns. |
| `tier1_simulator/vit_recalibration.py` | `results/vit_recalibration.json` | Qwen2.5-VL-7B, LLaVA-1.5-7B, LLaVA-Next-Mistral-7B measured TTFT/ITL vs simulator | Calibration scatter / prefill correction evidence. |

### Tier 2: simulator sensitivity and architecture

| Step / script | Output JSON | What it runs | Success / use |
|---|---|---|---|
| `tier2_simulator/chunk_size_sweep.py` | `results/chunk_size_sweep.json` | Qwen3-VL-4B, Qwen2.5-VL-7B, LLaVA-1.5-7B, LLaVA-Next-Mistral-7B; chunks 4/16/64/128/256/512/1024/full | Chunked prefill sensitivity. |
| `tier2_simulator/routing_mode_compare.py` | `results/routing_mode_compare.json` | Qwen3-VL-4B, Qwen2.5-VL-7B, LLaVA-1.5-7B, LLaVA-Next-Mistral-7B; conservative/optimistic/list routing | Routing policy sensitivity. |
| `tier2_simulator/eff_lat_ablation.py` | `results/eff_lat_ablation.json` | Five VLMs; reports A1/A2 effective latency factors and A1 simulator latency | Documents §6.1 caveat. Full on/off ablation still needs code patch. |
| `tier2_simulator/nvlink_compare.py` | `results/nvlink_compare.json` | Qwen3-VL-4B, Qwen2.5-VL-7B, LLaVA-1.5-7B; NVLINK_BRIDGE plus NVLink3/4 references; ngpu 1/2 | Simulator-only A2/NVLink comparison. |
| `tier2_simulator/roofline_per_vlm.py` | `results/roofline_per_vlm.json` | Five VLMs; prefill `L=569` and decode `L=1`; qkv/score/context/ffn AI | PIM target justification. Does not require Ramulator2. |
| `tier2_simulator/capacity_regime.py` | `results/capacity_regime.json` | Five VLMs; A1/A2 capacity breakdown (A6000 48 GB) | Capacity-bound vs throughput-bound argument. Does not require Ramulator2. |
| `tier2_simulator/pim_mode_compare.py` | `results/pim_mode_compare.json` | Qwen3-VL-4B, Qwen2.5-VL-7B, LLaVA-1.5-7B; bank/bg/buffer | PIM organization sensitivity. |
| `tier2_simulator/slo_throughput.py` | `results/slo_throughput.json` | Qwen3-VL-4B, Qwen2.5-VL-7B, LLaVA-1.5-7B; batches 1/2/4/8/16/32/64; SLO 30/50/70/100/150/200 ms/token | SLO-throughput curve. |
| `tier2_simulator/sensitivity_sweep.py` | `results/sensitivity_sweep.json` | Qwen3-VL-4B; batch 1/4/8/16/32 x `L` 569/1024/2048 x chunk 16/64/256/512 x PIM layer count 0/11/22/36 | Long heatmap grid, 240 simulator runs. |
| `tier2_simulator/w4a16_pim_sim.py` | `results/w4a16_pim_sim.json` | Five VLMs; batches 1/4/8; analytical projection of FC weight-byte ratio for BF16 / W8A16 / W4A16 over `dgx` and `dgx-attacc` | Sim panel of paper Fig.8 (quant x PIM compound gain); validate against measured w4a16/w8a16 runs. |

### Tier 2: real-GPU measurement (A6000 default; --tp 1 single GPU, --tp 2 A6000 x 2)

| Step / script | Output JSON | Default run | Success / use |
|---|---|---|---|
| `tier2_measurement/w4a16_awq_measure.py` | `results/w4a16_awq_measure.json` | Qwen2.5-VL-7B-AWQ and Qwen2.5-VL-3B-AWQ; image 672; `lout=128`; repeats 4 + warmup 1 | W4A16 latency/power evidence. |
| `tier2_measurement/w8a16_gptq_measure.py` | `results/w8a16_gptq_measure.json` | Qwen2.5-VL-7B GPTQ-Int8; image 672; `lout=128`; repeats 4 + warmup 1 | W8A16 latency/power evidence. |
| `tier2_measurement/quant_stability_test.py` | `results/quant_stability_test.json` | BF16, FP16, W4A16, W8A16; `--n_runs 50` in `run_all`; default script value is 100 | Numerical stability / empty output check. |
| `tier2_measurement/image_size_sweep.py` | `results/image_size_sweep.json` | Qwen2.5-VL-7B BF16; sizes 336/448/672/1008; `lout=128`; repeats 4 + warmup 1 | Visual token / image-size TTFT sensitivity. |
| `tier2_measurement/prompt_pattern_matrix.py` | `results/prompt_pattern_matrix.json` | Qwen2.5-VL-7B BF16; short/medium/long prompt x `lout` 32/128/512; repeats 3 + warmup 1 | Prompt and output-length sensitivity. |
| `tier2_measurement/vllm_bf16_baseline_aligned.py` | `results/vllm_bf16_baseline_aligned.json` | Qwen2.5-VL-7B, LLaVA-1.5-7B, LLaVA-Next-Mistral-7B BF16; batches 1/4/8; same (lin, lout, image_size) as `multi_vlm_full_sim.py`; repeats 3 + warmup 1 | Real-measured side of paper Fig.3 overlay (sim vs measured); Qwen3-VL / InternVL3 deferred to driver 545+ node. |

---

## Run Guide and Validation

### 1. Preflight

From repo root:

```bash
python -m compileall -q src 260511_additional_exp
python -c "import pandas, numpy; print('sim deps ok')"
test -x ramulator2/ramulator2 || test -x ramulator2/ramulator2.exe || test -x ramulator2/build/ramulator2 || test -x ramulator2/build/ramulator2.exe
```

For real-GPU measurement (any of A6000 / A100 / H100):

```bash
nvidia-smi
python -c "import vllm, torch, transformers, PIL; print('measurement deps ok')"
```

### 2. Recommended execution order

Run the smallest deterministic checks first:

```bash
python tier2_simulator/roofline_per_vlm.py
python tier2_simulator/capacity_regime.py
```

Then run simulator gates:

```bash
bash run_all_h100x1.sh --tier 1
bash run_all_h100x1.sh --tier 2sim
```

Finally run measurement on the GPU node (A6000 by default; pass `--tp 2`
to the individual scripts for A6000 x 2 / TP=2 runs):

```bash
bash run_all_h100x1.sh --tier meas
```

### 3. Post-run checklist

Expected JSON files after a full successful run:

```text
results/r2_paper_repro.json
results/upstream_baseline.json
results/multi_vlm_full_sim.json
results/ablation_contribution.json
results/vit_recalibration.json
results/chunk_size_sweep.json
results/routing_mode_compare.json
results/eff_lat_ablation.json
results/nvlink_compare.json
results/roofline_per_vlm.json
results/capacity_regime.json
results/pim_mode_compare.json
results/slo_throughput.json
results/sensitivity_sweep.json
results/w4a16_pim_sim.json
results/w4a16_awq_measure.json
results/w8a16_gptq_measure.json
results/quant_stability_test.json
results/image_size_sweep.json
results/prompt_pattern_matrix.json
results/vllm_bf16_baseline_aligned.json
```

Do this before using the results in paper tables:

- Confirm `run_all_h100x1.sh` exited with code 0.
- Check `results/_logs/*.log` for `FAILED`, `Traceback`, `error`, `nan`.
- Check every JSON has `metadata.git_commit`.
- Check simulator JSON values are not `null` for the plotted metrics.
- Check R2 `must` target status is `PASS`.
- Check measurement JSON files do not contain model-load errors.
- Recompute paper tables from JSON, not from terminal stdout.

---

## Experiment Coverage

| Claim / Figure | Source script(s) | Notes |
|---|---|---|
| Paper repro gate | `tier1_simulator/r2_paper_repro.py` | Must pass before trusting PIM speedup claims. |
| Legacy simulator regression | `tier1_simulator/upstream_baseline.py` | GPT/LLAMA/MT/OPT sanity sweep. |
| Multi-VLM speedup | `tier1_simulator/multi_vlm_full_sim.py` | Main VLM GPU-only vs AttAcc table. |
| Component ablation | `tier1_simulator/ablation_contribution.py` | Some entries require code-patch reruns. |
| ViT calibration | `tier1_simulator/vit_recalibration.py` | Connects measured TTFT to simulator prefill. |
| Chunk sensitivity | `tier2_simulator/chunk_size_sweep.py` | Prefill chunk-size effect. |
| Routing sensitivity | `tier2_simulator/routing_mode_compare.py` | Conservative / optimistic / list. |
| PIM-mode comparison | `tier2_simulator/pim_mode_compare.py` | bank / bg / buffer. |
| SLO throughput | `tier2_simulator/slo_throughput.py` | Uses per-token ITL SLO. |
| Roofline | `tier2_simulator/roofline_per_vlm.py` | Does not require Ramulator2. |
| Capacity regime | `tier2_simulator/capacity_regime.py` | Does not require Ramulator2. |
| Quantization (measured) | `tier2_measurement/w4a16_awq_measure.py`, `tier2_measurement/w8a16_gptq_measure.py` | Real H100/vLLM required. |
| Quantization x PIM (sim panel) | `tier2_simulator/w4a16_pim_sim.py` | Analytical projection; validate against measured AWQ/GPTQ JSONs. |
| Image/prompt sensitivity | `tier2_measurement/image_size_sweep.py`, `tier2_measurement/prompt_pattern_matrix.py` | Real H100/vLLM required. |
| Sim vs measured overlay | `tier2_measurement/vllm_bf16_baseline_aligned.py` paired with `tier1_simulator/multi_vlm_full_sim.py` | Matches `lin/lout/image_size/batch` 1-to-1. Qwen3-VL / InternVL3 deferred to driver 545+. |

---

## Important Limits

- `tier1_simulator/r2_paper_repro.py` only runs the modeled `DGX_Base` and `DGX_AttAcc`
  baselines. `DGX_Large` paper targets are recorded as skipped because the
  current CLI does not model a distinct `DGX_Large` system.
- `tier1_simulator/ablation_contribution.py` can toggle only the components exposed through
  CLI. `A_no_efflat` and `A_no_deepstack` are recorded as
  `requires_code_patch`.
- Qwen3-VL and InternVL3 are simulator-only in the driver-535/vLLM-0.7.3
  measurement stack.
- TP=2 vLLM measurements are deferred to a driver 545+ node because of the
  known NCCL/driver issue on this stack.
- Quantization scope is **W4A16 (AWQ) and W8A16 (GPTQ-Int8)** only -- both
  use public Hugging Face checkpoints. FP8 (Hopper transformer engine) is
  out of scope for this paper.

---

## Driver-535-Compatible Measurement Models

| Type | Model | HF path |
|---|---|---|
| BF16 baseline | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| W4A16 AWQ | Qwen2.5-VL-7B-AWQ | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` |
| W4A16 AWQ | Qwen2.5-VL-3B-AWQ | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` |
| W8A16 GPTQ-Int8 | Qwen2.5-VL-7B-Int8 | `Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int8` |
| BF16 baseline | LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` |
| BF16 baseline | LLaVA-Next-Mistral-7B | `llava-hf/llava-v1.6-mistral-7b-hf` |

Every measurement script uses `shared/vllm_helpers.py` so the prompt includes
the correct model-family image placeholder.

---

## Remaining Measurement Gaps

The scripts above are enough to collect the core simulator and synthetic H100
measurement evidence for the VLM + PIM paper.  They are not the complete set of
all possible paper-grade measurements.

Before writing final claims, verify whether the following are already covered
by existing `results/` or external logs.  If not, add them:

| Gap | Needed if claiming... | How to cover |
|---|---|---|
| Real workload VLM latency | Generalization beyond dummy gray images / synthetic prompts | Run the existing repo-level MMMU-Pro measurement scripts, e.g. `tests/r9_mmmu_pro_measurement.py`, for Qwen2.5-VL and LLaVA family. |
| Concurrent serving contention | Serving/SLO claims under request overlap | Run the existing repo-level concurrent serving script, e.g. `tests/r10_concurrent_serving.py`, or state that SLO-throughput is simulator-only. |
| BF16 baseline matched to quantized runs | W4A16/W8A16 speedup vs BF16 on identical prompt/image setup | Reuse existing BF16 H100 JSON only if prompt/image/`lout` match; otherwise rerun a BF16 baseline. |
| TP=2 real measurement | S2 real-system validation | Requires driver 545+ or a node where vLLM TP=2/NCCL works. |
| Full `A_no_efflat` / `A_no_deepstack` ablation | Complete component-attribution figure | Patch simulator toggles and rerun `tier1_simulator/ablation_contribution.py`. |
| `DGX_Large` paper baseline | Direct comparison to AttAcc paper's 2.48x / 2.59x large-baseline targets | Add a distinct `DGX_Large` system model or mark those paper targets as skipped. |

Minimum paper-ready set for the VLM + PIM results section:

- R2 paper-repro gate: `PASS` for the modeled must target.
- Multi-VLM GPU-only vs AttAcc table.
- Ablation table with clearly marked code-patch-only entries.
- Chunk/routing/PIM-mode/sensitivity sweeps.
- Roofline and capacity-regime justification.
- H100 BF16 measurement calibration plus quantization/image/prompt sensitivity.
- Real-workload latency evidence, unless the paper explicitly scopes results to
  synthetic VLM prompts.

---

## Per-script CLI Reference

Every Tier 1 simulator and the four no-subprocess analytic scripts
(`roofline_per_vlm.py`, `capacity_regime.py`) take **no CLI arguments**.
Model / batch / `lin` / `lout` are fixed inside the script.  Change the
constants at the top of the file to alter the matrix.

Tier 2 simulator scripts also have no CLI arguments today; the grids are
defined inline (`BATCHES`, `LS`, `CHUNKS`, `PIM_MODES`, etc.).

Tier 2 measurement scripts accept the following:

| Script | Flag | Default | Description |
|---|---|---|---|
| `w4a16_awq_measure.py` | `--models` | Qwen2.5-VL-7B-AWQ, Qwen2.5-VL-3B-AWQ | HF paths; pass space-separated list |
| `w4a16_awq_measure.py` | `--image_size` | 672 | square dummy image side |
| `w4a16_awq_measure.py` | `--lout` | 128 | fixed decode length (uses `min_tokens=lout`) |
| `w4a16_awq_measure.py` | `--repeats` | 4 | measured iterations |
| `w4a16_awq_measure.py` | `--warmup` | 1 | warmup iterations (excluded from stats) |
| `w8a16_gptq_measure.py` | same flags as W4A16 | Qwen2.5-VL-7B-GPTQ-Int8 | |
| `quant_stability_test.py` | `--n_runs` | 100 | runs per variant (master sh: 50) |
| `quant_stability_test.py` | `--variants` | all four | subset filter: `BF16 FP16 W4A16 W8A16` |
| `quant_stability_test.py` | `--lout`, `--image_size` | 64 / 672 | |
| `image_size_sweep.py` | `--sizes` | 336 448 672 1008 | space-separated image sizes |
| `image_size_sweep.py` | `--model`, `--lout`, `--repeats`, `--warmup` | Qwen2.5-VL-7B / 128 / 4 / 1 | |
| `prompt_pattern_matrix.py` | `--model`, `--image_size`, `--repeats`, `--warmup` | Qwen2.5-VL-7B / 672 / 3 / 1 | matrix `short/medium/long × lout {32,128,512}` fixed inside script |
| `vllm_bf16_baseline_aligned.py` | `--models` | ALIGNED_CONFIGS (Qwen2.5-VL-7B, LLaVA-1.5-7B, LLaVA-Next-Mistral-7B) | HF paths filter; only entries matching are run |
| `vllm_bf16_baseline_aligned.py` | `--batches`, `--lout`, `--repeats`, `--warmup`, `--prompt` | 1 4 8 / 128 / 3 / 1 / "Describe this image with 100 specific words." | aligned to `multi_vlm_full_sim.py` defaults |

To change a Tier 1 / Tier 2 simulator matrix without editing the source,
copy the script to a sandbox name and adjust the constant lists at the top
(`VLM_CONFIGS`, `MODELS`, `BATCHES`, `CHUNKS`, etc.).

---

## Calibration Refresh (vit_recalibration)

`tier1_simulator/vit_recalibration.py` compares the simulator's prefill
latency against a hard-coded `MEASURED` dict.  The dict currently uses
results from the previous campaign (MMMU-Pro real data, n=32).  When a new
H100 measurement campaign is run, refresh as follows:

1. Run the measurement scripts you trust (e.g.
   `tier2_measurement/w4a16_awq_measure.py`, an MMMU-Pro script, etc.).
2. Inspect the resulting JSON's `results.per_model[*].stats.ttft_ms.p50`
   and `results.per_model[*].stats.itl_ms.p50`.
3. Open `tier1_simulator/vit_recalibration.py` and update the
   `MEASURED` dict entries:

   ```python
   MEASURED = {
       "Qwen2.5-VL-7B": {
           "ttft_ms_p50": <new p50 ms>,
           "itl_ms_p50":  <new p50 ms/tok>,
           "image_size":  <int>,
           "lin_text":    <median seq_in tokens>,
           "lout":        <int>,
       },
       ...
   }
   ```

4. Rerun `python tier1_simulator/vit_recalibration.py`.

Do **not** commit MEASURED dict updates without rerunning the script and
checking that `g_corr` stays within +-30% across all listed models.  A
sudden jump usually means the lin_text/image_size mismatch between
simulator and measured workload.

---

## Component-patch Ablations (deepstack / efflat)

`A_no_efflat` and `A_no_deepstack` cannot be toggled from the CLI.  To run
them yourself:

### A_no_efflat

In `src/system.py`, locate `_apply_eff_lat()`.  Either:

- Make it a no-op: `return` immediately at the top of the function, or
- Comment out the line that divides exec_time by `eff_lat`.

Then rerun `python tier1_simulator/ablation_contribution.py` and rename
the resulting JSON (e.g. `results/ablation_contribution_no_efflat.json`).

### A_no_deepstack

In `src/config.py`, locate the `Qwen3-VL-4B` entry and set
`'has_deepstack': False` (or `'deepstack_layers': []`).  Rerun
`python tier1_simulator/ablation_contribution.py` and rename the JSON.

Restore the original code before running other experiments — both toggles
affect every PIM simulation in the repo, not only this script.

---

## Troubleshooting

### Simulator subprocess failures

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Missing ramulator2 executable under ramulator2` | Ramulator2 binary not built | Re-run Prerequisites step 0 (set_pim_ramulator.sh + cmake build) |
| `Trace generator not found` | `pim_ramulator_src/trace_gen/` missing | `git submodule update --init --recursive` |
| `ModuleNotFoundError: No module named 'pandas'` | pandas missing | `pip install -r requirements.txt` |
| Simulator returns 0 cycles | empty Ramulator trace (silent corruption) | Delete `ramulator.out` cache file and rerun; if persists, check Ramulator2 binary version |
| `KeyError: 'g_time'` from a downstream script | Old `output.csv` header (column rename) | Delete `output.csv` and rerun; do not parse stale CSV |

### vLLM / measurement failures

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: CUDA driver version is insufficient` | driver 535 + TP>=2 NCCL | Defer to driver 545+ node, or pass `--tp 1` |
| `ImportError: No module named 'vllm'` | vLLM not installed | `pip install vllm==0.7.3` (driver-535 compatible) |
| `Unsupported quantization` | model checkpoint mismatch with `quantization=...` | Verify HF path matches AWQ/GPTQ checkpoint exactly |
| HF download stalls | cache directory full or no internet | Set `HF_HOME=/big/disk`, pre-`huggingface-cli download <repo>` |
| Empty / NaN outputs only on FP16 | known long-VLM-sequence FP16 overflow | Use BF16 baseline; record in `quant_stability_test` results |

### Windows / encoding

| Symptom | Fix |
|---|---|
| `UnicodeEncodeError: 'cp949' codec` | Run with `chcp 65001` (cmd) or `$env:PYTHONIOENCODING="utf-8"` (PowerShell) |
| Bash master script fails to run | Use WSL or Git Bash; or run each Python script directly |

### Partial-run recovery

`run_all_h100x1.sh` writes per-step logs to `results/_logs/<step>.log`.
A failed step does not abort the whole driver — `FAILURES` counter increments
and final exit code is non-zero.  Re-run individual failing scripts directly
(see Per-script CLI Reference); no need to rerun the master script.

To clear stale outputs before a clean run:

```bash
rm -f results/*.json results/_logs/*.log output.csv
```

Do **not** delete `ramulator.out` unless you suspect cache corruption —
it speeds up re-runs dramatically.
