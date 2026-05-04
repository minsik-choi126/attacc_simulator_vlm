# AttAcc Simulator — VLM-aware Extension

> **Forked from** [scale-snu/attacc_simulator](https://github.com/scale-snu/attacc_simulator) (ASPLOS 2024 AttAcc paper). This fork extends the original LLM-only decode-time simulator to support GQA / VLM (Vision-Language Models) / chunked prefill / dual-deployment (H100 × 1 and H100 × 2) scenarios while preserving the original GPT / LLAMA / OPT paper-reproduction path.

For the original AttAcc design, see the upstream README section below or the [paper](https://dl.acm.org/doi/10.1145/3620665.3640422) published at [ASPLOS 2024](https://www.asplos-conference.org/asplos2024).

---

## Why this fork

Upstream `scale-snu/attacc_simulator` was built around GPT-3 175B style MHA decoder-only inference on DGX-A100 (8 GPUs × 5 HBMs × AttAcc). To analyze modern VLM serving on smaller deployments (e.g., 1–2 H100 with NVLink4), the following gaps had to be closed:

1. **GQA / MQA shape contract** — original assumed `num_kv_heads == num_q_heads` and `qkv_proj_out = 3 × hdim`. Modern Qwen3-VL / Qwen2.5-VL / InternVL3 / LLaVA-Next have `num_kv_heads ≪ num_q_heads`.
2. **PIM trace head count** — Ramulator wrapper used `layer.numOp` directly, conflating GPU compute (Q heads) with PIM trace (KV-read-bound, KV heads). For GQA this overcounted PIM time by `gqa_size`.
3. **Vision graph** — no ViT, no projector, no DeepStack injection, no LLaVA-Next anyres.
4. **Per-layer routing** — single `template × ndec` scaling collapsed all decoder layers into one. DeepStack injection at layers 5/11/17 (Qwen3-VL) needed depth-aware modeling.
5. **Deployment configurability** — `num_attacc=8`, `num_hbm=5`, `NVLINK3` were hardcoded. H100 × 1 (TP=1, AttAcc=1) and H100 × 2 (TP=2, AttAcc=2, NVLink4) needed first-class support.
6. **PIM softmax double-count** — upstream PIM softmax returned non-zero exec_time, but `gen_trace_attacc_bank.py Attention()` already includes softmax cycles inside the score MATMUL trace.
7. **Chunked prefill** — original prefill ran fully on GPU. Sampled extrapolation for PIM prefill with vLLM-style chunks needed.

This fork closes all of the above without breaking upstream behavior on legacy text-only models (GPT-175B, LLAMA-65B, etc.).

---

## Modifications at a glance

| ID | What changed | Files |
|---|---|---|
| M0 | Repo metadata cleanup (DeepStack `[5,11,17]` consistent everywhere) | README, scripts |
| M1 | CLI: `--num_attacc`, `--num_hbm`, `--interface NVLINK4`, `--tp`, `--max_L`, `--routing`, `--pim_layers`, `--prefill_chunk`, `--prefill_samples`, `--image_size` (+ width/height) | `main.py` |
| M2a | `Layer.pim_numOp` field; `Transformer` GQA fields (`num_q_heads`, `num_kv_heads`, `qkv_proj_out_total`, `q_proj_out`, `kv_proj_out`, `ff_intermediate`, `ffn_type`, `activation`); `fc_tp` / `attn_tp` / `ff_tp` split | `src/model.py` |
| M2b | KV capacity formula uses `num_kv_heads × dhead` (not `hdim`); per-GPU split by `attn_tp`; weight per-GPU by `fc_tp` | `src/system.py` |
| M2c | `_pipeline` minimum_ratio uses `num_kv_heads` (single division); `comm_x2g*` matched by prefix | `src/system.py` |
| M3 | Modern model configs (5 VLMs + 3 LLM backbones); legacy list-format preserved with auto-derived GQA fields | `src/config.py` |
| M4 | ViT graph (`_build_vit`): qkv / score / softmax / context / proj / 2× norm / ff1 / act / ff2, single-template × `vit_layers` `numOp` scaling | `src/model.py` |
| M5 | Projector graph (`_build_projector`): `mlp` / `mlp_with_merger` / `pixel_shuffle_mlp`; optional `vit_broadcast` G2G when `fc_tp > 1` | `src/model.py` |
| M6.1 | Sum-stage PIM dispatch (`_select_sum_device`): MATMUL / SOFTMAX / X2G route to accelerator when `group_device != 'gpu'` | `src/system.py` |
| M6.3 | `_simulate_pim_prefill_score`: chunked prefill via sampled extrapolation; sub-layer with `m=1`, `n=accumulated_L`, `k=dhead`, `numOp = Q heads`, `pim_numOp = KV heads`; chunk count multiplied externally | `src/system.py` |
| M6.4 | `System.get_pipelining_efficiency_latency()`: §6.1 caveat as latency-mode; applied to score (and 0-time softmax/context placeholders) before `_pipeline()` | `src/system.py` |
| M7-pre | Routing groups: `sum_decoder_groups` / `gen_decoder_groups`, `routing_meta`, `(group_name, device, count, indices)` tuples; uniform `t × ndec` scaling removed | `src/model.py`, `src/system.py` |
| M7 | `Routing` class: `conservative` / `optimistic` / `list`; DeepStack auto-forces list mode | `src/model.py` |
| M8 | `get_capacity_breakdown()` per-GPU dict accessor (kept tuple `get_required_mem_capacity` for `sum()` back-compat) | `src/system.py` |
| M9 | DeepStack `deepstack_add` (LayerType.ACT) injected into sum_decoder when `layer_idx in deepstack_layers`; gen path untouched | `src/model.py` |
| M12 | `select_best_resolution()` and `compute_visual_tokens()` for LLaVA-Next anyres (672×672 → 2928 tokens, plan target 2880 ±10%) | `src/model.py` |
| M13 | Ramulator `max_L` propagation: constructor arg, trace_args `--maxlen`, cache key, file_name; legacy stale-cache invalidated automatically | `src/ramulator_wrapper.py`, `src/system.py` |
| M14 | NVLink4 via `--interface NVLINK4`; existing G2G time model reused (no fixed-formula override); `comm_g2g` cost falls naturally to 0 at TP=1 | `src/config.py` |
| P1 | PIM softmax returns `(0, [0,...])` to avoid double-counting (score's Ramulator trace already includes softmax cycles) | `src/devices.py` |

### Concern fixes from review rounds

- `required_cap` CSV column: per-GPU semantics (was system-total).
- `temp_memory` formula: missing `* a_byte` on trailing term restored.
- Ramulator wrapper: `len(df.columns) > 12 → pdb.set_trace()` trap removed.
- `num_xpu` double-divide in `minimum_ratio` removed.
- `--word default='2'` (string) → `2` (int).
- `pandas` made optional at import time, asserted at execution.
- CSV `with open(..., newline='')` for Windows compatibility.
- `os.system("rm output.csv")` → `os.remove(output_path)`.

---

## Supported models

### Legacy text-only (LLM list format, paper repro)

`GPT-175B` (R2 paper-repro target — 4.84× / 2.48×), `GPT-89B`, `GPT-13B` (smoke baseline), `LLAMA-7B`, `LLAMA-65B` (Fig.13), `MT-76B/146B/310B/530B/1008B`, `OPT-66B`.

### Modern text-only (LLM dict format, R1 sanity / VLM backbone)

| Model | ndec / hdim / nq / nkv / dhead / ff_intermediate | Used as |
|---|---|---|
| `Qwen3-4B` | 36 / 2560 / 32 / 8 / 128 / 9728 | Qwen3-VL backbone |
| `Vicuna-7B` | 32 / 4096 / 32 / 32 / 128 / 11008 | LLaVA-1.5 backbone (MHA) |
| `Mistral-7B` | 32 / 4096 / 32 / 8 / 128 / 14336 | LLaVA-Next backbone |

### VLM in-framework (R3 / R7 targets)

| Model | LLM | n_q / n_kv | ViT | Projector | DeepStack | AnyRes |
|---|---|---|---|---|---|---|
| `Qwen3-VL-4B` (Primary) | Qwen3 36L 2560 | 32 / 8 | 24L 1024 | mlp_with_merger | **[5,11,17]** | × |
| `Qwen2.5-VL-7B` | Qwen2.5 28L 3584 | 28 / 4 | 32L 1280 | mlp_with_merger | × | × |
| `InternVL3-8B-hf` | Qwen2.5-spec 28L | 28 / 4 | 24L 1024 | pixel_shuffle_mlp | × | × |
| `LLaVA-1.5-7B` | Vicuna 32L | 32 / 32 (MHA) | 24L 1024 | mlp | × | × |
| `LLaVA-Next-Mistral-7B` | Mistral 32L | 32 / 8 | 24L 1024 | mlp | × | **5 grids** |

### Out-of-scope (paper observation only)

`Llama-3.2-V` (cross-attention architecture), `Qwen3.5` (hybrid linear attention) — these architectures need framework changes beyond this fork's scope.

---

## Deployment scenarios

| Scenario | GPUs | TP | NUM_ATTACC | NUM_HBM | Inter-GPU | PIM aggregate |
|---|---|---|---|---|---|---|
| **S1: H100 × 1** | 1 | 1 | 1 | 5 | none | 18.1 TB/s |
| **S2: H100 × 2** | 2 | 2 | 2 | 5 | NVLink4 900 GB/s (HD 450) | 36.2 TB/s |
| Paper repro (DGX-A100) | 8 | 8 | 8 | 5 | NVLink3 600 GB/s | 145 TB/s |

Constraint enforced in `main.py`:
```
assert num_attacc == tp == ngpu  # 1 AttAcc per GPU
```

### Pipelining latency-mode efficiency (`eff_lat`, paper §6.1)

Per-GPU `n_kv_heads` distributed over `num_hbm = 5`. If any HBM ends up with 1 head, GEMV / softmax cannot pipeline (1.4× slowdown).

| Model | n_kv | S1 eff_lat (TP=1) | S2 eff_lat (TP=2) |
|---|---|---|---|
| Qwen3-VL-4B | 8 | 0.80 | 0.57 |
| Qwen2.5-VL-7B | 4 | 0.57 | **0.29** |
| InternVL3-8B-hf | 4 | 0.57 | **0.29** |
| LLaVA-1.5-7B (MHA) | 32 | 0.91 | 0.80 |
| LLaVA-Next-Mistral | 8 | 0.80 | 0.57 |

`batch ≥ 2` returns `1.0` (heads ample, plan §0.4 simplification).

---

## Quick start

### Local smoke (no Ramulator binary, GPU baseline only)

```bash
python main.py --gpu A100a --ngpu 8 --tp 8 --num_attacc 8 \
  --model GPT-13B --lin 16 --lout 2 --batch 1
# Expected: latency ~5.51 ms (legacy path preserved)

python main.py --gpu H100 --ngpu 1 --tp 1 --num_attacc 1 --num_hbm 5 --interface NVLINK4 \
  --model Qwen3-VL-4B --lin 569 --lout 128 --batch 1 --image_size 672 --max_L 2048
```

### dgx-attacc S1 (H100 × 1, TP=1, AttAcc=1)

```bash
python main.py --system dgx-attacc --gpu H100 --ngpu 1 --tp 1 \
  --num_attacc 1 --num_hbm 5 --interface NVLINK4 \
  --model Qwen3-VL-4B --lin 569 --lout 128 --batch 1 \
  --image_size 672 --prefill_chunk 512 --prefill_samples 8 --max_L 2048
```

### dgx-attacc S2 (H100 × 2, TP=2, AttAcc=2)

```bash
python main.py --system dgx-attacc --gpu H100 --ngpu 2 --tp 2 \
  --num_attacc 2 --num_hbm 5 --interface NVLINK4 \
  --model Qwen3-VL-4B --lin 569 --lout 128 --batch 1 \
  --image_size 672 --prefill_chunk 512 --prefill_samples 8 --max_L 2048
```

### R2 paper repro (DGX-A100, GPT-175B)

```bash
python main.py --system dgx-attacc --gpu A100a --ngpu 8 --tp 8 \
  --num_attacc 8 --num_hbm 5 --interface NVLINK3 \
  --model GPT-175B --lin 2048 --lout 128 --batch 64
# Target: vs DGX_Base = 4.84× ± 20% (must), vs DGX_Large = 2.48× ± 20% (should)
```

### Routing modes

```bash
--routing default        # single 'all' group (default behavior, no Routing class)
--routing conservative   # all decoder layers on accelerator (DeepStack auto-forced to list)
--routing optimistic     # count-compressed: (acc, count) + (gpu, ndec - count)
--routing list           # per-layer groups; required for DeepStack injection at specific indices
--pim_layers 0,8,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,33  # custom layer set
```

DeepStack model + `dgx-attacc` without `--routing` triggers an auto-warning and forces list mode internally.

---

## Tests

```bash
python tests/r1_sanity.py            # KV cache TP1 80.02 / TP2 40.01 MiB for Qwen3-VL-4B at L=569
python tests/m6_4_eff_lat.py         # Reproduces §0.4 eff_lat table (7 cases)
python tests/vlm_graph_sanity.py     # Vision graph + DeepStack inject + AnyRes token count
python tests/m6_1_prefill_fake.py    # Chunked sampled prefill contract via FakePIM (no Ramulator)
python tests/m14_nvlink.py           # G2G time at TP=1 / TP=2 NVLink4 / TP=2 NVLink3
```

All five tests pass without `pandas` installed and without the Ramulator binary (PIM-bound checks use a `FakePIM` stub).

---

## Layer shape contract

For `score` MATMUL (and the placeholder `softmax` / `context` layers in the same attention block):

| Field | Meaning | Used by |
|---|---|---|
| `layer.m` | query count (chunked prefill: 1, decode: 1, full prefill: L) | NOT used by Ramulator wrapper — multiplied externally |
| `layer.n` | accumulated KV length | `ramulator_wrapper.run()` as `l` |
| `layer.k` | `dhead` | `ramulator_wrapper.run()` as `dhead` |
| `layer.numOp` | Q heads per AttAcc (GPU compute / FLOPs) | `Layer.get_flops()`, GPU device |
| `layer.pim_numOp` | KV heads per AttAcc (PIM trace, KV-read dominated) | `ramulator_wrapper.run()` as `num_ops_per_attacc`, divided by `num_hbm` internally |

**Score MATMUL Ramulator trace already includes softmax + context cycles** (see `pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py` `Attention()`). Therefore PIM softmax and PIM context return `(0, [0,...])` by design. Adding non-zero values would double-count the same hardware activity.

---

## Limitations / known gaps

- `dgx-attacc` execution requires both `pandas` and the Ramulator2 binary. Without them, only the GPU baseline (`dgx`) and CPU offload (`dgx-cpu`) paths run end-to-end.
- Decode-stage `comm_x2g_qkv` is a single combined Q+K+V transfer; intra-attention Q/K/V split is left as a future refinement (does not affect the prefill chunked path or the R3 corrected-E2 ratio significantly).
- `optimistic` and `list` routing produce identical per-layer cost when the selected layer set is the same. They diverge once layer-specific graphs (DeepStack, future ViT-stage projector injection) are present.
- `batch ≥ 2 → eff_lat = 1.0` is a plan-spec simplification. A more accurate batch-aware distribution (heads = `n_kv × batch`) is left as future refinement.
- `attn_tp != fc_tp` (occurs only when `ngpu > num_kv_heads`, e.g. Qwen2.5-VL-7B with `--tp 8`) emits a WARNING and uses an analytical fallback for the KV-repartition cost.
- Multi-image VLM serving is not modeled; single image per request assumed.
- `Llama-3.2-V` cross-attention and `Qwen3.5` hybrid linear attention architectures are out-of-framework.

---

## Citation

If you use this fork in academic work, please cite the original AttAcc paper:

```bibtex
@inproceedings{park2024attacc,
  title={AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference},
  author={Park, Jaehyun and Choi, Jaewan and Kyung, Kwanhee and Kim, Michael Jaemin and Kwon, Yongsuk and Kim, Nam Sung and Ahn, Jung Ho},
  booktitle={Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
  pages={103--119},
  year={2024}
}
```

Upstream: [scale-snu/attacc_simulator](https://github.com/scale-snu/attacc_simulator)

---

# Upstream README (preserved for reference)

The original AttAcc simulator README, including build instructions for Ramulator2, Python prerequisites, and the original AttAcc command-trace generation flow, follows below.

## Simulator for AttAcc
This repository includes a Python-based simulator designed to analyze transformer-based generation model (TbGM) inference in a heterogeneous system consisting of an xPU and an Attention Accelerator (AttAcc). AttAcc is an accelerator for the attention layer of TbGM, consisting of an HBM-based processing-in-memory (PIM) structure. In simulating an xPU and AttAcc system, the simulator outputs the performance and energy usage of the xPU, while the behavior of AttAcc is simulated using a properly modified [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2). The memory device of AttAcc in Ramulator2 is HBM3, and AttAcc\_bank, AttAcc\_BG, and AttAcc\_buffer represent AttAcc deploying processing units per bank, per bank group, or per pseudo-channel (on the buffer die), respectively.

## Prerequisites
- Python 3.8+
- `pandas` (required for `dgx-attacc` mode; optional otherwise)
- cmake, g++, clang++ (for building Ramulator2)

The original simulator was tested under Ubuntu 22.04.3 LTS, g++ 12.3.0, Python 3.8.8.

## How to install (full Ramulator pipeline)
```bash
git clone <this-fork-url>
cd attacc_simulator
git submodule update --init --recursive
bash set_pim_ramulator.sh
cd ramulator2
mkdir build && cd build
cmake ..
make -j
cp ramulator2 ../ramulator2
cd ../../
```

## Original CLI (still supported)
```bash
python main.py --system dgx --gpu A100a --ngpu 8 \
  --model GPT-175B --lin 2048 --lout 128 --batch 1

python main.py --system dgx-attacc --gpu A100a --ngpu 8 \
  --model GPT-175B --lin 2048 --lout 128 --batch 1 \
  --pim bank --powerlimit --ffopt --pipeopt
```

The new flags introduced in this fork (`--num_attacc`, `--num_hbm`, `--interface`, `--tp`, `--max_L`, `--routing`, `--pim_layers`, `--prefill_chunk`, `--prefill_samples`, `--image_size`) all default to upstream-compatible values, so legacy commands continue to produce upstream-equivalent results on GPT-175B and similar MHA models.

## Ramulator command-trace generation
See `pim_ramulator_src/trace_gen/gen_trace_attacc_{bank,bg,buffer}.py` and `ramulator2/` configurations for upstream details. The trace generator now accepts `--maxlen` (already present upstream) which is propagated by `src/ramulator_wrapper.py` based on the `--max_L` CLI argument.

## License

This fork retains the upstream license. See `LICENSE` for details.

## Contact

Original AttAcc authors:

- Jaehyun Park — jhpark@scale.snu.ac.kr
- Jaewan Choi — jwchoi@scale.snu.ac.kr

Fork-specific issues: please open a GitHub issue on this fork's repository.
