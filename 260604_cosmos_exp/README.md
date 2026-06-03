# 260604 Cosmos 3 + PIM measurement runbook

Owner: Minsik · last updated 2026-06-04

Implements the experiment plan in `../260604_cosmos_experiment.md`.
Today's scope: **(1) theoretical PIM gain calculation + (2) empirical
H100 anchor measurements** across the omni-modal task matrix.  The
simulator extension (DM denoising loop, MoT layer, tier policy, event
queue) is deferred.

## Directory layout

```
260604_cosmos_exp/
├── README.md                          (this file)
├── shared/                            common utilities
│   ├── hw_detect.py                   nvidia-smi -> host tag, peak BW
│   ├── result_aggregator.py           save / load <name>.json + per-host copy
│   └── cosmos_facts.py                Cosmos 3 Nano/Super arch constants
├── measurement/
│   ├── e2e_latency_matrix.py          Phase 1.1 driver (omni-task matrix)
│   ├── vram_profile.py                Phase 1.2
│   ├── nsys_bandwidth.py              Phase 1.4 (Nsight Systems wrapper)
│   ├── resolution_scaling.py          Phase 1.6
│   ├── frame_scaling.py               Phase 1.7
│   ├── tokens_per_modality.py         Phase 1.8
│   ├── denoise_step_sweep.py          Phase 3.1
│   ├── denoise_step_traffic.py        Phase 3.2 (Nsight wrapper)
│   ├── guidance_sweep.py              Phase 3.6
│   ├── streaming_arrival.py           Phase 4.1
│   ├── modality_slo_budget.py         Phase 4.2
│   ├── modality_interference.py       Phase 4.3
│   ├── vllm_omni/
│   │   └── runner.py                  vLLM-Omni backend for matrix driver
│   └── pytorch/
│       ├── runner.py                  PyTorch+Diffusers backend
│       ├── phase_breakdown.py         Phase 1.3
│       ├── ar_vs_dm.py                Phase 1.5
│       ├── attention_pattern.py       Phase 2.1
│       └── kv_temporal_locality.py    Phase 2.2
├── analysis/
│   ├── nvidia_repro.py                Phase 0.2 (depends on 1.1)
│   ├── theoretical_pim_gain.py        Phase 3.5 (paper hook anchor)
│   ├── modality_kv_breakdown.py       Phase 2.5
│   └── per_modality_slo.py            Phase 4.5
└── results/                           output JSONs (per-host suffix kept)
```

## Run order on H100 node

```bash
cd attacc_simulator/260604_cosmos_exp

# Phase 0 -- environment + NVIDIA benchmark sanity
python measurement/env_setup.py
# (after 1.1 below)
python analysis/nvidia_repro.py

# Phase 1 -- workload characterization
python measurement/e2e_latency_matrix.py            # ~12 H100-hours
python measurement/vram_profile.py
python measurement/pytorch/phase_breakdown.py
python measurement/nsys_bandwidth.py
python measurement/pytorch/ar_vs_dm.py
python measurement/resolution_scaling.py
python measurement/frame_scaling.py
python measurement/tokens_per_modality.py

# Phase 2 -- Topic A anchors
python measurement/pytorch/attention_pattern.py
python measurement/pytorch/kv_temporal_locality.py
python analysis/modality_kv_breakdown.py

# Phase 3 -- Topic B anchors
python measurement/denoise_step_sweep.py
python measurement/denoise_step_traffic.py
python measurement/guidance_sweep.py
python analysis/theoretical_pim_gain.py            # <-- paper hook

# Phase 4 -- Topic C anchors
python measurement/streaming_arrival.py            # gates topic C
python measurement/modality_slo_budget.py
python measurement/modality_interference.py
python analysis/per_modality_slo.py
```

Each script:
- detects host via `shared/hw_detect.py`
- writes `results/<name>.json` and `results/<name>_<host>.json`
- tolerates framework-missing / task-unsupported failures so the matrix
  records gaps instead of crashing

## 2×H100 split (per 2026-06-04 confirm)

- **#1** runs the e2e_latency_matrix.py / topic-specific measurements
- **#2** runs `analysis/nvidia_repro.py` after 1.1 + a parallel Super
  (TP=2) E2E sanity that pins down NVIDIA benchmark numbers
- Super × TP=1 is *not* expected to fit; the matrix driver auto-skips

## Task matrix axes (per Phase 1.1)

| axis | values |
|---|---|
| model | Cosmos3-Nano, Cosmos3-Super |
| tp | 1, 2 |
| batch | 1, 2, 4, 8 |
| framework | pytorch, vllm_omni |
| task | t2v, t2i, i2v, v2v, t2a, multi2v, multi2action |
| resolution | 720p (1.6 sweeps 256p/480p/720p) |
| frames | 189 (1.7 sweeps 24/96/189/300; video-output only) |

## Key paper hooks (theoretical anchors)

- `analysis/theoretical_pim_gain.py` produces two distinct numbers:
  - `upper_bound_gain_x = ATTACC_BW / H100_BW = 72.2x`
    -- KV-only ceiling, NOT realized
  - `realized_gain_x = (weight + KV) / max(weight/H100_BW, KV/ATTACC_BW)`
    -- the honest paper anchor.  At Nano @256K, b=8 this is **10.66x**
    (kv_dominance 90.6%).  At Nano @ b=1 it is only 2.2x.
- Two-tier output in `cosmos_theoretical_pim_gain.json`:
  - `rows[]`        = pure analytic grid (model x ctx x batch x steps),
                      no measured override
  - `measured_anchors[]` = one row per Phase 3.2 status=='measured' cell,
                      each at its own (model, batch, resolution, frames,
                      denoise_steps, engine_tag, actual_context_tokens)
- `analysis/modality_kv_breakdown.py` shows which modality dominates
  the KV cache budget per scenario -- input for Topic A's tier policy
- `analysis/per_modality_slo.py` formalizes per-modality SLO targets
  if Phase 4.1/4.3 validate concurrent streaming
- `analysis/nvidia_repro.py` compares Phase 1.1 rows against the 20
  verbatim cells in `NVIDIA_BENCHMARK` (H100 NVL / H100_80GB_HBM3_SXM /
  H200_141GB_HBM3).  Use `COSMOS_NVIDIA_KEY=H100_NVL` env to disambiguate
  the H100 variant.  Diffusers 256p comparisons are flagged with
  `resolution_mismatch` caveat (NVIDIA uses 320x192 internally; we run
  at 448x256).

## Pre-flight checklist on H100 node

1. `python measurement/env_setup.py` -- confirms vLLM-Omni / Diffusers /
   Cosmos weights all present; if any READY check fails, fix that
   before running the matrix.  Repo IDs are the official
   `nvidia/Cosmos3-{Nano,Super}` (no dash before 3).
2. Pipeline class is `Cosmos3OmniPipeline` (single class for t2v / t2i /
   i2v; differentiated by `num_frames` / `image=` / `enable_sound=True`).
   Scheduler is `UniPCMultistepScheduler.from_config(..., flow_shift=10.0)`.
   Official quality-control negative prompts are in
   `shared/cosmos_facts.NEGATIVE_PROMPT_T2V` / `_I2V`.
3. Set `COSMOS_NVIDIA_KEY=H100_NVL` or `H100_80GB_HBM3_SXM` before
   running Phase 0.2 sanity so it matches the right NVIDIA benchmark
   column.
4. `nvidia-smi` must be on PATH so `hw_detect.detect_host()` picks
   "H100"; otherwise everything falls back to "A6000" tags.

## Known limitations / TBD

- `analysis/theoretical_pim_gain.py`: the 72x ceiling is KV-only and
  unrealistic for batch=1 workloads.  Useful paper number is the
  realized gain at the configured (model, ctx, batch).  At Nano @ b=1
  it is only 2.2x (weight-bound); at Nano @ b=8 ctx=256K it is 10.66x
  (90.6% KV dominance).  The story Topic B can tell is therefore a
  "long context + batch >= 8" regime, not a flat 72x claim.
- `measurement/denoise_step_traffic.py` (Phase 3.2): the nsys
  `cudaProfilerStart/Stop` capture EXCLUDES model load + warmup but
  INCLUDES text encode + denoise loop + VAE decode within one rep.
  `per_step = total / num_inference_steps` therefore amortizes those
  one-time costs.  `kv_per_step_derived = per_step - weight_bytes`
  assumes weights are read once per denoise step at H100 BW.
  `TODO_NVTX_FILTER`: a stricter per-step measurement requires
  NVTX-range-bounded gpummu / dramread (nsys-version dependent) -- the
  runner already pushes `cosmos:step_{n}` ranges; CSV-side filtering
  by those ranges is not yet implemented.
- CSV parser is STRICT: it accepts only (a) headers explicitly
  containing 'read' + a byte unit AND not write/written/store, OR
  (b) an Operation/Op/Kind/Type column whose rows are 'read'/'load'.
  No 'non-write byte column' fallback.  If a given nsys version uses
  a header style outside these two patterns, the parser returns
  `status='analytic_only'` and Phase 3.5 will ignore that row.
- `analysis/modality_kv_breakdown.py` uses placeholder audio (50 Hz) /
  action (30 Hz) token rates.  Phase 1.8 (`tokens_per_modality.py`)
  must run first for empirically anchored rates.
- `measurement/pytorch/attention_pattern.py` + `kv_temporal_locality.py`
  install a `CapturingProcessor` on the first `Attention` module in
  `pipe.transformer`.  If the Cosmos3 release does not expose
  `attn.to_q/to_k/to_v/to_out` in the standard diffusers way, these
  scripts need adapting on the H100 host.
- `measurement/streaming_arrival.py` requires vLLM-Omni's actual
  `OmniSamplingParams` accepting `modalities=["text"|"image"|"audio"]`.
  If the API differs, edit the param construction.

## Adapting to other hosts

The whole tree is HW-aware: scripts run as-is on any host that has the
right framework versions.  Output JSONs get per-host suffixes so an
A6000 syntax-check run does not clobber the H100 measurement.
