# AttAcc Simulator: Structure + Modification Map

Upstream `scale-snu/attacc_simulator` 위에 VLM-aware 확장이 어디에 들어갔는지
한눈에 보여주는 문서.

Legend:
```
[U]     UPSTREAM        upstream에서 그대로 사용
[M:Mx]  MODIFIED        M-id에 해당하는 자리에서 수정
[N:Mx]  NEW FILE        새로 추가된 파일 (M-id 또는 카테고리)
[F]     FORK ARTIFACT   본 fork에서 추가된 paper-grade 보조 파일
```

---

## 1. Top-Level Architecture

```
                     ┌──────────────────────────────────────┐
                     │           main.py [M:M1]             │
                     │   argparse (CLI), system bring-up    │
                     └──────────────┬───────────────────────┘
                                    │
                ┌───────────────────┼───────────────────────┐
                │                   │                       │
                ▼                   ▼                       ▼
        ┌──────────────┐    ┌──────────────┐        ┌──────────────┐
        │  config.py   │    │  system.py   │        │  devices.py  │
        │ [M:M2/M3]    │    │ [M:M2b/M2c   │        │ [M:P1]       │
        │              │    │  M6.1/M6.3   │        │              │
        │ - VLM dicts  │    │  M6.4/M7-pre │        │ - GPU model  │
        │ - PIM cfg    │    │  M7/M8/M14]  │        │ - PIM model  │
        └──────┬───────┘    └──────┬───────┘        │ - CPU offload│
               │                   │                └──────┬───────┘
               │                   │ build / route         │
               │                   ▼                       │
               │           ┌──────────────┐                │
               │           │  model.py    │                │
               │           │ [M:M2a/M4    │                │
               │           │  M5/M9/M12]  │                │
               │           │              │                │
               │           │ - Layer      │                │
               │           │ - Transformer│                │
               │           │ - Routing    │                │
               │           └──────┬───────┘                │
               │                  │ Layer list             │
               │                  ▼                        │
               │           per-layer dispatch              │
               │                  │                        │
               │     ┌────────────┴────────────┐           │
               │     │                         │           │
               │     ▼                         ▼           │
               │ GPU path               PIM path           │
               │     │                         │           │
               │     │ FC/NORM/G2G/X2G         │ MATMUL    │
               │     │ (FlashAttn-2 etc)       │ /SOFTMAX  │
               │     └──────────┬──────────────┘           │
               │                │                          │
               │                ▼                          │
               │       ┌────────────────────┐              │
               │       │ ramulator_wrapper  │              │
               │       │       .py          │              │
               │       │ [M:M13]            │              │
               │       │                    │              │
               │       │ - trace_gen disp.  │              │
               │       │ - max_L cache key  │              │
               │       │ - pim_numOp/n_kv   │              │
               │       └─────────┬──────────┘              │
               │                 │                         │
               │                 ▼                         │
               │   ┌─────────────────────────┐             │
               │   │  ramulator2 binary [U]  │             │
               │   │   + trace_gen/*.py [U]  │             │
               │   └─────────────────────────┘             │
               │                                           │
               └─────────► output.csv [M:M2b s_x2g 추가] ◄─┘
```

---

## 2. File-by-file Annotation Tree

```
attacc_simulator/                        # repo root
│
├── main.py                              [M:M1] argparse 확장
│   ├── --num_attacc / --num_hbm         [N] CLI flag 추가
│   ├── --interface NVLINK4              [N] interface 선택
│   ├── --tp                             [N] tensor parallel
│   ├── --max_L                          [N] Ramulator trace 길이 한계
│   ├── --routing / --pim_layers         [N] routing mode + layer list
│   ├── --prefill_chunk / --prefill_samples  [N] chunked prefill 제어
│   ├── --image_size / width / height    [N] VLM image input
│   └── assert num_attacc==tp==ngpu      [N] 1 AttAcc/GPU 강제
│
├── src/
│   ├── type.py                          [U] enum: DataType, LayerType, …
│   │
│   ├── config.py                        [M:M3]
│   │   ├── make_xpu_config              [M:M1] H100 + NVLink4
│   │   ├── make_pim_config              [M:M1] num_attacc/num_hbm 인자
│   │   └── make_model_config
│   │       ├── legacy list dict 변환    [M:M3]
│   │       ├── Qwen3-4B                 [N:M3] 신규 entry
│   │       ├── Qwen3-VL-4B              [N:M3] DeepStack [5,11,17]
│   │       ├── Qwen2.5-VL-7B            [N:M3]
│   │       ├── InternVL3-8B-hf          [N:M3]
│   │       ├── Vicuna-7B                [N:M3]
│   │       ├── LLaVA-1.5-7B             [N:M3]
│   │       ├── Mistral-7B               [N:M3]
│   │       └── LLaVA-Next-Mistral-7B    [N:M3] AnyRes 5 grids
│   │
│   ├── model.py                         [M:M2a]
│   │   ├── class Layer
│   │   │   ├── numOp (= num_q_heads)    [M:M2a]
│   │   │   └── pim_numOp (= num_kv)     [N:M2a] PIM trace 분리
│   │   │
│   │   ├── class Routing                [N:M7]
│   │   │   ├── conservative / optimistic / list
│   │   │   └── DeepStack auto list mode
│   │   │
│   │   └── class Transformer
│   │       ├── num_q_heads / num_kv     [M:M2a] GQA contract
│   │       ├── q_proj_out / kv_proj_out [M:M2a]
│   │       ├── fc_tp / attn_tp / ff_tp  [M:M2a] TP split
│   │       ├── ff_intermediate          [M:M2a] gated FFN
│   │       ├── _build_vit               [N:M4] ViT graph
│   │       ├── _build_projector         [N:M5] mlp / merger / pixshuf
│   │       ├── compute_visual_tokens    [N:M12] AnyRes best-fit
│   │       ├── _build_sum_one_layer
│   │       │   ├── comm_x2g_kv          [N:M2a] K+V forward
│   │       │   ├── comm_x2g_q           [N:M2a] Q forward
│   │       │   ├── deepstack_add        [N:M9] sum-only inject
│   │       │   └── comm_x2g_return      [N:M2a] return path
│   │       └── build (routing groups)   [M:M7-pre] per-layer dispatch
│   │
│   ├── system.py                        [M:M2b / M2c / M6 / M7 / M14]
│   │   ├── System.__init__
│   │   │   └── max_L                    [N:M13]
│   │   │
│   │   ├── set_routing                  [N:M7]
│   │   ├── set_image_size               [N:M4]
│   │   ├── set_prefill_config           [N:M6.3]
│   │   ├── get_pipelining_efficiency_latency  [N:M6.4] paper sec.6.1
│   │   │
│   │   ├── _pipeline                    [M:M2c] minimum_ratio (num_kv 기반)
│   │   │                                + comm_x2g_* prefix match
│   │   ├── _select_sum_device           [N:M6.1] prefill PIM dispatch
│   │   ├── _select_gen_device           [M:M6.1] 기존 gen dispatch 정리
│   │   ├── _simulate_pim_prefill_score  [N:M6.3] chunked sampled
│   │   ├── _apply_eff_lat               [N:M6.4] sec.6.1 caveat 적용
│   │   ├── simulate
│   │   │   ├── routing groups loop      [M:M7-pre] ndec uniform scale 제거
│   │   │   ├── per-layer s_perf / g_perf [M] x2g 케이스 추가
│   │   │   └── s_x2g 컬럼 출력          [M:M2b]
│   │   │
│   │   ├── get_required_mem_capacity    [M:M2b] num_kv 기반 KV 계산
│   │   └── get_capacity_breakdown       [N:M8] per-GPU dict
│   │
│   ├── devices.py                       [M:P1]
│   │   ├── class xPU                    [U]
│   │   ├── class PIM
│   │   │   ├── score MATMUL → ramulator [U]
│   │   │   ├── context MATMUL → (0,..)  [U] graph order placeholder
│   │   │   └── SOFTMAX → (0,..)         [M:P1] double-count 차단
│   │   └── ...
│   │
│   └── ramulator_wrapper.py             [M:M13]
│       ├── Ramulator(num_hbm, max_L)    [N:M13] kwargs 확장
│       ├── trace_gen path auto-search   [N:M13] ramulator2/ + pim_ramulator_src/
│       ├── --maxlen pass-through        [N:M13]
│       ├── cache key (L, max_L, …)      [M:M13] max_L 포함
│       ├── num_ops = pim_numOp          [M:M2a] GQA KV-head
│       └── pandas optional              [M:M13]
│
├── pim_ramulator_src/
│   └── trace_gen/                       [U] AttAcc paper trace generator
│       └── gen_trace_attacc_{bank,bg,buffer}.py
│
├── ramulator2/                          [U] HBM3-PIM cycle-accurate sim
│
├── tests/                               [N:F] fork artifacts
│   ├── r1_sanity.py                     [N:F] KV cache size
│   ├── m6_4_eff_lat.py                  [N:F] paper sec.6.1 table
│   ├── vlm_graph_sanity.py              [N:F] ViT / DeepStack / AnyRes
│   ├── m6_1_prefill_fake.py             [N:F] FakePIM chunked prefill
│   ├── m14_nvlink.py                    [N:F] NVLink G2G
│   ├── r2_paper_repro.py                [N:F] GPT-175B paper repro
│   ├── r3_gate.py                       [N:F] Qwen3-VL S1/S2 corrected E2
│   ├── r6_h100_measurement.py           [N:F] HF accelerate (deprecated)
│   ├── r6_vllm_measurement.py           [N:F] vLLM 0.7.3 paper-grade
│   ├── r7_paper_runner.sh               [N:F] multi-VLM driver
│   ├── r7_multi_vlm_runner.sh           [N:F] same
│   ├── r9_mmmu_pro_measurement.py       [N:F] real MMMU-Pro + power
│   ├── r10_concurrent_serving.py        [N:F] Poisson 4 qps
│   ├── calibrate_scaling.py             [N:F] 42-combo grid search
│   └── sim_correction_factor.py         [N:F] s_corr / g_corr
│
├── results/                             [N:F] r6..r10 + MMMU-Pro JSON
│   └── README.md
│
├── docs/                                [N:F]
│   ├── attacc_simulator_plan_final_v1.md
│   └── attacc_simulator_patch_implementation_report.md
│
└── 260511_additional_exp/               [N:F] paper-grade campaign
    ├── README.md
    ├── run_all_h100x1.sh                [N:F] master driver
    │
    ├── shared/
    │   ├── sim_runner.py                [N:F] main.py subprocess wrapper
    │   │                                       + e2e_ms helper
    │   │                                       + strict mode
    │   │                                       + g_time alias
    │   ├── result_aggregator.py         [N:F] standard JSON + percentile
    │   ├── plot_helpers.py              [N:F] matplotlib helpers
    │   └── vllm_helpers.py              [N:F] per-model chat template
    │
    ├── tier1_simulator/                 # foundation gates
    │   ├── r2_paper_repro.py            [N:F] GPT-175B 4.84x gate
    │   ├── upstream_baseline.py         [N:F] legacy LLM regression
    │   ├── multi_vlm_full_sim.py        [N:F] 5 VLM x dgx/dgx-attacc
    │   ├── ablation_contribution.py     [N:F] M-component decomposition
    │   └── vit_recalibration.py         [N:F] measured TTFT calib
    │
    ├── tier2_simulator/                 # sensitivity + architecture
    │   ├── chunk_size_sweep.py          [N:F]
    │   ├── routing_mode_compare.py      [N:F]
    │   ├── eff_lat_ablation.py          [N:F]
    │   ├── sensitivity_sweep.py         [N:F] 240-cfg grid
    │   ├── nvlink_compare.py            [N:F]
    │   ├── roofline_per_vlm.py          [N:F] AI vs H100 ridge
    │   ├── slo_throughput.py            [N:F]
    │   ├── capacity_regime.py           [N:F]
    │   ├── pim_mode_compare.py          [N:F] bank/bg/buffer
    │   └── w4a16_pim_sim.py             [N:F] quant + PIM projection
    │
    ├── tier2_measurement/               # real H100 + vLLM 0.7.3
    │   ├── w4a16_awq_measure.py         [N:F]
    │   ├── w8a16_gptq_measure.py        [N:F]
    │   ├── quant_stability_test.py      [N:F] BF16/FP16/W4/W8 NaN
    │   ├── image_size_sweep.py          [N:F]
    │   ├── prompt_pattern_matrix.py     [N:F]
    │   └── vllm_bf16_baseline_aligned.py [N:F] multi_vlm-aligned
    │
    └── results/                         [N:F] experiment JSONs land here
```

---

## 3. Modification Catalog (M0 ~ M14 + P1)

| ID | 한줄 설명 | 주요 파일 | 카테고리 |
|---|---|---|---|
| M0 | repo metadata cleanup (DeepStack [5,11,17] 통일) | README, scripts | Tooling |
| M1 | CLI 확장 + 1 AttAcc/GPU assert | `main.py`, `config.py` | Deployment |
| M2a | `Layer.pim_numOp` 분리 / GQA fc_tp · attn_tp / Q forward / return X2G / FFN 정확화 / Layer shape contract | `model.py` | GQA contract |
| M2b | KV capacity 식 (n_kv × dhead) / per-GPU split / `s_x2g` 컬럼 | `system.py` | GQA contract |
| M2c | `_pipeline` minimum_ratio num_kv 기반 / comm_x2g_* prefix match | `system.py` | GQA contract |
| M3 | 5 VLM + 3 LLM-backbone dict config | `config.py` | VLM graph |
| M4 | `_build_vit()` ViT graph | `model.py` | VLM graph |
| M5 | `_build_projector()` mlp / merger / pixel_shuffle | `model.py` | VLM graph |
| M6.1 | Prefill PIM dispatch (`_select_sum_device`) | `system.py` | PIM exec |
| M6.3 | `_simulate_pim_prefill_score()` chunked sampled | `system.py` | PIM exec |
| M6.4 | `_apply_eff_lat()` + `get_pipelining_efficiency_latency()` | `system.py` | PIM exec |
| M7-pre | depth-aware routing groups; `perf * ndec` uniform 제거 | `system.py`, `model.py` | Routing |
| M7 | `Routing` class (conservative / optimistic / list) + DeepStack auto | `model.py` | Routing |
| M8 | `get_capacity_breakdown()` per-GPU dict | `system.py` | Capacity |
| M9 | DeepStack `deepstack_add` inject (sum-only, layer 5/11/17) | `model.py` | VLM graph |
| M12 | `select_best_resolution()` LLaVA-Next AnyRes | `model.py` | VLM graph |
| M13 | Ramulator wrapper `max_L` cache + trace_gen auto-search | `ramulator_wrapper.py`, `system.py` | Tooling |
| M14 | NVLink4 interface + comm_g2g 자연 모델링 | `config.py` | Deployment |
| P1  | PIM SOFTMAX `(0, [0,...])` return (double-count 차단) | `devices.py` | Correctness |

카테고리 분포:
- **Deployment** : M1 / M14 / (assert) — H100 x1·x2 deployment
- **GQA contract** : M2a / M2b / M2c — Qwen / InternVL / LLaVA-Next 정확도 핵심
- **VLM graph** : M3 / M4 / M5 / M9 / M12 — ViT + projector + DeepStack + AnyRes
- **PIM exec** : M6.1 / M6.3 / M6.4 — prefill PIM + chunked + sec.6.1 caveat
- **Routing** : M7-pre / M7 — depth-aware + DeepStack 위치 보존
- **Capacity** : M8 — per-GPU breakdown
- **Tooling** : M0 / M13 — cache key / metadata / wrapper robustness
- **Correctness** : P1 — semantic bug fix

---

## 3-bis. VLM Pipeline: Vision Encoder / Merger / LM Decoder

VLM 한 request 안에서 sub-pipeline 4단계. 각 단계가 simulator graph의 어느
부분으로 들어가는지 분리.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    REQUEST = 1 image + text prompt                   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
   ╔══════════════════════════════════════════════════════════════════╗
   ║  Stage 1.  VISION ENCODER  (per-request, GPU only, single pass)  ║
   ║──────────────────────────────────────────────────────────────────║
   ║                                                                  ║
   ║   [Image HxW]                                                    ║
   ║      │   patchify (patch_size 14 or 16)                          ║
   ║      ▼                                                           ║
   ║   [pre-merge tokens]   T_pre = (H/patch)^2  ×  (Anyres if N>1)   ║
   ║      │                                                           ║
   ║      ▼                                                           ║
   ║   ┌──────────────────────────────────────────────────┐           ║
   ║   │  ViT block (×vit_layers, single-template trick)  │           ║
   ║   │                                                  │           ║
   ║   │   vit_qkv (FC)                                   │           ║
   ║   │   vit_score (MATMUL)                             │           ║
   ║   │   vit_softmax (SOFTMAX)                          │           ║
   ║   │   vit_context (MATMUL)                           │           ║
   ║   │   vit_proj (FC)                                  │           ║
   ║   │   vit_norm1 (NORM)                               │           ║
   ║   │   vit_ff1 (FC) + vit_gelu (ACT) + vit_ff2 (FC)   │           ║
   ║   │   vit_norm2 (NORM)                               │           ║
   ║   │                                                  │           ║
   ║   │   numOp = vit_num_heads × batch × vit_layers     │           ║
   ║   │   (10 Layer objects, scaled by vit_layers in     │           ║
   ║   │    numOp -- avoids 24× Layer object explosion)   │           ║
   ║   └──────────────┬───────────────────────────────────┘           ║
   ║                  │                                               ║
   ║                  ▼                                               ║
   ║   [post-encoder tokens]  (still pre-merger spatial layout)       ║
   ║                                                                  ║
   ║   Code        : model.py :: Transformer._build_vit  [N:M4]       ║
   ║   Dispatch    : system.py simulate() (GPU only, count=1)         ║
   ║   FLOPs / mem : numOp scaling => exact L_layers x per-layer cost ║
   ╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
   ╔══════════════════════════════════════════════════════════════════╗
   ║  Stage 2.  MERGER / PROJECTOR  (per-request, GPU only)           ║
   ║──────────────────────────────────────────────────────────────────║
   ║                                                                  ║
   ║   3 projector_type 분기:                                         ║
   ║                                                                  ║
   ║   (A) "mlp"               LLaVA-1.5 / LLaVA-Next-Mistral          ║
   ║       proj_mlp_fc1 (FC, vit_hidden -> hdim)                      ║
   ║       proj_mlp_act (ACT)                                         ║
   ║       proj_mlp_fc2 (FC, hdim -> hdim)                            ║
   ║       T_post = T_pre  (no token reduction)                       ║
   ║                                                                  ║
   ║   (B) "mlp_with_merger"   Qwen3-VL / Qwen2.5-VL                   ║
   ║       in_dim = vit_hidden × spatial_merge_size^2                 ║
   ║       proj_merger_fc1 (FC, in_dim -> hdim)                       ║
   ║       proj_merger_act (ACT)                                      ║
   ║       proj_merger_fc2 (FC, hdim -> hdim)                         ║
   ║       T_post = T_pre / spatial_merge_size^2  (token down-sample) ║
   ║                                                                  ║
   ║   (C) "pixel_shuffle_mlp" InternVL3-8B-hf                        ║
   ║       in_dim = vit_hidden × spatial_merge_size^2                 ║
   ║       proj_pixel_fc1 / act / fc2                                 ║
   ║       T_post = T_pre / spatial_merge_size^2                      ║
   ║                                                                  ║
   ║   (optional) vit_broadcast G2G    (only if fc_tp > 1)            ║
   ║                                                                  ║
   ║   Code        : model.py :: Transformer._build_projector  [N:M5] ║
   ║   Dispatch    : system.py vision_decoder stage (GPU only)        ║
   ╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
                  [visual tokens + text tokens]
                  T_visual + T_text => total L_in
                              │
                              ▼
   ╔══════════════════════════════════════════════════════════════════╗
   ║  Stage 3.  LM PREFILL  (per-request, ndec layers, GPU + PIM)     ║
   ║──────────────────────────────────────────────────────────────────║
   ║                                                                  ║
   ║   per LM layer (ndec times, routing group-aware):                ║
   ║                                                                  ║
   ║   qkv (FC, qkv_proj_out_total / fc_tp)        on GPU             ║
   ║       │                                                          ║
   ║       ├─► comm_x2g_kv  (X2G,  2 * kv_proj_out / fc_tp) [N:M2a]   ║
   ║       │                                                          ║
   ║       ├─► comm_x2g_q   (X2G,  q_proj_out / fc_tp)      [N:M2a]   ║
   ║       │                                                          ║
   ║       ▼                                                          ║
   ║   score   (MATMUL  L x L x dhead × num_q_heads)   on PIM         ║
   ║                Layer.numOp     = num_q  / attn_tp                ║
   ║                Layer.pim_numOp = num_kv / attn_tp     [N:M2a]    ║
   ║                ramulator trace = score + softmax + context       ║
   ║                                                                  ║
   ║   softmax (SOFTMAX, placeholder)        on PIM        [M:P1]     ║
   ║                exec_time = 0 (covered by score trace)            ║
   ║                                                                  ║
   ║   context (MATMUL, placeholder)         on PIM        [U]        ║
   ║                exec_time = 0 (covered by score trace)            ║
   ║                                                                  ║
   ║   *if layer_idx ∈ deepstack_layers (Qwen3-VL only)*  [N:M9]     ║
   ║   ├─► deepstack_add  (ACT, residual add)        on GPU           ║
   ║                                                                  ║
   ║   comm_x2g_return  (X2G, q_proj_out / attn_tp)        [N:M2a]    ║
   ║   proj  (FC, hdim, hdim / fc_tp)                on GPU           ║
   ║   comm_g2g  (G2G, hdim)                         all-reduce       ║
   ║   norm1 (NORM)                                  on GPU           ║
   ║                                                                  ║
   ║   FFN -- ffn_type branch:                                        ║
   ║      "gated"   (Qwen3/Qwen2.5/Mistral/Vicuna/LLaMA SwiGLU)        ║
   ║         ff1 + ff2 (FC) + glu (ACT) + ff3 (FC)         [M:M2a]    ║
   ║      "standard"  (GPT / OPT GELU MLP)                            ║
   ║         ff1 (FC) + gelu (ACT) + ff2 (FC)                         ║
   ║                                                                  ║
   ║   comm_g2g (G2G, hdim)                                           ║
   ║   norm2 (NORM)                                                   ║
   ║                                                                  ║
   ║   Code        : model.py :: Transformer._build_sum_one_layer     ║
   ║                 [M:M2a, M:M2c, N:M9]                             ║
   ║   PIM dispatch: system.py :: _select_sum_device       [N:M6.1]   ║
   ║   PIM score   : _simulate_pim_prefill_score (chunked) [N:M6.3]   ║
   ║   eff_lat     : _apply_eff_lat (paper sec.6.1)        [N:M6.4]   ║
   ║                                                                  ║
   ║   Routing-group iteration: ndec layers split across groups       ║
   ║                            (conservative / optimistic / list)    ║
   ║                                                                  ║
   ║   Output: KV cache of size 2 * L * num_kv * dhead * dbyte * ndec ║
   ╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
   ╔══════════════════════════════════════════════════════════════════╗
   ║  Stage 4.  LM DECODE  (lout-1 steps, ndec layers each)           ║
   ║──────────────────────────────────────────────────────────────────║
   ║                                                                  ║
   ║   per decode stage  (kv_len = lin + stage):                      ║
   ║                                                                  ║
   ║   qkv  (FC, batch=1, qkv_proj_out_total / fc_tp)                 ║
   ║   comm_x2g_qkv (X2G, batch=1, qkv merged)             [N:M2a]    ║
   ║                                                                  ║
   ║   score   (MATMUL, m=1, n=kv_len, k=dhead)      on PIM           ║
   ║              numOp = num_q / attn_tp                             ║
   ║              pim_numOp = num_kv / attn_tp             [N:M2a]    ║
   ║                                                                  ║
   ║   softmax (SOFTMAX, m=1, n=kv_len)              on PIM [M:P1]    ║
   ║              exec_time = 0                                       ║
   ║                                                                  ║
   ║   context (MATMUL, m=1, n=dhead, k=kv_len)      on PIM [U]       ║
   ║              exec_time = 0                                       ║
   ║                                                                  ║
   ║   (DeepStack NOT applied in decode -- sum-only injection by M9)  ║
   ║                                                                  ║
   ║   comm_x2g_return + proj + comm_g2g + norm1 + FFN + comm_g2g     ║
   ║      + norm2                                                     ║
   ║                                                                  ║
   ║   Code        : model.py :: Transformer._build_gen_one_layer     ║
   ║   Dispatch    : system.py :: _select_gen_device       [M:M6.1]   ║
   ║   eff_lat     : applied per attn block via _apply_eff_lat        ║
   ║                                                                  ║
   ║   Reported: g_time (ms/token) = per-stage avg time               ║
   ║             E2E = s_time + g_time × (lout - 1)                   ║
   ╚══════════════════════════════════════════════════════════════════╝
```

### Per-VLM Component Map

| Model | Vision encoder (ViT) | Merger / Projector | LM backbone | DeepStack | AnyRes |
|---|---|---|---|---|---|
| **Qwen3-VL-4B** | 24L hidden 1024 16-head, gelu_pytorch_tanh, patch 16 | mlp_with_merger (spatial_merge 2) | Qwen3 36L hdim 2560, n_q 32 / n_kv 8, ff 9728, SwiGLU | **layers [5,11,17]** | -- |
| **Qwen2.5-VL-7B** | 32L hidden 1280 16-head, silu, patch 14 | mlp_with_merger (spatial_merge 2) | Qwen2.5 28L hdim 3584, n_q 28 / n_kv 4, ff 18944, SwiGLU | -- | -- |
| **InternVL3-8B-hf** | 24L hidden 1024 16-head, gelu, patch 14 | pixel_shuffle_mlp (ratio 0.5) | Qwen2.5-spec 28L hdim 3584, n_q 28 / n_kv 4, ff 18944, SwiGLU | -- | -- |
| **LLaVA-1.5-7B** | 24L hidden 1024 16-head, gelu (CLIP-ViT), patch 14 | mlp (no merger) | Vicuna 32L hdim 4096, n_q 32 / n_kv 32 (**MHA**), ff 11008, SwiGLU | -- | -- |
| **LLaVA-Next-Mistral-7B** | 24L hidden 1024 16-head, gelu (CLIP-ViT), patch 14 | mlp (no merger) | Mistral 32L hdim 4096, n_q 32 / n_kv 8, ff 14336, SwiGLU | -- | **5 grids** |

각 model entry는 `src/config.py make_model_config` 안 dict로 정의 [M:M3].
ViT / projector / LLM dimension은 `_build_vit` / `_build_projector` /
`_build_sum_one_layer` 가 dict 필드를 그대로 사용 — paper의 "model-agnostic
framework" 주장 근거.

### Visual-token Count per Model (L_visual)

```
Image 672x672 input:
   Qwen3-VL-4B (patch 16, merge 2)    :  (672/16)^2 / 4         = 441 tokens
   Qwen2.5-VL-7B (patch 14, merge 2)  :  (672/14)^2 / 4         = 576 tokens
   InternVL3 default 448x448 (patch 14)
                                       :  (448/14)^2            = 1024 pre-shuf
                                       :  × 0.25 (pixel_shuffle) =  256 tokens
   LLaVA-1.5 default 336x336 (patch 14)
                                       :  (336/14)^2            =  576 tokens
   LLaVA-Next 672x672 (Anyres 5 grids, base 336)
                                       :  best-fit 672x672 grid:
                                          (672/14)^2  = 2304 patch
                                          + (336/14)^2 = 576 base
                                          + 48 newline (use_image_newline)
                                                       = 2928 tokens
```

`select_best_resolution()` [N:M12] picks the optimal grid from
`image_grid_pinpoints`; `compute_visual_tokens()` returns the final
post-merger count consumed by `_build_vit` and `_build_sum_one_layer`.

---

## 4. Data Flow per Inference Stage

```
 USER CLI
    │
    ▼
 main.py --model X --system Y --num_attacc 1 --tp 1 ...
    │ assert num_attacc == tp == ngpu  [M:M1]
    ▼
 make_model_config(X) ──► config dict (q/kv heads, ff_intermediate, deepstack ...)
                                                                       [M:M3]
                          │
                          ▼
 Transformer.__init__   ──► fc_tp / attn_tp / ff_tp split  [M:M2a]
                          │   Layer with numOp + pim_numOp [N:M2a]
                          ▼
 Routing.to_groups(ndec) ──► [(name, device, count, indices), ...]
                                                                       [N:M7]
                          │
                          ▼
 Transformer.build(routing)
     │
     ├─► vision_decoder = [_build_vit + _build_projector]   [N:M4 / M5]
     │
     ├─► sum_decoder_groups = {}                            [N:M7-pre]
     │     for each group:
     │       _build_sum_one_layer
     │         qkv FC                  [M:M2a fc_tp]
     │         comm_x2g_kv             [N:M2a]
     │         comm_x2g_q              [N:M2a]
     │         score MATMUL            [M:M2a numOp+pim_numOp]
     │         softmax SOFTMAX         (PIM returns 0)      [M:P1]
     │         context MATMUL          (PIM returns 0)      [U]
     │         deepstack_add           [N:M9] if layer_idx in [5,11,17]
     │         comm_x2g_return         [N:M2a]
     │         proj FC                 [M:M2a fc_tp]
     │         comm_g2g                [U / M:M14 NVLink4]
     │         FFN (gated/swiglu)      [M:M2a ff_intermediate]
     │
     └─► gen_decoder_groups = {}                            [N:M7-pre]
            per-stage builds (same pattern but kv_len = lin + stage)

 System.simulate(batch, lin, lout, ...)
     │
     ├─► routing setup                                      [N:M7]
     │
     ├─► vision_decoder run (GPU-only, count=1)             [N:M4]
     │
     ├─► for each routing group:
     │     _simulate_sum_group(decoder, group_device)
     │         _select_sum_device → GPU or PIM              [N:M6.1]
     │         if score MATMUL on PIM:
     │             _simulate_pim_prefill_score              [N:M6.3]
     │                 chunked sampled extrapolation
     │                 ramulator_wrapper.run(pim_numOp)     [M:M13]
     │         _apply_eff_lat(group)                        [N:M6.4]
     │     _simulate_gen_group(decoder, group_device)
     │         similar dispatch + eff_lat
     │
     └─► output.csv columns
          s_time, s_fc, s_matmul, ..., s_x2g (new)          [M:M2b]
          g_time (ms), g_fc, g_qkv_time, ..., g_energy (nJ)
          required_cap_per_gpu (renamed)                    [M:M8]
```

---

## 5. Layer Shape Contract (paper-critical)

`Layer` 인스턴스가 ramulator_wrapper로 전달될 때 의미 약속:

```
Layer.m       = query count            (chunked prefill: 1, full prefill: L,
                                         decode: 1)
                                        Wrapper does NOT read this.
                                        m-scaling applied externally.
                                                                      [N:M6.3]

Layer.n       = accumulated KV length
                Wrapper reads as `l` (sequence length).               [U]

Layer.k       = dhead
                Wrapper reads as `dhead`.                              [U]

Layer.numOp   = num_q_heads per AttAcc
                GPU FLOPs / softmax counting.
                Used by Layer.get_flops() + Layer.get_size().          [M:M2a]

Layer.pim_numOp = num_kv_heads per AttAcc
                Ramulator trace (KV-read bound, paper sec.4).
                Wrapper reads via getattr(layer, 'pim_numOp',
                                            layer.numOp).             [N:M2a]
```

PIM `score` MATMUL의 ramulator trace는 **score + softmax + context cycles
전체**를 포함 — `softmax` 와 `context` 는 graph placeholder로 `(0, ...)` 반환.
PIM `softmax`도 마찬가지로 `(0, ...)` 반환 (double-count 차단)             [M:P1]

---

## 6. Deployment Scenarios

```
S1 ─ H100 x 1 (TP=1) ──────────────┐
                                    │   simulator + measurement both work
                                    ├─► num_attacc=1 num_hbm=5
                                    │   NVLink4 (intra-GPU 의미 없음)
                                    │   PIM aggregate = 18.1 TB/s

S2 ─ H100 x 2 (TP=2) ──────────────┐
                                    │   simulator OK / measurement: driver 545+
                                    ├─► num_attacc=2 num_hbm=5
                                    │   NVLink4 inter-GPU 900 GB/s
                                    │   PIM aggregate = 36.2 TB/s
                                    │   eff_lat 0.29 (Qwen2.5-VL) ⚠

DGX A100 x 8 (paper repro) ────────┐
                                    │   simulator only (paper sec.7.2)
                                    ├─► num_attacc=8 num_hbm=5
                                    │   NVLink3 600 GB/s
                                    │   target speedup 4.84x / 2.48x
```

main.py assert `num_attacc == tp == ngpu` 강제 [M:M1].

---

## 7. Where to Look First

paper draft 작성 시 매핑:

| Section | 핵심 파일 + 함수 |
|---|---|
| Architecture diagram | 이 문서 sec.1 + `model.py::Transformer.build` |
| Method: GQA contract | `model.py::Layer.__init__` + `Transformer.__init__` [M:M2a] |
| Method: chunked prefill | `system.py::_simulate_pim_prefill_score` [N:M6.3] |
| Method: sec.6.1 caveat | `system.py::get_pipelining_efficiency_latency` [N:M6.4] |
| Method: VLM graph | `model.py::_build_vit / _build_projector / _build_sum_one_layer` |
| Method: DeepStack | `model.py::_build_sum_one_layer` deepstack_add [N:M9] |
| Method: AnyRes | `model.py::select_best_resolution` [N:M12] |
| Method: routing | `model.py::Routing` + `system.py` simulate routing loop |
| Evaluation: gate R2 | `260511_additional_exp/tier1_simulator/r2_paper_repro.py` |
| Evaluation: multi-VLM | `260511_additional_exp/tier1_simulator/multi_vlm_full_sim.py` |
| Evaluation: ablation | `260511_additional_exp/tier1_simulator/ablation_contribution.py` |
| Evaluation: quant x PIM | `260511_additional_exp/tier2_simulator/w4a16_pim_sim.py` |
| Evaluation: measured  | `260511_additional_exp/tier2_measurement/*` + `tests/r9_*` |

---

## 8. Modification Numbers at a glance

- **Upstream source files touched** : 5 (`main.py`, `src/config.py`, `src/model.py`, `src/system.py`, `src/devices.py`, `src/ramulator_wrapper.py`)
- **M-IDs** : 14 (M0..M14, M2a/b/c sub-ids 포함하면 16) + P1 = 17 modification tags
- **New helper modules** : 4 (`shared/sim_runner.py`, `result_aggregator.py`, `plot_helpers.py`, `vllm_helpers.py`)
- **New test / experiment scripts** : 36 (tests/ 15 + 260511_additional_exp/ 21)
- **Lines of code added** :
  - upstream src/*.py: ~+1500 lines
  - tests/ + docs/: ~+5500 lines
  - 260511_additional_exp/: ~+2200 lines
  - Total: **~+9200 lines on top of upstream**

---

## 9. One-line Take-away

> **Upstream AttAcc (decode-only, MHA, DGX-8, FP16) → VLM-aware (prefill+decode,
> GQA/MHA, H100 x{1,2}, BF16/W4A16) with paper-grade calibration**, all
> backward-compatible: legacy GPT/LLAMA/OPT/MT models continue to reproduce
> upstream behavior unchanged.
