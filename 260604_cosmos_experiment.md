# 260604 실험 runbook — Cosmos 3 + PIM (Omni-modal World Model)

**Last update**: 2026-06-04 v2 (confirm 반영, simulator 보류, TP=1+2 / batch>1 / dual framework 확장)
**Owner**: Minsik
**Scope**: 260601 (VLM) 와 *별도*. Cosmos 3 Nano / Super 의 워크로드 특성 + PIM 활용 가능 영역 정량화. 최종 목표는 **system top-tier paper 후보 (A / B / C) 의 데이터-기반 결정**.

**Resource**: H100 80GB × 2 (user 가용)
- TP=1 → Cosmos3-Nano (16B = 32 GB weight) 단독 / 짧은 context
- TP=2 → Cosmos3-Nano (큰 batch / 긴 context) + Cosmos3-Super (64B = 128 GB weight)
- **두 GPU 모두 measurement** — #1 우리 측정 매트릭스, #2 NVIDIA inference_benchmarks.md 재현 sanity

---

## TL;DR — 오늘 실험 목표 (v2 핵심 pivot)

> **(1) 이론적 PIM gain 계산** + **(2) 그를 뒷받침할 실측 anchor** 만 본다. Simulator 확장은 보류.

세부:
- Cosmos 3 의 inference 가 어디서 BW-bound 인지 — *순수 H100 measurement* 로 anchor 확보
- 거기서 **AttAcc paper claim BW (242 TB/s) 를 가정** 했을 때 산술적 lower-bound 가 얼마인지 계산 (= 이론적 PIM gain)
- 측정 axis: **TP ∈ {1,2}**, **batch ∈ {1, 2, 4, 8}** (real-time 시나리오), **framework ∈ {vLLM-Omni, PyTorch+Diffusers}**, **model ∈ {Nano, Super}**

> **Simulator 확장 (DM denoising loop, MoT layer, tier policy, event queue) 은 별도 후속 작업 — 오늘 scope 에서 제외.**

---

## 1. Cosmos 3 사실 정리 (cross-checked)

### 1.1 Architecture 정확값 (HF config.json 기준)

| Component | Nano | Super |
|---|---|---|
| **Text** hidden / layers / Q heads / KV heads / FFN | 4096 / 36 / 32 / 8 / 12288 | 5120 / 64 / 64 / 8 / 25600 |
| **max_position_embeddings** | 262144 (256K) | 262144 (256K) |
| dhead | 128 | 128 |
| vocab_size | 151936 | 151936 |
| dtype | BF16 | BF16 |
| tie_word_embeddings | false | false |
| **Vision (ViT)** depth / hidden / patch / heads | 27 / 1152 / 16 / 16 | 27 / 1152 / 16 / 16 |
| spatial_merge_size | 2 | 2 |
| **temporal_patch_size** | 2 | 2 |
| DeepStack layers | [8, 16, 24] | [8, 16, 24] |
| out_hidden_size (ViT→LLM proj) | 4096 | 5120 |
| Total parameters | 16B (8B reasoner + 8B generator) | 64B (32B + 32B) |
| Backbone | Qwen3-VL-8B | Qwen3-VL-32B |

### 1.2 MoT (Mixture-of-Transformers) 구조 — 핵심

- 두 개의 transformer tower: **AR reasoner** + **DM (Diffusion) generator**
- *동일 layer 안에서* AR token 과 DM token 이 **joint attention** 으로 묶임 (KV cache 공유)
- *layer 별로* AR / DM token 이 **separate QKV / FFN parameter set** 사용
- 즉 1 layer 의 weight ≈ 2× standard transformer (AR 분 + DM 분), 단 KV cache 는 합쳐서 1 개

### 1.3 Generation 방식

- **Text → AR autoregressive decoding** (다른 LLM 과 동일)
- **Image / Video / Audio / Action → Diffusion iterative denoising**, **35 step default**, guidance_scale 6.0, flow_shift 10.0
- 각 denoise step 마다 *전체 KV (AR + 이전 DM noise)* 를 read → score / softmax / context → output projection

### 1.4 입력 / 출력 modality — **모두 측정 대상** (v2 확장)

- **입력**: text + image + video clip + audio + action 궤적 + embodiment 정보 (9D camera / 32D humanoid 등)
- **출력**: dynamic video + image + audio (stereo AAC 48 kHz) + action commands

#### Task 매트릭스 (Phase 1.1 의 또 다른 축)

| Task | 입력 modality | 출력 modality | 비고 |
|---|---|---|---|
| `t2v` | text | video | 기본, NVIDIA benchmark 매핑 |
| `t2i` | text | image | image-only generation (denoise step 동일하나 frames=1) |
| `i2v` | image + text | video | image-conditioned video (DriveGen 시나리오) |
| `v2v` | video + text | video | video continuation / extension |
| `t2a` | text | audio | audio generation (stereo AAC, framework 지원 필요) |
| `multi2v` | text + image + audio | video | full omni input → video (Topic C 의 streaming 시나리오 baseline) |
| `multi2action` | text + image + video | action seq | world model / agent inference (embodied robot) |

- video: 256p (448×256) / 480p (832×480) / 720p (1280×720), **5-300 frames**, default 189 frames @ 24 fps = 7.875 sec
- audio: 48 kHz stereo (AAC) — tokenizer 가 sec 당 N token 으로 변환 (Phase 1.8 에서 실측)
- image: 단일 frame, frames=1 + denoise_steps 동일

> Modality 별 KV / token 양과 latency profile 이 *서로 다르기 때문에* — Topic A (tiered KV) 의 modality-aware policy 와 Topic C (per-modality SLO) 는 이 매트릭스 전체가 anchor.

### 1.5 공식 H100 / H200 / B200 inference benchmark (NVIDIA Cosmos repo `inference_benchmarks.md`)

공식 표 구조 (`https://github.com/NVIDIA/cosmos/blob/main/inference_benchmarks.md`, 2026-06-04 확인):

- 컬럼: `256p/1, 256p/4, 256p/8, 480p/1, 480p/4, 480p/8, 720p/1, 720p/4, 720p/8` (resolution / TP)
- 행: inference engine ∈ {PyTorch, vLLM-Omni, Diffusers}
- GPU 별로 별도 표 분리: `RTX_PRO_6000_Blackwell, H20, H100_NVL, H200_NVL, H100_80GB_HBM3_SXM, H200_141GB_HBM3, B200, B300`
- task 별로 별도 표 분리: t2v / i2v / t2i
- 공란 = "아직 측정 안 됨" (지원 안 함 아님)

**우리가 직접 page 에서 인용한 row (anchor 비교용)**:

| Model | Task | GPU | Engine | 256p/1 | 256p/4 | 256p/8 | 480p/1 | 480p/4 | 480p/8 |
|---|---|---|---|---|---|---|---|---|---|
| Cosmos3-Nano | t2v | H200 141GB HBM3 | PyTorch | **3.34 s** | 3.19 s | 13.97 s | 214.28 s | 67.48 s | 41.26 s |

그 외 수치는 공식 page 에 있지만 우리가 verbatim 으로 옮기지 않은 상태 — Phase 0.2 sanity 비교 전에 표 추가 인용 필요.

> 이전 v1 draft 에 적었던 "Nano 720p/1 H100 NVL = 3.95 s, Super 720p/1 = 101.27 s" 수치는 **page 와 어긋난 추정치** 였음. 위 verified row 만 paper 인용 가능.

### 1.6 우리가 도출한 derived metrics (paper-anchor 후보)

| 지표 | Nano | Super | 비교: LLaVA-Next |
|---|---|---|---|
| KV cache per token (BF16) | 144 KB | 256 KB | 14 KB |
| KV per request (256K context) | **38.7 GB** | **68.7 GB** | 0.04 GB (full lin 3008) |
| Single H100 80GB max batch | ~1 | <1 (TP=2 필수) | ~14 |
| Weight bytes (BF16) | 32 GB | 128 GB | 14 GB |
| KV / weight crossover context (b=1) | 217 K tok | 488 K tok | — |
| KV / weight crossover context (b=8) | 27 K tok | 61 K tok | — |
| 1 video 의 DM denoise read total (35 step × (W + KV)) | **2.2 TB (b=1) / 12.1 TB (b=8)** | **6.8 TB (b=1) / 24.6 TB (b=8)** | — |

#### 이론적 AttAcc 가속 (수정된 framing — *weight 는 H100, KV 만 PIM*)

Per step time = max(weight/H100_BW, KV/AttAcc_BW). 천장은 **KV-dominated regime 에서 72×**. 실제 realized:

| Cosmos3-Nano | b=1 | b=8 |
|---|---|---|
| ctx=32K | 1.15× | 2.21× |
| ctx=131K | 1.60× | 5.83× |
| **ctx=262K** | **2.21×** | **10.66×** |

| Cosmos3-Super | b=1 | b=8 |
|---|---|---|
| ctx=32K | 1.07× | 1.54× |
| ctx=131K | 1.27× | 3.15× |
| **ctx=262K** | **1.54×** | **5.29×** |

**해석**: Cosmos 3 가 AttAcc 의 strong case 가 되려면 *batch ≥ 8 AND context ≥ 128K* 의 영역이어야 함 (KV dominance > 80%). 그래야 AttAcc paper Fig.14 의 4.84× 를 넘어서 ~5-11× 의 realized speedup 확보 가능. Real-time (batch=1) 시나리오 에서는 **2.2× 이하** 라서 weight-bound — 이 fact 가 Topic A (capacity, tiered KV) 의 *복합 angle* 정당화.

---

## 2. 세 주제 (A / B / C) 의 실험 분기

### Topic A — Modality-aware Hierarchical KV Tiering
*256K KV 가 modality / temporal locality 별로 BW pressure 가 다르다* → bank / BG / buffer 의 PIM tier 를 동시에 활용해 *system-wide capacity & BW Pareto 확장*.

핵심 검증: video KV 의 temporal locality (얼마나 오래된 frame 의 KV 가 attend 되나) + modality 별 access frequency.

### Topic B — AR + Diffusion Generation + PIM
*35 step denoising × full KV read* 가 AttAcc 의 BW 우위를 *paper Fig.14 의 4.84× 보다 한 자릿수 더 강력하게* 만든다.

핵심 검증: 1 video generation 의 read traffic 정확 측정 + AttAcc 시뮬에서 denoising loop ratio 계산.

### Topic C — Multi-modal Streaming SLO + PIM
*Audio (16kHz) / video (24fps) / action (event) / text* 가 비동기 도착 → modality-specific SLO + PIM 의 추가 자원 dimension.

핵심 검증: Cosmos 3 가 실제로 streaming 시나리오로 추론되는가 + per-modality latency budget.

---

## 3. 실험 리스트 (오늘 scope = M + A, S 항목은 [DEFERRED])

분류: **M** = H100 measurement, **A** = analysis on collected data, ~~**S**~~ = simulator extension (오늘 제외, 후속).

**Phase 1 매트릭스 axis** (모든 측정 실험 공통):

| Axis | 값 | 비고 |
|---|---|---|
| `model` | Nano / Super | Nano TP=1 가능, Super TP≥2 필수 |
| `tp` | 1, 2 | Nano 는 둘 다, Super 는 TP=2 |
| `batch` | 1, 2, 4, 8 | batch=1 만 보면 real-time scenario 가 안 잡힘 (user req) |
| `framework` | vLLM-Omni, PyTorch+Diffusers | dual measurement (cost x2 OK, framework overhead 분리) |
| `task` | **t2v / t2i / i2v / v2v / t2a / multi2v / multi2action** | omni-modal coverage (Section 1.4) |
| `resolution` | 720p (default), 1.6 에서 256p / 480p / 720p sweep | video 출력 task 에서만 의미 있음 |
| `frames` | 189 (default ≈ 7.9 s @ 24 fps), 1.7 에서 24 / 96 / 189 / 300 sweep | t2v / i2v / v2v / multi2v 에만 적용 |

> Super × TP=1 은 *layer-wise sim anchor* 용이라 부분 측정만 (1 layer forward latency 등).
> `t2a` 와 `multi2action` 은 framework support 가 불확실 → Phase 0.1 environment check 에서 가능 여부 확인 후 매트릭스 cull.

### Phase 0 — 환경 / 기반 (1 일)

| # | Type | 제목 | 목적 | H100 hours |
|---|---|---|---|---|
| 0.1 | M | Cosmos 3 환경 셋업 — vLLM-Omni + PyTorch + Diffusers + Nano/Super weight download | 측정 가능 상태 확인 | 4 |
| 0.2 | A | NVIDIA `inference_benchmarks.md` 재현 sanity — Nano 720p t2v 3.95 s 가 우리 H100 에서도 나오는지 | baseline 신뢰 anchor | 1 |
| ~~0.3~~ | ~~S~~ | ~~`src/model.py` 에 DM tower 분기 + Layer 빌더~~ | **[DEFERRED]** — 시뮬레이터는 별도 후속 작업 | — |
| ~~0.4~~ | ~~S~~ | ~~`src/system.py` 에 denoising loop 파라미터~~ | **[DEFERRED]** | — |

### Phase 1 — Cosmos 3 workload characterization (1-2 주, A/B/C 공통)

| # | Type | 제목 | 목적 | H100 hours |
|---|---|---|---|---|
| 1.1 | M | **E2E latency 매트릭스** — model × TP × batch × framework × resolution × frames | "TP=1 batch=1 720p 가 N s" 정도가 아닌 *시나리오 전체* 의 latency surface 확보 | 12 |
| 1.2 | M | **Peak VRAM 프로파일** — model load + KV cache growth, batch × TP 별 | 메모리 capacity 정확 측정 + AttAcc 의 capacity 우위 anchor | 6 |
| 1.3 | M | **Phase-wise breakdown** — model load / prefill / first denoise / subsequent denoise / decode / total | AR ↔ DM 시간 비중 정확 측정 | 8 |
| 1.4 | M | **Bandwidth profile via Nsight Systems** — DRAM read GB / step / phase | **BW saturation 확인 (이론적 PIM gain 의 anchor)** | 10 |
| 1.5 | M | **AR-only vs DM-only** (vLLM `--reasoner`, `--omni` 분리 / PyTorch 분기) | tower 별 compute 분해 | 4 |
| 1.6 | M | **Resolution scaling** 256p / 480p / 720p × Nano + Super × TP × batch | 해상도에 따른 BW 압력 scaling law | 10 |
| 1.7 | M | **Frame count scaling** 24 / 96 / 189 / 300 frames × Nano × TP × batch | temporal scaling | 8 |
| 1.8 | M | **Tokenizer 정확 token 수 측정** — image / video frame / audio sec 당 token | KV/req 추정의 *실측 anchor* (38 GB / 67 GB derivation 정확화) | 2 |

### Phase 2 — Topic A (Tiered KV) 전용 실측 (1-2 주)

| # | Type | 제목 | 목적 | H100 hours |
|---|---|---|---|---|
| 2.1 | M | **Attention pattern analysis** — Nsight Compute 의 sparse-vs-dense pattern, batch × TP 별 | sliding window / sparse attention 여부 검증 | 8 |
| 2.2 | M | **Temporal KV locality** — 매 denoise step 별 attended token 집합 (token age histogram), Hook attention output | "cold KV 가 정말 안 attend 되나" 검증 | 12 |
| ~~2.3~~ | ~~S~~ | ~~PIM tier mode 비교 simulator~~ | **[DEFERRED]** | — |
| ~~2.4~~ | ~~S~~ | ~~Tier policy simulator~~ | **[DEFERRED]** | — |
| 2.5 | A | **Per-modality KV breakdown** — text / image / video / audio token 비중 + 각자 KV cost | tier 결정 근거, paper figure 후보 | 0 |

### Phase 3 — Topic B (AR+DM Generation) 전용 실측 (1-2 주)

| # | Type | 제목 | 목적 | H100 hours |
|---|---|---|---|---|
| 3.1 | M | **Denoise step sweep** — `num_inference_steps=10/20/35/50/100` × batch | denoise step ↔ latency linear 검증 + 35 step 정합성 | 8 |
| 3.2 | M | **Denoise step 의 read traffic** (Nsight DRAM read GB / step) | "step 마다 full KV read" 가설 검증 → **이론적 PIM gain 의 핵심 anchor** | 10 |
| ~~3.3~~ | ~~S~~ | ~~AttAcc simulator denoising loop~~ | **[DEFERRED]** | — |
| ~~3.4~~ | ~~S~~ | ~~Sim AR vs AR+DM PIM utilization~~ | **[DEFERRED]** | — |
| 3.5 | A | **이론 BW lower-bound 비교 계산** — H100 3.35 TB/s vs AttAcc 242 TB/s, *측정된 read traffic* 대입 | 이상적 가속 ceiling 정량 (페이퍼 hook) | 0 |
| 3.6 | M | **guidance_scale / flow_shift 영향** — 1.0 / 6.0 (default) / 12.0 | inference 알고리즘 variation 의 BW 영향 | 4 |

### Phase 4 — Topic C (Streaming SLO) 전용 실측 (2 주)

| # | Type | 제목 | 목적 | H100 hours |
|---|---|---|---|---|
| 4.1 | M | **Streaming arrival pattern measurement** — vLLM-Omni / PyTorch 에 audio/video/text 비동기 도착 가능한가 | thesis foundation 검증 (가능하지 않으면 C 폐기) | 6 |
| 4.2 | M | **Single-modality latency budget** — audio / video / action / text 의 individual P50 / P95 / P99, batch × TP 별 | per-modality SLO 정량 | 8 |
| 4.3 | M | **Mixed-modality interference** — 3 stream 동시 in-flight 시 P99 dilation | scheduling 필요성 검증 | 12 |
| ~~4.4~~ | ~~S~~ | ~~Event-driven simulator extension~~ | **[DEFERRED]** | — |
| 4.5 | A | **Per-modality SLO formalization** — 4 modality 의 latency budget 표 + paper figure 후보 | new metric definition | 0 |

---

## 4. 총 자원 budget (v2 — simulator 제외)

- **H100 시간 합**: Phase 0.1 + 1.1-4.3 (M only) ≈ **132 hours** (TP×batch×framework axis 확장으로 v1 의 112 → 132)
- **2×H100 병렬 wall-clock**: ≈ **3 일** (H100 #1 우리 매트릭스, #2 NVIDIA 재현 sanity + Phase 1.6/1.7 sweep)
- **분석 (A 항목)**: 측정 완료 후 **3-5 일** (Section 3.5 이론 lower-bound 계산이 핵심)
- ~~시뮬레이터~~ — **오늘 scope 제외, 후속 작업**

총 **2-3 주** 안에 *paper-pitch 의 데이터 anchor 확보 → A/B/C 중 진행 방향 결정*.

---

## 5. 산출물 (260604_cosmos_exp/results/ 디렉토리)

per-host JSON convention: `<name>.json` (default) + `<name>_<host>.json` (예: `_h100`) 둘 다 save.

| 파일 | 의미 | Phase |
|---|---|---|
| `cosmos_env_check.json` | Phase 0.1 환경 / 버전 / weight path | 0 |
| `cosmos_nvidia_repro.json` | Phase 0.2 NVIDIA benchmark 재현 sanity | 0 |
| `cosmos_e2e_latency.json` | Phase 1.1 의 (model × TP × batch × framework × res × frames) 매트릭스 | 1 |
| `cosmos_vram_profile.json` | Phase 1.2 의 peak / over-time memory | 1 |
| `cosmos_phase_breakdown.json` | Phase 1.3 의 phase 별 시간 분해 | 1 |
| `cosmos_bandwidth_profile.json` | Phase 1.4 의 DRAM read / phase (Nsight Systems) | 1 |
| `cosmos_ar_vs_dm.json` | Phase 1.5 의 tower 별 시간 | 1 |
| `cosmos_resolution_scaling.json` | Phase 1.6 (res × model × TP × batch) | 1 |
| `cosmos_frame_scaling.json` | Phase 1.7 (frames × model × TP × batch) | 1 |
| `cosmos_tokens_per_modality.json` | Phase 1.8 (실측 token count) | 1 |
| `cosmos_attention_pattern.json` | Phase 2.1 의 sparse / dense (Nsight Compute) | 2 |
| `cosmos_kv_temporal_locality.json` | Phase 2.2 의 token age histogram | 2 |
| `cosmos_modality_kv_breakdown.json` | Phase 2.5 | 2 |
| `cosmos_denoise_step_sweep.json` | Phase 3.1 | 3 |
| `cosmos_denoise_step_traffic.json` | Phase 3.2 (Nsight DRAM read / step) | 3 |
| `cosmos_theoretical_pim_gain.json` | **Phase 3.5 — 이론적 PIM gain 계산 결과 (핵심 산출물)** | 3 |
| `cosmos_guidance_sweep.json` | Phase 3.6 | 3 |
| `cosmos_streaming_arrival.json` | Phase 4.1 | 4 |
| `cosmos_modality_slo_budget.json` | Phase 4.2 | 4 |
| `cosmos_modality_interference.json` | Phase 4.3 | 4 |
| `cosmos_per_modality_slo.json` | Phase 4.5 formalization | 4 |

---

## 6. 결정 트리 — 측정 결과를 보고 paper 방향 결정

```
Phase 1 (workload characterization) 완료
    │
    ├─ Phase 2.2 (cold KV access frequency) 측정
    │     ├─ "cold KV 가 5% 이하 attend" → Topic A 강화 (tier benefit 큼)
    │     └─ "cold KV 가 50% 이상 attend" → Topic A 약화, 다른 angle 검토
    │
    ├─ Phase 3.2 (denoise step traffic, KV-only derived) 측정
    │     ├─ "kv_dominance > 80% at b≥8, ctx≥128K" → Topic B 강화 (realized ≥ 5×)
    │     ├─ "kv_dominance 50~80%" → Topic B 중간 (realized 2~3×)
    │     └─ "kv_dominance < 50%, weight-bound" → Topic B 약화 (realized < 2×)
    │
    └─ Phase 4.1 (streaming 가능성) 측정
          ├─ "vLLM-Omni 가 multi-stream support" → Topic C 진입
          └─ "1-shot generation only" → Topic C 폐기

Phase 2/3/4 결과를 paper hook 의 *quantitative thesis* 와 매칭
    │
    ├─ A + B 둘 다 강 → unified paper "Heterogeneous PIM for Omni-modal Inference"
    ├─ B 만 강 → "Diffusion-AR Inference Architecture" single-angle paper
    ├─ A 만 강 → "Hierarchical KV Tiering" extension paper (AttAcc follow-up)
    └─ 셋 다 약 → 시뮬레이터 결과만으로 *workload characterization* 페이퍼 (lower-tier 옵션)
```

---

## 7. 의존성 + risks

### Risks (paper 깨는 가능성)
- **R-CO1**: Cosmos 3 의 sliding window / sparse attention 이 명시되면 Topic A 의 "cold KV idle" 가설 무너짐
- **R-CO2**: DM denoising 의 step-wise traffic 이 *KV 전체* 가 아닌 *summary token* 만 read 라면 Topic B 의 72× leverage 추정 깨짐
- **R-CO3**: Cosmos 3 가 *batch-static 1-shot generation* 만 가정하면 Topic C 의 streaming SLO 가설 폐기
- **R-CO4**: vLLM-Omni 가 attention pattern dump / Nsight integration 어려우면 Phase 1.4 + 2.1 + 2.2 측정 막힘 → simulator extrapolation 만 가능 → paper anchor 약화
- **R-CO5**: Cosmos 3 의 inference 가 *vLLM-Omni 가 아닌 PyTorch only* 면 measurement 코드 작성 부담 ↑

### 의존성 / 사전 작업
- Cosmos 3 의 tech report PDF (research.nvidia.com) 정독 — `num_inference_steps`, joint attention 의 정확 정의, denoising 의 attended set
- Cosmos 3 GitHub (`nvidia/cosmos`) 의 `inference_benchmarks.md` 외의 inference script 코드 분석
- H100 노드에 충분한 디스크 용량 (Nano + Super weights ≈ 200 GB, KV cache 측정 시 추가)

---

## 8. 의사결정 결과 (2026-06-04 confirmed)

- [x] **Phase 1~4 의 18 개 실험 전부 진행** — 단 simulator (S) 항목은 보류, M + A 만 today scope
- [x] **2×H100 의 할당**: #1 우리 측정 매트릭스, #2 NVIDIA inference_benchmarks.md 재현 sanity. Super 는 TP=2.
- [x] **Cosmos 3 serving framework**: vLLM-Omni + PyTorch + Diffusers **둘 다 측정** (framework overhead 분리, cost x2 OK)
- [x] **TP / batch sweep**: Nano 는 TP=1 AND TP=2, batch ∈ {1, 2, 4, 8} (real-time scenario 고려)
- [x] **simulator 확장 보류** — 오늘은 *이론적 PIM gain 계산 + 실측 anchor* 만. simulator 는 후속 작업.
- [x] **실험 directory**: `attacc_simulator/260604_cosmos_exp/` 별도 폴더, `260601` 과 완전 분리
- [ ] **paper venue 우선순위**: ISCA / HPCA vs ASPLOS vs OSDI / SOSP — Phase 1 결과 본 후 결정

---

## 9. 다음 행동 (구현 순서)

1. **`260604_cosmos_exp/` 디렉토리 생성** + 구조:
   ```
   260604_cosmos_exp/
   ├── measurement/        # H100 실측 스크립트 (Phase 0, 1, 2, 3, 4 의 M 항목)
   │   ├── vllm_omni/      # vLLM-Omni 기반
   │   └── pytorch/        # PyTorch + Diffusers 기반
   ├── analysis/           # A 항목 (이론 계산, breakdown)
   ├── shared/             # hw_detect, result_aggregator 등 공통 유틸 (260601 에서 빌려옴)
   └── results/            # JSON 산출물 (per-host copy convention)
   ```
2. **Phase 0.1**: `measurement/env_setup.py` — vLLM-Omni + PyTorch + Diffusers + Cosmos weight 다운로드 / 환경 체크
3. **Phase 0.2**: `analysis/nvidia_repro.py` — NVIDIA benchmark 표 재현 sanity. 매칭 시 `COSMOS_NVIDIA_KEY` env 로 H100_NVL / H100_80GB_HBM3_SXM 구분 (지금 verified row 20 개). Diffusers 256p 비교 시 caveat 표시 (320×192 vs 448×256)
4. **Phase 1.1**: `measurement/e2e_latency_matrix.py` — 핵심 매트릭스 측정 스크립트
5. **Phase 1.2~1.8**: 나머지 measurement 스크립트
6. **Phase 2.1~2.2, 3.1~3.2, 3.6, 4.1~4.3**: topic-specific measurement
7. **Phase 3.5**: `analysis/theoretical_pim_gain.py` — *이론적 PIM gain 계산 (페이퍼 hook 의 가장 중요한 산출물)*
8. **Phase 2.5, 4.5**: analysis 마무리

각 스크립트는 hw_detect 호출 → 결과는 `<name>.json` + `<name>_<host>.json` 두 copy save.

---

## 10. Paper hook 계산법 (Phase 3.5) — 정직한 framing

PIM 가속의 *천장* 과 *실현치* 가 다름. AttAcc paper 의 4-stack BW 242 TB/s 를 가정해도:

```
upper_bound_gain  = ATTACC_BW / H100_BW = 242 / 3.35 = 72.2x  (KV-only, 도달 불가)
realized_gain     = (weight + KV) / max(weight/H100_BW, KV/ATTACC_BW)
                    -- weight 는 H100 그대로, KV 만 PIM 으로
kv_dominance(%)   = KV / (KV + weight) * 100
                    -- 100% 에 가까울수록 realized → upper_bound
```

**Cosmos3-Nano realized_gain 격자**:

| ctx ↓ \ batch → | 1 | 8 |
|---|---|---|
| 32 K | 1.15× | 2.21× |
| 131 K | 1.60× | 5.83× |
| **262 K** | 2.21× | **10.66×** |

→ paper hook 으로 쓸 anchor 는 **batch=8 + 256K context → 10.66×**.

### Phase 3.2 의 measured vs Phase 3.5 의 grid

- Phase 3.5 의 grid `rows[]` = **순수 analytic** (model × ctx × batch × denoise_steps). 매트릭스 cell 의 ctx 는 hypothetical.
- Phase 3.2 의 measured rows 는 **`measured_anchors[]` 별도 섹션** 으로 들어감. 각 anchor 는 자기 자신의 `(model, batch, resolution, frames, denoise_steps, engine_tag, actual_context_tokens)` 를 들고 다님. analytic grid 의 ctx 값에 *옆으로 끼워넣지 않음*.
- Phase 3.5 가 measured 를 가져올 땐 **위 7개 key 모두 strict 일치** 필요. batch=1 측정값이 batch=8 cell 에 silent substitution 되지 않음.

### Phase 3.2 의 actual vs analytic context

- `actual_context_tokens` = `estimate_visual_tokens(resolution, frames)` + text_tokens_approx — Cosmos 가 실제 처리하는 context. **이것만이 measured-anchor 매칭 key**
- `analytic_context_anchor_tokens` = `--analytic-context-tokens` 인자값. analytic KV/req 라벨 출력용. **매칭에 사용 금지**
- temporal_groups 는 **ceil** (`-(-frames // tp)`). 189 frames, tp=2 → 95 groups (floor 94 로 짤라 1 group 누락하던 버그 fix)

### Diffusers 256p 해상도 caveat

- `cosmos_facts.RESOLUTIONS["256p"]` = (448, 256) — PyTorch / vLLM-Omni canonical
- `ENGINE_RESOLUTION_OVERRIDES["Diffusers"]["256p"]` = (320, 192) — NVIDIA 공식 benchmark page 의 Diffusers row 가 내부적으로 사용하는 해상도
- `nvidia_repro.py` 가 engine=Diffusers + col=256p/\* 매칭 시 `caveats=["resolution_mismatch ..."]` 표기

### Phase 3.2 의 nsys CSV 파싱 strictness

- explicit `read` 키워드 + byte unit 컬럼만 1차 매칭
- 또는 Operation/Op/Kind/Type 컬럼이 있을 때 `Operation == read/load/ld/READ` row 만 합산 (대소문자 무관)
- "non-write byte column" fallback 완전 제거 → `Total [MB]` 같은 read+write 합계가 read 로 오인되는 버그 fix

---

## 갱신 history

- **2026-06-04 v1** — 초안. Cosmos 3 facts cross-checked (HF config.json + NVIDIA benchmark). 18 실험 list, 결정 트리, risks 포함.
- **2026-06-04 v2** — confirm 반영. Simulator (S) 항목 보류 (오늘 scope 제외). Phase 1 매트릭스 axis 확장 (TP=1+2, batch ∈ {1,2,4,8}, framework={vLLM-Omni, PyTorch}). NVIDIA repro sanity 를 Phase 0.2 로 추가. H100 시간 112 → 132 hours. 산출물 / 다음 행동 갱신.
- **2026-06-04 v3** — 공식 source 일괄 정렬 + 누적 risk fix. (1) HF repo `nvidia/Cosmos3-{Nano,Super}` (대시 제거), 단일 `Cosmos3OmniPipeline`, `UniPCMultistepScheduler(flow_shift=10.0)`, 공식 negative_prompt 반영. (2) `NVIDIA_BENCHMARK` raw markdown 으로 재구축 — 컬럼 시프트 버그 fix, 빈 셀 보존, H100_NVL / H100_80GB_HBM3_SXM / H200_141GB_HBM3 합쳐 20 row + `NVIDIA_BENCHMARK_NOTES` 로 `(*)` annotation 보존. (3) Phase 3.5 paper hook 의 72× framing 정정 → realized_gain = `max(weight/H100, KV/AttAcc)`, anchor 는 Nano @256K b=8 = 10.66×. (4) Phase 3.2 honest boundary: `cudaProfilerStart/Stop` 으로 model load + warmup 제외, capture range = text encode + denoise + decode (amortized). status ∈ {measured, analytic_only}. (5) Phase 3.5 의 measured override 제거 — `measured_anchors[]` 별도 섹션. strict lookup key = (model, batch, resolution, frames, denoise_steps, engine_tag, actual_context_tokens). (6) Phase 3.2 의 `actual_context_tokens` 추가 (visual + text), `analytic_context_anchor_tokens` 로 rename. (7) `estimate_visual_tokens` 의 temporal_groups ceil 적용 (189f → 95 groups). (8) `ENGINE_RESOLUTION_OVERRIDES["Diffusers"]["256p"] = (320, 192)`. (9) CSV unit parser regex 기반 재작성 + write column 제외 + Operation row 필터 path. (10) 26 스크립트 import sanity 0 failures, 9 case CSV parser unit test PASS, schema test PASS, NVIDIA benchmark spot-check PASS.
