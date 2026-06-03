# 260601 실험 runbook — VLM Simulator Calibration & Gain 측정

**Last update**: 2026-06-04 v5 (R7-R16 + B1/B2 invalid-lin fix, capacity_framing HW-aware + fail-fast, 체크리스트 가설 reframe, H100 main 선언)
**Owner**: Minsik
**Repo**: `attacc_simulator/` (origin `minsik-choi126/attacc_simulator_vlm`, branch `main`)
**Latest commit**: `2b1be3b` (Runbook v4 — H100 readiness + R16 + hw_detect 문서화). 본 갱신 (v5) 은 그 위에 stage 됨.

> **Canonical HW = H100 (paper main table source).**
> A6000 결과는 *sanity / reference* 용으로만 인용. paper figure / main table 수치는 H100 결과 (`*_h100.json`) 로 작성. cross-HW 비교 (R12 / cross_hw_compare) 가 필요하면 두 HW 결과 모두 git push 후 별도 정합성 표만 추가.

## TL;DR — 노드에 도착해서 무엇을 하면 되는가

```bash
cd attacc_simulator
git pull                                                          # 최신 받음
ls ramulator2/ramulator2* 2>/dev/null || bash set_pim_ramulator.sh # 없으면 빌드
nvidia-smi --query-gpu=name --format=csv,noheader                 # A6000 / H100 확인

# 한 줄로 전체 실험 -- 스크립트가 HW 를 자동 감지해 sim 의 gpu/interface 선택
python 260511_additional_exp/calibration/run_all.py

# 끝난 후 -- 각 결과는 <name>.json (latest) 와 <name>_<host>.json (per-HW) 둘 다 저장됨
ls 260511_additional_exp/results/{calibration_*,capacity_regime_*,slo_throughput_*}.json
```

A6000 노드에서 한 번, H100 노드에서 한 번 돌린 후, 한 곳에서:

```bash
python 260511_additional_exp/calibration/cross_hw_compare.py --save
```

→ `results/calibration_cross_hw.json` + 콘솔 표.

### 동일 명령으로 양쪽 HW — 어떻게 가능?

`260511_additional_exp/shared/hw_detect.py` 가 `nvidia-smi` 로 host GPU 를 감지해서 sim 의 `gpu` / `interface` 인자를 자동 매핑:

| Host GPU | sim_runner `gpu` | sim_runner `interface` | roofline ridge (ops/byte) |
|---|---|---|---|
| RTX A6000 | `A6000` | `NVLINK_BRIDGE` (112 GB/s) | 403 |
| H100 SXM5 | `H100` | `NVLINK4` (900 GB/s) | 295 |
| A100 SXM4 | `A100a` | `NVLINK3` (600 GB/s) | 201 |

따라서 동일한 `run_all.py` 명령이 A6000 노드에선 A6000 sim config 를, H100 노드에선 H100 sim config 를 자동 선택. **수동 flag 불필요**.

---

## 목표 한 문장 정리

1. **Phase 1 (정확도)** — simulator 의 `s_time` (prefill) / `g_time` (decode) 가 vLLM 실측의 TTFT / ITL 과 ≈ 1.0× 범위에 일치하는지 5 VLM × image_size × batch 8 단계에서 검증. A6000 / H100 양쪽에서.
2. **Phase 2 (VLM-specific gain)** — paper 의 main claim 인 *"AttAcc 가 VLM 에서 LLM 대비 추가로 얻는 gain"* 을 4 방식으로 정량 측정.
3. **Phase 3 (figure 갱신)** — 기존 multi-VLM matrix / SLO / roofline / capacity 결과를 Fix A+B+C 적용 후 재실행해서 paper 의 표/그림에 들어갈 최신 수치 산출.

---

## 시뮬레이터 수정사항 (commit `1be1997` + `이번 commit`)

| Fix | 파일 | 변경 요지 | LLM regression |
|---|---|---|---|
| **A** | `src/model.py` `compute_vit_attention_tokens` | ViT attention 을 full token 으로 (spatial_merge 가 attention 단계에서 일찍 적용되던 버그 제거). Qwen2.5-VL / Qwen3-VL / InternVL3 의 sim_s 가 8× 가까이 증가하여 측정값에 근접 | 0 (`vit_layers=0` 일 때 호출 안 됨) |
| **B** | `src/model.py` + `src/system.py` + `src/config.py` | VLM 공통 floor overhead 22 ms 추가 (`vlm_floor_overhead_ms`, 5 VLM dict 만 설정). LLaVA-1.5 의 22.5 ms gap 흡수 | 0 (default 0.0, vision_decoder 빈 graph 일 때 skip) |
| **C** | `src/system.py` `get_required_mem_capacity` / `get_capacity_breakdown` | `dgx-attacc` 일 때 KV cache → AttAcc HBM 측 (paper §7.1 가정). `kv_on_attacc()` helper, `kv_per_attacc` / `available_kv_attacc` / `max_batch_at_default_L_attacc` 필드 신규 | 0 (`kv_on_attacc()` False 면 기존 3-tuple bit-identical) |

### Fix C 적용 후 `capacity_regime` 의 정량 변화

Pre-Fix-C (현재 origin 의 legacy JSON): KV 가 GPU 측에 살았으므로 LLaVA 의 *48 GB GDDR6* 가 ceiling → max_batch 88 / 91.
Post-Fix-C (run_all.py 재실행 후): KV 가 AttAcc HBM 측 (8 GPU × 5 HBM × 16 GB = 640 GB 총량, per-device 약 80 GB) → max_batch 약 **197 / 209**.

이는 **paper-correct** 값. 기존 88/91 은 우리 simulator 의 capacity 모델 버그였음 (KV 를 GPU 에 둠). 결과적인 narrative 는 변하지 않음:
- LLaVA 197 / 209 ≪ Qwen3-VL-4B 836+ → **여전히 capacity-bound vs throughput-bound 분류 유지**
- 단, paper Fig 의 X 축 값만 88→197 등으로 corrected. 이전 deck 의 88/91 narrative 는 정성적 메시지로는 OK 하나 *수치는 갱신* 권장.

---

## 환경 점검 (Step 0)

| 항목 | 명령 | 기대 |
|---|---|---|
| GPU 인식 | `nvidia-smi --query-gpu=name --format=csv,noheader` | `NVIDIA RTX A6000` / `NVIDIA H100 80GB` |
| hw_detect 작동 | `python 260511_additional_exp/shared/hw_detect.py` | `detected host = A6000` 또는 `H100`, sim gpu / interface 출력 |
| Ramulator2 binary | `ls attacc_simulator/ramulator2/ramulator2*` | 파일 존재 |
| vLLM | `python -c "import vllm; print(vllm.__version__)"` | A6000 노드 = `0.17.0` (실제 확인됨). H100 노드 = vLLM 0.17+ 권장 |
| 모델 캐시 | `ls ~/.cache/huggingface/hub` | qwen / llava / internvl prefix |
| LLM 회귀 | `python 260511_additional_exp/tier1_simulator/upstream_baseline.py` | 7 LLM 모두 `status: ok` |

`upstream_baseline` 결과가 *기존 JSON 과 bit-identical* 이면 Fix A+B+C 의 LLM-side 비침입 가정 확인. 차이 있으면 *즉시 중단*하고 Fix 분기 게이트 점검.

### H100 노드 첫 실행 시 추가 점검

A6000 노드에서 이미 실행 완료된 상태에서 H100 노드를 처음 셋업할 때:

| 항목 | 명령 | 기대 |
|---|---|---|
| Ramulator2 빌드 | `cd ramulator2 && mkdir -p build && cd build && cmake .. && make -j && cp ramulator2 ../ramulator2` | binary 생성 (~5 분) |
| HF 모델 캐시 | `huggingface-cli login` (필요 시) + 첫 모델 로드 시 자동 download | 5 VLM ~ 100 GB |
| vLLM 호환성 | 0.17 의 V1 엔진은 `RequestOutput.metrics` 제거함 → calibration 의 `_measure_vllm` 가 wall-clock 2-pass 로 측정해야 함 (A6000 에서 이미 적용된 패치) | TTFT / ITL 값이 ≠ None |
| 메모리 여유 | H100 80 GB > A6000 48 GB 이므로 A6000 에서 OOM 이던 batch=128 cells 도 성공 가능 | LLaVA-1.5 batch=128 셀이 `ok` 또는 `oom` (capacity 한계) |
| 디스크 공간 | per-host JSON 까지 합치면 results/ ≈ 200 MB | `df -h .` |

---

## 한 번에 돌리는 워크플로 — `run_all.py`

`run_all.py` 가 다음 순서로 실행:

| Step | 스크립트 | phase tag | 의존 | 의미 |
|---|---|---|---|---|
| 0 | `tier1_simulator/upstream_baseline.py` | `regression` | — | LLM bit-identical 회귀 |
| 1 | `calibration/run_calibration.py` | `phase1` | vLLM, HF 모델 | sim vs vLLM 정확도 (batch 1-128) |
| 2 | `tier1_simulator/vit_recalibration.py` | `phase1` | — | legacy s_corr 비교 (7.9× → 0.9-1.3× 예상) |
| 3 | `tier1_simulator/multi_vlm_full_sim.py` | `phase3` | Ramulator2 | 5 VLM speedup matrix 갱신 |
| 4 | `tier2_simulator/slo_throughput.py` | `phase2` | Ramulator2 | SLO throughput (B4 의 prerequisite) |
| 5 | `tier1_simulator/vlm_vs_llm_pair.py` | `phase2` | Ramulator2 | B1 — LLM vs VLM 의 speedup delta |
| 6 | `tier1_simulator/prefill_decomp_vlm.py` | `phase2` | Ramulator2 | B2 — prefill vs decode 기여 분해 |
| 7 | `tier1_simulator/visual_token_scaling.py` | `phase2` | Ramulator2 | B3 — visual token 수 sweep |
| 8 | `tier1_simulator/capacity_framing.py` | `phase2` | Step 4 의 JSON | B4 — SLO throughput 의 batch×ITL 분해 |
| 9 | `tier2_simulator/roofline_per_vlm.py` | `phase3` | — | Roofline 그림 데이터 |
| 10 | `tier2_simulator/capacity_regime.py` | `phase3` | — | Post-Fix-C 의 GPU+AttAcc 양면 capacity 표 |

옵션:
- `--skip phase3` — figure refresh 빼고 (multi_vlm / roofline / capacity_regime). 단 slo_throughput 은 phase2 라 그대로 실행
- `--skip phase2` — gain 측정 빼고
- `--skip regression` — upstream_baseline 빼고
- `--continue-on-error` — 첫 실패에서 멈추지 않음

전체 walltime ~ 30-60 분 (vLLM 5 모델 로딩 ~15 분 + 시뮬 200+ 회).

---

## 출력 위치

모든 결과 JSON 은 `attacc_simulator/260511_additional_exp/results/` 안에 저장.

**Per-host JSON 컨벤션 (R16)**: HW-dependent 결과는 `<name>.json` (latest 가 overwrite) 와 `<name>_<host>.json` (per-HW 보존) 둘 다 저장. 양쪽 HW 에서 실행 후 두 종류 모두 git push 하면 A6000 / H100 결과가 공존.

| JSON | per-host suffix? | 의미 |
|---|---|---|
| `calibration_<a6000\|h100>.json` | ✓ (이미 host suffix) | Phase 1 sim vs vLLM 매트릭스 |
| `calibration_cross_hw.json` | — | A6000 + H100 병합 |
| `vit_recalibration[_<host>].json` | ✓ | s_corr / g_corr 5 VLM, legacy 비교 용 |
| `multi_vlm_full_sim[_<host>].json` | ✓ | 5 VLM × 3 batch speedup matrix |
| `slo_throughput[_<host>].json` | ✓ | SLO=30 ms/tok max batch, ITL, throughput |
| `vlm_vs_llm_pair[_<host>].json` | ✓ | LLM↔VLM speedup delta (3 pair × 7 batch) |
| `prefill_decomp_vlm[_<host>].json` | ✓ | 모델별 prefill / decode / e2e speedup + 기여% |
| `visual_token_scaling[_<host>].json` | ✓ | image_size 별 visual_tokens, speedup |
| `capacity_framing[_<host>].json` | ✓ | SLO throughput 의 batch_ratio × ITL_ratio 분해. **HW-dependent** — 입력 slo_throughput_<host>.json 이 host 별이라 분해 결과도 host 별. v5 부터 per-host suffix 저장 |
| `roofline_per_vlm[_<host>].json` | ✓ | layer AI 분류 + host ridge |
| `capacity_regime[_<host>].json` | ✓ | **Post-Fix-C** GPU+AttAcc 양면 capacity, max_batch |

### ⚠ 현재 `results/` 의 R16 per-host 진척 상황 (2026-06-04)

R16 (per-host JSON save) 코드가 적용된 후 *실측 재실행* 된 결과는 **`calibration_a6000.json` 한 개뿐**. 나머지 `results/*.json` 은 R16 이전 (≈ `8f598b2` ~ `3fea285`) 산출물 → host-agnostic 단일 파일만 존재. H100 노드에서 돌리기 전, 또는 A6000 결과 보존을 원하면 아래 스크립트를 R16 코드로 **A6000 에서 재실행** 해서 `*_a6000.json` 도 생성해두는 게 좋음:

- `multi_vlm_full_sim.py`, `slo_throughput.py`, `vlm_vs_llm_pair.py`, `prefill_decomp_vlm.py`
- `visual_token_scaling.py`, `vit_recalibration.py`, `roofline_per_vlm.py`, `capacity_regime.py`
- `capacity_framing.py` (v5 부터 per-host save)
- `ablation_contribution.py`

### ⚠ `calibration_a6000.json` 은 R15 이전 stale 결과

현재 `results/calibration_a6000.json` 의 cells 64/64 가 `actual_lin_delta_vs_target ≈ 98` + `visual_tokens_estimated == 0` 으로 R15 (tokenizer 기반 prompt sizing + visual_tokens telemetry) **이전** 산출물. 코드 (`run_calibration.py`) 는 이미 R15 fix 가 들어가 있지만 *결과 JSON 이 그 fix 적용 전* 의 것. 따라서:

- **이 파일을 paper-grade Lin=X anchor 로 인용 금지**.
- H100 에서 돌리거나 A6000 에서 재실행 후 새 결과로 덮어쓸 것.

### ⚠ B1 / B2 의 LLaVA-Next-Mistral lin invalid (v5 fix)

v4 까지 B1 / B2 의 LLaVA-Next-Mistral cell 은 `lin=704` 였으나 `compute_visual_tokens` 가 1776 (img=336) / 2928 (img=672) 을 줘서 *lin ≤ visual_tokens* invalid. v5 부터:

- `vlm_vs_llm_pair.py` 의 (Mistral-7B ↔ LLaVA-Next-Mistral-7B, 336): **lin 704 → 1856**
- `prefill_decomp_vlm.py` 의 (LLaVA-Next-Mistral-7B, 672): **lin 704 → 3008**

이 두 값은 calibration 의 LLaVA-Next-Mistral (336, 1856) / (672, 3008) cell 과 일치. 기존 `vlm_vs_llm_pair.json` / `prefill_decomp_vlm.json` 의 LLaVA-Next-Mistral row 는 invalid 였으니 폐기 + 재실행 필요.

---

## 검증 체크리스트

각 실험 완료 후 확인:

### Sanity (실패 시 코드 / 실행 환경 의심)

- [ ] Step 0 — `upstream_baseline.json` 의 7 LLM s_time/g_time 이 origin 의 값과 일치 (bit-identical 회귀)
- [ ] Step 1 — `calibration_<hw>.json` 의 모든 cell `status == "ok"` 또는 `oom` / `lin_below_visual_tokens`. 의도하지 않은 `error` 없음
- [ ] Step 1 — 각 cell 의 `vllm.actual_lin_tokens_p50` vs `lin` 목표 차이 (`actual_lin_delta_vs_target`) 가 ±5% 안 (예: lin=704 cell 의 actual 이 670-740 사이). 5% 이상 벗어나면 paper 의 "Lin=X" 주장이 약해짐 → `_make_prompt_for_lin` 의 iter 수 / phrase chunk 조정 필요
- [ ] Step 4 — `slo_throughput.json` 의 GPU only 컬럼이 batch 4-16 에서 SLO 만족 ceiling 잡힘
- [ ] Step 8 — `capacity_framing.json` 의 `decomposition_delta_pct ≤ 5%` (batch_ratio × itl_ratio 분해 sanity)

### 검증할 가설 (실패가 paper story 자체에 대한 정보가 됨 — 자동 fail 처리 금지)

A6000 기존 실행에서 *이미 반대로 나온* 항목이 있음. H100 main 에서 같은 결과가 나오면 paper 방향 자체를 재고해야 하므로, 체크리스트의 통과 / 실패 모두 분석 대상이다.

- [ ] Step 1 — 5 VLM 평균 `s_corr ∈ [0.85, 1.20]`
  - 현재 A6000: Qwen2.5-VL / LLaVA-1.5 가 `s_corr < 0.5` (sim over-predicts). H100 에서도 비슷하면 vLLM 0.17 FlashAttn 효과 → ViT cost model 재조정 필요
- [ ] Step 2 — `vit_recalibration.json` 의 Qwen2.5-VL `s_corr` 7.9× → 1.0× 근처로 떨어짐
  - Fix A+B 적용 가설. H100 에서도 1.0× 근처면 model.py 수정이 valid; 아니면 Cosmos 측 별도 보정 필요
- [ ] Step 5 — `vlm_vs_llm_pair.json` 의 3 pair 중 최소 2 pair 에서 `delta > 0` (VLM > LLM speedup)
  - **현재 A6000: 3 pair 모두 `delta < 0`**. H100 에서도 음수면 "VLM 이 LLM 보다 AttAcc 의 강한 case" 라는 paper hook 폐기 → Topic 재정의
- [ ] Step 6 — `prefill_decomp_vlm.json` 의 VLM `prefill_speedup > 1` (visual token 으로 prefill 메모리 압박 → PIM 효과)
  - **현재 A6000: `prefill_speedup ≈ 0.95×`** (PIM 이 VLM prefill 에 도움 안 됨). H100 에서도 동일하면 Topic B 의 "diffusion + AR" angle 만으로 paper hook 정당화 필요
- [ ] Step 7 — `visual_token_scaling.json` 의 e2e speedup vs visual_tokens monotonic 증가
  - **현재 A6000: 반대로 감소** (visual_tokens ↑ → e2e_speedup ↓). H100 동일하면 *"고해상도 VLM 이 AttAcc 의 strong case"* 가설 폐기
- [ ] Step 10 — `capacity_regime.json` 의 LLaVA 가 capacity-bound (max_batch < 250) 유지. Fix-C 후 수치는 88/91 → 197/209 로 갱신됨 (paper-correct)

Cross-HW:
- [ ] `cross_hw_compare.py` 출력의 각 cell `consistency_delta_pct ≤ 10%`. A6000 ↔ H100 의 절대 latency 는 다르지만 simulator-vs-vLLM *비율* 은 같아야 함. 두 host 사이의 비율 deviation 이 크면 hw_detect / interface mapping 검토

---

## 시뮬레이터 코드 검증 (모델 수정 review 가 진행 중일 때)

| 호출 | 기대 |
|---|---|
| `system.kv_on_attacc()` — `dgx` 시스템 | `False` |
| `system.kv_on_attacc()` — `dgx-attacc` 시스템 | `True` |
| `system.get_required_mem_capacity(b, lin, lout)` — `dgx`, batch=54, lin=2048 | `(weight, kv_total, temp)` 3-tuple, kv > 0 |
| `system.get_required_mem_capacity(b, lin, lout)` — `dgx-attacc` | `(weight, 0, temp)`, kv on GPU = 0 |
| `system.get_attacc_kv_capacity(...)` — `dgx-attacc` | KV bytes (양수) |
| `system.get_capacity_breakdown(...)` 키 — `dgx` | `weight_per_gpu`, `kv_per_gpu`, `available_kv`, `max_batch_at_default_L` |
| `system.get_capacity_breakdown(...)` 키 — `dgx-attacc` | 추가 `kv_per_attacc`, `available_kv_attacc`, `attacc_capacity_total`, `max_batch_at_default_L_attacc`. `max_batch_at_default_L` = AttAcc-side 의 값 |

---

## Phase 2 신규 스크립트 명세 (참고)

| ID | 스크립트 | 입력 매트릭스 | 출력 핵심 필드 |
|---|---|---|---|
| B1 | `vlm_vs_llm_pair.py` | 3 pair × 7 batch × 2 systems (`dgx`, `dgx-attacc`) | `speedup_llm`, `speedup_vlm`, `delta`, `delta_pct_of_llm` |
| B2 | `prefill_decomp_vlm.py` | 5 VLM × 3 batch × 2 systems | `prefill_speedup`, `decode_speedup`, `e2e_speedup`, `prefill_contrib_pct` |
| B3 | `visual_token_scaling.py` | 2 VLM × 4 image_size × 2 batch × 2 systems, `lin = visual_tokens + 64` 동적 | `visual_tokens`, `e2e_speedup`, `prefill_speedup` |
| B4 | `capacity_framing.py` | `slo_throughput.json` 재가공 | `batch_ratio`, `itl_ratio`, `speedup_total`, `decomposition_delta_pct` |

---

## 알려진 리스크 / FIX 이력

| ID | 항목 | 상태 |
|---|---|---|
| R1 | vLLM 프롬프트 길이 시뮬 lin 불일치 | FIXED — `_make_prompt_for_lin(target_lin, visual_tokens)` 동적 산출 |
| R2 | llm_cache 가 5 VLM 누적 → OOM | FIXED — hf_id 그룹별 load → 측정 → `del llm` + `cuda.empty_cache()` |
| R3 | `*1000` 으로 raw ms 1000× 부풀어짐 | FIXED — sim_runner 가 이미 ms 반환, `*1000` 제거 |
| R4 | visual_token_scaling 의 LIN=704 고정 | FIXED — `lin_dynamic = visual_tokens + 64` |
| R5 | Fix C 미구현 | FIXED — system.py system-aware (`kv_on_attacc()`) |
| R6 | `capacity_framing` < `slo_throughput` 순서 | FIXED — slo_throughput 을 phase2 로 이동 |
| R7 | Fix C 가 `max_batch_at_default_L=0` 으로 capacity_regime 깨뜨림 | FIXED — `max_batch_at_default_L` 가 system limiting batch 로 동작 |
| R8 | vLLM 입력에 모델별 placeholder 누락 | FIXED — `vllm_helpers.make_image_input(hf_id, prompt, image)` 사용 |
| R9 | `lin <= visual_tokens` cell 조용히 통과 | FIXED — `status: lin_below_visual_tokens` 반환 + LLaVA-Next 336 lin=1856 으로 조정 |
| R10 | `--skip phase3` 시 stale slo_throughput | FIXED — slo_throughput → phase2 |
| R11 | `capacity_regime.py` 가 GPU 측 필드만 출력 | FIXED — `kv_per_attacc_mib`, `available_kv_attacc_mib`, `kv_resident_side` 추가 |
| R12 | cross_hw_compare key 가 (label, image_size, batch) 만이라 lin 다른 cell 잘못 병합 | FIXED — `(label, image_size, lin, batch)` 4-tuple key |
| R13 | `capacity_regime.json` 이 Fix C 이전 format 으로 남음 | FIXED — 로컬 재실행해서 새 필드들 (kv_resident_side / kv_per_attacc_mib / max_batch_attacc_side) 채워서 commit |
| R14 | `get_capacity_breakdown` docstring 이 "GPU side" 라고 적혔지만 실제로 `max_batch_at_default_L` 는 system limiting batch 로 overwrite | FIXED — docstring 갱신 |
| R15 | vLLM 프롬프트 길이 맞추기가 "영문 6 chars ≈ 1 token" 근사라 paper 의 "Lin=X" 정확성 약함 | FIXED — `_get_tokenizer(hf_id)` 로 AutoTokenizer 캐시 + 반복 prompt 사이즈 조정 (±2 token), JSON 에 `actual_lin_tokens_p50` / `actual_lin_delta_vs_target` 기록. tokenizer 못 받으면 char heuristic fallback |
| R16 | Phase 2/3 스크립트 9 개가 `gpu="A6000"` / `interface="NVLINK_BRIDGE"` / `GPUType.A6000` 하드코딩 → H100 노드에서 같은 코드가 A6000 sim config 로 돔 | FIXED — `shared/hw_detect.py` (`detect_host` / `sim_gpu_tag` / `sim_interface_tag` / `gputype_enum`) 신규 + 9 개 스크립트 (vit_recalibration, multi_vlm_full_sim, vlm_vs_llm_pair, prefill_decomp_vlm, visual_token_scaling, ablation_contribution, slo_throughput, capacity_regime, roofline_per_vlm) 자동 분기. 또한 7 개 결과 JSON 이 `<name>_<host>.json` 추가 사본 저장 |

---

## 부록 — 기존 21 실험 vs 본 runbook 의 관계

본 runbook 은 21 실험을 *재실행 / 재가공* 하는 것이고, 새 실험 4 종 (B1-B4) 를 *추가* 하는 것임.

- 기존 21 실험의 *코드* (스크립트) 는 그대로 유지
- 결과 JSON 은 Fix A+B+C 적용 후 재실행 시 *덮어씌워짐*
- `capacity_regime.json` 만은 88/91 → 197/209 처럼 *정량적으로 바뀜* (R7 의 paper-correct 갱신)
- 나머지는 정량 변화 < 5% (Fix B 의 +22 ms 가 양쪽에 같이 적용되므로 ratio 거의 그대로)

---

## 갱신 history

- **2026-06-01 v1** — 초안
- **2026-06-01 v2** — risks 부록 추가 (R1-R6)
- **2026-06-01 v3** — R7-R11 반영. 가이드 명료화 (TL;DR, Step 0 환경 점검, 시뮬레이터 코드 검증 표, capacity_regime 의 88→197 정량 변화 명시)
- **2026-06-04 v4** — R12-R16 반영. H100 readiness — `hw_detect.py` 신규 + 9 스크립트 자동 분기 + per-host JSON 사본. TL;DR 에 HW 자동 매핑 표 (A6000/H100/A100 → sim gpu/interface/ridge) 추가. H100 노드 첫 실행 시 추가 점검 표 추가. 출력 위치 표를 per-host suffix 컨벤션 반영하여 갱신.

---

# 실행 결과 & 분석 — 2026-06-02 (RTX A6000, GPU1)

> 본 절은 260601 runbook 을 실제 실행한 결과 정리 + 분석. 실행 환경이 runbook 가정(vLLM 0.7.3, DGX/H100)과 달라 일부 코드 수정이 필요했고, 그 내용/근거를 함께 기록한다.

## 0. 실행 환경

| 항목 | 값 |
|---|---|
| 일시 | 2026-06-02 (UTC) |
| GPU | NVIDIA RTX A6000 49 GB ×2 — **GPU1=calibration, GPU0=r9 (병렬)** |
| Driver | 595.71.05 |
| Python / Torch | 3.10.14 / 2.10.0+cu128 (CUDA 12.8) |
| vLLM | **0.17.0** (runbook 가정 0.7.3) |
| transformers | 4.57.6 |
| Repo commit | `8f598b2` (R15) + 본 실행의 수정들 |
| HF_HOME | `/131_data/geeho/minsik/tmp/run_260601/hf_cache` (모델 5종 + MMMU-Pro 전부 tmp) |
| Ramulator2 | 로컬 빌드 (`b7c7027` + PIM 패치), `ramulator2/ramulator2` |

## 1. 완결성 점검 (무엇이 돌았고 무엇이 빠졌나)

| Step | 산출물 | 상태 |
|---|---|---|
| 0 upstream_baseline (LLM 회귀) | `upstream_baseline.json` | ✅ |
| 1 calibration (sim vs vLLM, 5 VLM) | `calibration_a6000.json` | ✅ (2-pass 실측) |
| 2 vit_recalibration (legacy 비교) | `vit_recalibration.json` | ✅ (legacy 측정치 재사용) |
| 3 multi_vlm_full_sim | `multi_vlm_full_sim.json` | ✅ |
| 4 slo_throughput | `slo_throughput.json` | ✅ |
| 5 B1 vlm_vs_llm_pair | `vlm_vs_llm_pair.json` | ✅ |
| 6 B2 prefill_decomp_vlm | `prefill_decomp_vlm.json` | ✅ |
| 7 B3 visual_token_scaling | `visual_token_scaling.json` | ✅ (src-path 버그 수정 후) |
| 8 B4 capacity_framing | `capacity_framing.json` | ✅ |
| 9 roofline_per_vlm | `roofline_per_vlm.json` | ✅ |
| 10 capacity_regime | `capacity_regime.json` | ✅ |
| r9 MMMU-Pro 실측 (5 VLM) | `r9_*_mmmu_tp1.json` | ✅ (TTFT/ITL/E2E/energy) |

**빠진 것 (불가/옵션)**
- **H100 측 calibration + `cross_hw_compare`** → H100 하드웨어 없음(이 노드 A6000만). cross-HW 비교 불가.
- **r10 concurrent serving** → run_all 비포함(옵션). 미실행.
- **InternVL3 calibration vLLM** → 초기엔 placeholder 버그로 실패했으나 **수정 후 정상 측정됨**(아래 §3).

## 2. 실행 중 발견/수정한 문제 (환경 차이 기인)

| # | 문제 | 근본 원인 | 조치 |
|---|---|---|---|
| F1 | `run_all.py` 모든 step `No such file` | `ROOT=HERE.parents[2]` 가 repo 한 단계 위를 가리킴 | `parents[1]` 로 수정 |
| F2 | Ramulator2 "NOT FOUND" | F1 과 동일(엉뚱 경로 조회) | F1 수정으로 해결 |
| F3 | Step 7 `ModuleNotFoundError: src` | `visual_token_scaling.py` 의 `parents[2]` off-by-one | `parents[1]` 로 수정 |
| F4 | calibration 한 모델 실패 시 전체 크래시(저장 전) | vLLM 예외가 `main()` 밖으로 전파 | 셀 단위 try/except 가드 |
| F5 | **calibration/r9 TTFT·ITL 전부 None** | **vLLM 0.17(V1 엔진)이 `RequestOutput.metrics` 제거** (`VLLM_USE_V1` 토글도 없음) | **wall-clock 2-pass 측정으로 전환**(아래) |
| F6 | InternVL3 vLLM 크래시(`Failed to apply prompt replacement`) | 이미지 placeholder 를 `<image>` 로 줌 — InternVL*-hf 는 `<IMG_CONTEXT>` | placeholder 교체(vllm_helpers + r9) |
| F7 | LLaVA-Next img=672 8셀 `prompt(5937) > max_model_len(4096)` | calibration `_load_llm` 이 4096 하드코딩 | LLaVA-Next 만 8192 로 |

### F5 상세 — "metrics 뽑는 vLLM 버전" 결론
- 모델 아키: `Qwen3VLForConditionalGeneration`, `InternVLForConditionalGeneration` → **최신 vLLM(=0.17) 필요**
- 0.17 = V1 전용, `out.metrics` 제거 → **native TTFT/ITL 불가**
- ∴ **"native metrics + 5모델 지원"을 동시에 만족하는 vLLM 버전은 없음**(신모델이 0.17 강제 ↔ 0.17은 metrics 없음).
- 정확 측정법 = 버전 무관 **wall-clock 2-pass**: `max_tokens=1` → TTFT, full(`min=max=LOUT`) → E2E, `ITL=(E2E−TTFT)/(LOUT−1)`. warmup 1회 후 측정. 시뮬레이터의 prefill(`s`)·decode(`g`) 정의와 정확히 대응.

## 3. Step 1 — Calibration (sim vs vLLM 실측, 2-pass)

`calibration_a6000.json` — 5 VLM × image_size × batch(1–128), `s_corr = meas_TTFT / sim_s`, `g_corr = meas_ITL / sim_g`.

**64 셀 전부 측정 성공** (sim ok ×64, vLLM ok ×64). image_size 별 `min..median..max` (batch 1–128 스윕):

| 모델 | img | 셀 | s_corr (min..med..max) | g_corr (min..med..max) |
|---|---|---|---|---|
| Qwen3-VL-4B | 336 | 8 | 0.12 .. 0.28 .. 0.62 | 0.25 .. 0.73 .. 1.01 |
| Qwen3-VL-4B | 672 | 8 | 0.08 .. 0.20 .. 0.42 | 0.26 .. 0.74 .. 1.02 |
| Qwen2.5-VL-7B | 336 | 8 | 0.07 .. 0.21 .. 0.47 | 0.33 .. 0.78 .. 0.96 |
| Qwen2.5-VL-7B | 672 | 8 | 0.04 .. 0.12 .. 0.31 | 0.33 .. 0.79 .. 0.96 |
| LLaVA-1.5-7B | 336 | 8 | 0.07 .. 0.20 .. 0.44 | 0.35 .. 0.70 .. 0.90 |
| LLaVA-Next-Mistral-7B | 336 | 8 | 0.04 .. 0.07 .. 0.19 | 0.13 .. 0.54 .. 0.84 |
| LLaVA-Next-Mistral-7B | 672 | 8 | 0.03 .. 0.05 .. 0.11 | 0.09 .. 0.48 .. 0.82 |
| InternVL3-8B-hf | 448 | 8 | 0.08 .. 0.19 .. 0.45 | 0.33 .. 0.78 .. 0.96 |

**해석**
- **s_corr < 1 (대부분 0.05–0.45)** — 시뮬레이터가 실측 TTFT 대비 prefill 을 **과대예측**. batch=1 에서 최고(0.4–0.6), batch↑ 일수록 급감(sim prefill 이 실측보다 batch 에 더 가파르게 증가). LLaVA-Next 가 가장 낮음(0.03–0.19, anyres 로 sim prefill 이 특히 큼).
- **g_corr 중앙값 0.5–0.8, batch=1 에서 ≈1.0** — decode 는 저batch 에서 sim 과 거의 일치, batch↑ 시 실측 ITL 이 sim 보다 천천히 증가해 비율 하락.
- **runbook 목표 `s_corr∈[0.85,1.20]` 미달** — Fix A/B 이후에도 (vLLM 0.17 / A6000 / wall-clock 2-pass 기준) **prefill 모델이 여전히 과대예측**. 측정 방식이 runbook 가정(native TTFT)과 달라 절대 비교는 주의(§11). 추세상 prefill 항(특히 batch 스케일링·anyres)의 재보정 필요.

## 4. Step 3 — Multi-VLM speedup matrix (AttAcc vs GPU, e2e)

| 모델 | img | e2e speedup (b=1 / 4 / 8) |
|---|---|---|
| Qwen3-VL-4B | 672 | 3.22 / 2.78 / 2.60 |
| Qwen2.5-VL-7B | 672 | 4.87 / 3.38 / 2.55 |
| InternVL3-8B-hf | 448 | 5.32 / 3.90 / 2.98 |
| LLaVA-1.5-7B | 336 | 2.73 / 2.40 / 2.21 |
| LLaVA-Next-Mistral-7B | 672 | 3.34 / 2.93 / 2.83 |

→ AttAcc(dgx-attacc)가 GPU-only 대비 e2e **2.2–5.3×**. batch↑ 일수록 speedup↓ (decode 비중 감소 + prefill 비가속).

## 5. Step 6 (B2) — Prefill vs Decode 분해

| 모델 | decode speedup | prefill speedup | e2e speedup |
|---|---|---|---|
| Qwen3-VL-4B | 3.76–4.10× | 0.93–0.97× | 2.77–3.30× |
| Qwen2.5-VL-7B | 4.88–6.44× | 0.94–1.00× | 2.83–5.10× |
| InternVL3-8B-hf | 4.88–6.44× | 0.91–1.00× | 3.30–5.42× |
| LLaVA-1.5-7B | 2.94–3.02× | 0.86–0.92× | 2.36–2.78× |
| LLaVA-Next-Mistral-7B | 4.17–4.54× | 0.95–1.00× | 2.94–4.03× |

→ **AttAcc 이득은 전적으로 decode 에서 발생**(3.8–6.4×). prefill 은 ≤1.0×(비가속, 약간의 오버헤드). `prefill_contrib_pct < 0`.

## 6. Step 7 (B3) — Visual token 민감도  ⚠️ 가설과 반대

| 모델 | img / visual_tokens | e2e speedup (b=1) |
|---|---|---|
| Qwen2.5-VL-7B | 336 / 144 | **5.63** |
| Qwen2.5-VL-7B | 672 / 576 | 5.15 |
| Qwen2.5-VL-7B | 1008 / 1296 | 4.01 |
| Qwen2.5-VL-7B | 1344 / 2304 | **2.89** |
| LLaVA-Next-Mistral | 336–1344 / 1776→2928 | 3.87 → 3.61 (anyres 상한 2928 에서 평탄) |

→ runbook 가설("visual token↑ → speedup↑")과 **정반대**: visual token 이 늘수록 e2e speedup **감소**. 이유는 §9 분석.

## 7. Step 5 (B1) — LLM↔VLM pair  ⚠️ 가설과 반대

| pair | delta = vlm_speedup − llm_speedup |
|---|---|
| Vicuna-7B → LLaVA-1.5-7B | −0.044 … −0.035 |
| Mistral-7B → LLaVA-Next-Mistral | −0.166 … −0.134 |
| Qwen3-4B → Qwen3-VL-4B | −0.551 … −0.208 |

→ **모든 pair 에서 delta < 0** — VLM 이 LLM 백본 대비 AttAcc speedup 이 오히려 약간 **작음**. runbook 기대(≥2 pair delta>0)와 반대. 이유는 §9.

## 8. Step 4 / 8 / 9 / 10 — Throughput·Capacity·Roofline

- **B4 capacity_framing**: SLO=30ms/tok 에서 AttAcc 가 batch 4× × ITL 2.3× = **throughput ~9.2×** (분해 오차 ~0%). Qwen3-VL throughput_attacc ≈ 6300 tok/s vs GPU 687 tok/s.
- **capacity_regime (Fix C, KV on AttAcc)** max_batch (A1 TP=1): Qwen3-VL **836**, Qwen2.5-VL **1802**, InternVL3 **2931**, **LLaVA-1.5 197 / LLaVA-Next 209**. → LLaVA 계열만 capacity-bound(<250), 나머지는 throughput-bound. (runbook 의 88/91→197/209 정정과 일치)
- **roofline**: prefill 의 `qkv`/`ffn` 은 compute-bound(AI≈430–480, PIM 무이득), `score`/`context`(attention) 만 memory-bound(AI≈100, PIM 타깃). decode 의 attention 이 PIM 핵심 타깃임을 확인.

## 9. MMMU-Pro 실데이터 측정 (r9, 5 VLM, n=32, 실측)

실제 MMMU-Pro "standard(4 options)" 32문항(실이미지+멀티초이스), wall-clock 2-pass, GPU0.

| 모델 | seq_in p50 | TTFT p50/p95 (ms) | ITL p50 (ms/tok) | E2E p50 (ms) | avg W | J/req | J/tok |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 334 | 154.6 / 214.6 | 21.4 | 2875.6 | 284 | 815.8 | 6.37 |
| Qwen3-VL-4B | 270 | 109.4 / 154.9 | 13.0 | 1766.8 | 275 | 487.0 | 3.80 |
| LLaVA-1.5-7B | 620 | 153.7 / 183.0 | 19.9 | 2678.4 | 281 | 752.7 | 5.88 |
| LLaVA-Next-Mistral-7B | 1979 | 390.1 / 514.6 | 19.8 | 2901.5 | 277 | 804.6 | 6.29 |
| InternVL3-8B-hf | 1586 | 369.5 / 749.6 | 19.6 | 2885.8 | 280 | 810.1 | 6.33 |

→ TTFT 는 seq_in(visual+text)에 비례(LLaVA-Next 1979토큰 → TTFT 390ms). ITL 은 모델 크기에 따름(Qwen3-VL 4B 가 13ms 로 최저). Qwen3-VL-4B 가 E2E·에너지 모두 최효율.

## 10. 종합 분석 — corrected 시뮬레이터가 말하는 것

1. **AttAcc 이득의 원천은 decode(KV-attention) 단 하나.** roofline 상 decode 의 score/context 만 memory-bound → PIM 가속(decode 3.8–6.4×). prefill 의 qkv/ffn 은 compute-bound → 비가속(prefill ≤1.0×).
2. **그래서 visual token 이 많을수록 e2e 이득이 줄어든다(B3 ↓)**, 그리고 **VLM 이 LLM 백본보다 이득이 작다(B1 delta<0)**. 둘 다 같은 메커니즘: 비전 토큰은 *prefill*(비가속부)을 키우므로 e2e 에서 가속부(decode)의 비중을 희석. → runbook 의 두 가설("visual↑→gain↑", "VLM>LLM")은 **본 corrected 시뮬레이터에선 성립하지 않음**(정직한 반증).
3. **다만 lout=128 같은 디코드-중심 워크로드에선 여전히 e2e 2.2–5.3× 이득** + capacity 측면에서 KV-on-AttAcc 로 max_batch 가 크게 늘어(LLaVA 197/209, 그 외 800–2900) **throughput ~9× (B4)**. 즉 VLM 서빙의 실이득은 "VLM 고유 추가 gain" 보다 **decode/throughput/capacity** 축에서 온다.
4. **Calibration(§3)**: 절대 prefill 은 sim 이 실측 TTFT 대비 과대예측(s_corr<1) 경향, decode 는 비교적 근접. (상세 수치 §3 표) — runbook 목표 `s_corr∈[0.85,1.20]` 와의 격차는 §3 에서 논의.

## 11. 한계 / 주의

- **TTFT/ITL = wall-clock 2-pass**(vLLM 0.17 native metrics 부재). 스케줄링·파이썬 오버헤드가 batch=1 소형값에 수 ms 포함될 수 있음(추세·비율은 유효).
- **단일 HW(A6000)**: cross-HW(H100) 일관성 검증 미수행.
- vit_recalibration 의 measured 값은 **이전 캠페인 재사용**(이번 vLLM 실측 아님) — legacy 비교용.
- 시뮬레이터 system="dgx"/"dgx-attacc" 는 8×A100 가정 모델, 실측은 A6000×1 → s_corr 는 동일 A6000 sim↔A6000 실측 비율로 해석.

## 갱신 history (이어서)
- **2026-06-02** — A6000 GPU1/GPU0 병렬 실행 결과 추가. vLLM 0.17 대응(2-pass TTFT/ITL), InternVL3 placeholder·LLaVA-Next max_model_len·ROOT/src-path 버그 수정. r9 5VLM 실측, calibration 5VLM, sim 11 step 전부 갱신.
