# 260601 실험 runbook — VLM Simulator Calibration & Gain 측정

**Last update**: 2026-06-01 v3 (R7-R11 반영)
**Owner**: Minsik
**Repo**: `attacc_simulator/` (origin `minsik-choi126/attacc_simulator_vlm`, branch `main`)

## TL;DR — 노드에 도착해서 무엇을 하면 되는가

```bash
cd attacc_simulator
git pull                                                          # 최신 받음
ls ramulator2/ramulator2* 2>/dev/null || bash set_pim_ramulator.sh # 없으면 빌드
nvidia-smi --query-gpu=name --format=csv,noheader                 # A6000 / H100 확인

# 한 줄로 전체 실험
python 260511_additional_exp/calibration/run_all.py

# 끝난 후
ls 260511_additional_exp/results/calibration_*.json
```

A6000 노드에서 한 번, H100 노드에서 한 번 돌린 후, 한 곳에서:

```bash
python 260511_additional_exp/calibration/cross_hw_compare.py --save
```

→ `results/calibration_cross_hw.json` + 콘솔 표.

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
| Ramulator2 binary | `ls attacc_simulator/ramulator2/ramulator2*` | 파일 존재 |
| vLLM | `python -c "import vllm; print(vllm.__version__)"` | `0.7.3` (또는 호환) |
| 모델 캐시 | `ls ~/.cache/huggingface/hub` | qwen / llava / internvl prefix |
| LLM 회귀 | `python 260511_additional_exp/tier1_simulator/upstream_baseline.py` | 7 LLM 모두 `status: ok` |

`upstream_baseline` 결과가 *기존 JSON 과 bit-identical* 이면 Fix A+B+C 의 LLM-side 비침입 가정 확인. 차이 있으면 *즉시 중단*하고 Fix 분기 게이트 점검.

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

| JSON | 의미 |
|---|---|
| `calibration_a6000.json` / `calibration_h100.json` | Phase 1 sim vs vLLM 매트릭스 |
| `calibration_cross_hw.json` | A6000 + H100 병합 |
| `vit_recalibration.json` | s_corr / g_corr 5 VLM, legacy 비교 용 |
| `multi_vlm_full_sim.json` | 5 VLM × 3 batch speedup matrix |
| `slo_throughput.json` | SLO=30 ms/tok max batch, ITL, throughput |
| `vlm_vs_llm_pair.json` | LLM↔VLM speedup delta (3 pair × 7 batch) |
| `prefill_decomp_vlm.json` | 모델별 prefill / decode / e2e speedup + 기여% |
| `visual_token_scaling.json` | image_size 별 visual_tokens, speedup |
| `capacity_framing.json` | SLO throughput 의 batch_ratio × ITL_ratio 분해 |
| `roofline_per_vlm.json` | layer AI 분류 |
| `capacity_regime.json` | **Post-Fix-C** GPU+AttAcc 양면 capacity, max_batch |

---

## 검증 체크리스트

각 실험 완료 후 확인:

- [ ] Step 0 — `upstream_baseline.json` 의 7 LLM s_time/g_time 이 origin 의 값과 일치 (bit-identical 회귀)
- [ ] Step 1 — `calibration_<hw>.json` 의 모든 cell `status == "ok"` 또는 `oom` / `lin_below_visual_tokens`. 의도하지 않은 `error` 없음
- [ ] Step 1 — 5 VLM 평균 `s_corr ∈ [0.85, 1.20]`
- [ ] Step 1 — 각 cell 의 `vllm.actual_lin_tokens_p50` vs `lin` 목표 차이 (`actual_lin_delta_vs_target`) 가 ±5% 안 (예: lin=704 cell 의 actual 이 670-740 사이). 5% 이상 벗어나면 paper 의 "Lin=X" 주장이 약해짐 → `_make_prompt_for_lin` 의 iter 수 / phrase chunk 조정 필요
- [ ] Step 2 — `vit_recalibration.json` 의 Qwen2.5-VL `s_corr` 7.9× → 1.0× 근처로 떨어짐
- [ ] Step 4 — `slo_throughput.json` 의 GPU only 컬럼이 batch 4-16 에서 SLO 만족 ceiling 잡힘
- [ ] Step 5 — `vlm_vs_llm_pair.json` 의 3 pair 중 최소 2 pair 에서 `delta > 0` (VLM 이 LLM 보다 큰 speedup)
- [ ] Step 6 — `prefill_decomp_vlm.json` 의 VLM `prefill_contrib_pct > 0` (visual token 효과 확인)
- [ ] Step 7 — `visual_token_scaling.json` 의 speedup vs visual_tokens monotonic 증가
- [ ] Step 8 — `capacity_framing.json` 의 `decomposition_delta_pct ≤ 5%` (batch_ratio × itl_ratio 분해 sanity)
- [ ] Step 10 — `capacity_regime.json` 의 LLaVA 가 capacity-bound (max_batch < 250) 유지. 단 *수치는 88/91 → 197/209 로 갱신됨* (paper-correct)

Cross-HW:
- [ ] `cross_hw_compare.py` 출력의 각 cell `consistency_delta_pct ≤ 10%`

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
