# AttAcc Simulator Modification Plan — final_v1

> 모든 plan variant 통합. legacy/iteration history 없음. 수정사항 + 검증 + 타임라인만.

---

## 0. Locked Decisions

### 0.1 Deployment hardware (2 scenarios, first-class)

기본 H100 SXM5, per-GPU 80 GB HBM3 (5 stacks × 16 GB).
**Tensor Core dense (no sparsity)**: FP16 989.4 TFLOPS / FP8 1979 TFLOPS. (sparsity-enabled 2× = FP16 1979 / FP8 3958 — 우리 paper는 dense만 사용, 명시.)
HBM BW 3,352 GB/s.

| Scenario | GPUs | TP | NUM_ATTACC | NUM_HBM | Inter-GPU | PIM aggregate |
|---|---|---|---|---|---|---|
| **S1: H100 × 1** | 1 | TP = 1 | 1 | 5 | none | **18.1 TB/s** |
| **S2: H100 × 2** | 2 | TP = 2 | 2 | 5 | NVLink4 900 GB/s (HD 450) | **36.2 TB/s** |

PIM aggregate = 670.4 × 9 × 5 × num_attacc × 0.6 util.
PIM internal scale 9 (HBM3 power-constrained, paper §4.1).

**모든 5 모델이 single H100 80 GB에 들어감** (Qwen3-VL-4B 8 GB ~ InternVL3 16 GB). S1/S2 둘 다 정상 시나리오.

| 차이 | S1 (TP=1) | S2 (TP=2) |
|---|---|---|
| 장점 | inter-GPU 통신 0, latency 단순 | 큰 batch (capacity 2배), per-GPU compute 분담 |
| 단점 | 낮은 batch capacity, 단일 GPU compute bottleneck | NVLink4 all-reduce 매 layer (~0.46 ms/Qwen3-VL prefill) |

Paper repro (R2)는 별도 simulated DGX (`--ngpu 8 --num_attacc 8 --gpu A100a --interface NVLINK3`). Hardware idle.

### 0.2 Models in scope

5 in-framework (시뮬 대상):

| 모델 | LLM | n_q / n_kv / gqa | Visual tokens | KV/req @ default |
|---|---|---|---|---|
| Qwen3-VL-4B (Primary) | Qwen3 36L hdim=2560 | 32/8/4 | 441 (672²) | 80.0 MB |
| Qwen2.5-VL-7B | Qwen2.5 28L hdim=3584 | 28/4/7 | 576 (672²) | 40.4 MB |
| InternVL3-8B-hf | Qwen2.5-spec | 28/4/7 | 256 (448²) | 21.0 MB |
| LLaVA-1.5-7B | Vicuna 32L hdim=4096 | 32/32 (MHA) | 576 (336²) | 352 MB |
| LLaVA-Next-Mistral-7B | Mistral 32L hdim=4096 | 32/8/4 | ~2880 (Anyres) | 376 MB |

Baselines: Qwen3-4B, Vicuna-7B, Mistral-7B (text-only sanity), GPT-175B (R2 paper repro).
Out-of-scope: Llama-3.2-V (cross-attn arch), Qwen3.5 (hybrid linear attn) — paper observation only.

### 0.3 Per-GPU max batch (S1 vs S2)

H100 80 GB, weight + 2 GB activation buffer 가정. KV per req per GPU at default L (vis+text+gen).

| 모델 | weight FP16 | KV/req @ default L | **S1 max batch** (TP=1, full weight) | **S2 max batch** (TP=2, half weight + half KV) |
|---|---|---|---|---|
| Qwen3-VL-4B | 8 GB | 80 MB (S1) / 40 MB (S2) | ~875 | ~1850 |
| Qwen2.5-VL-7B | 14 GB | 40 MB / 20 MB | ~1600 | ~3500 |
| InternVL3-8B-hf | 16 GB | 21 MB / 11 MB | ~3000+ | ~6000+ |
| **LLaVA-1.5-7B (MHA)** | 13 GB | 352 MB / 176 MB | **~184** | **~406** |
| **LLaVA-Next-Mistral** | 14 GB | 376 MB / 188 MB | **~170** | **~372** |

S2 (TP=2)가 모든 모델에서 max batch 약 2× (capacity 2배 효과). C3 capacity argument: 모델별 15-18× 차이는 두 scenario 모두 동일.

### 0.4 Pipelining latency-mode eff_lat (S1 vs S2)

num_hbm=5 per AttAcc. T_softmax / T_GEMV ≈ 0.4 (paper §6.1).

| 모델 | n_kv | **S1 (TP=1)** per-GPU n_kv / dist / eff_lat | **S2 (TP=2)** per-GPU n_kv / dist / eff_lat |
|---|---|---|---|
| Qwen3-VL-4B | 8 | 8 / [2,2,2,1,1] / **0.80** | 4 / [1,1,1,1,0] / **0.57** |
| Qwen2.5-VL-7B | 4 | 4 / [1,1,1,1,0] / **0.57** | 2 / [1,1,0,0,0] / **0.29** ⚠️ |
| InternVL3-8B-hf | 4 | 4 / [1,1,1,1,0] / **0.57** | 2 / [1,1,0,0,0] / **0.29** ⚠️ |
| LLaVA-1.5-7B (MHA) | 32 | 32 / [7,7,6,6,6] / **0.91** | 16 / [4,4,3,3,2] / **0.80** |
| LLaVA-Next-Mistral | 8 | 8 / [2,2,2,1,1] / **0.80** | 4 / [1,1,1,1,0] / **0.57** |

**핵심 finding**:
- S1에서는 Qwen2.5-VL/InternVL3가 0.57로 그래도 절반 이상
- S2에서는 0.29로 떨어짐 (per-GPU 2 head만 남아 3 HBM idle)
- LLaVA-1.5 (MHA)은 S1/S2 둘 다 best (head 풍부)
- Single-batch 시나리오: S1이 더 우호적 (Qwen2.5-VL/InternVL3에 한해)

Batch ≥ 2 시 모든 케이스 eff_lat → 1.0 (head 분배 충분).

### 0.5 Critical interpretations (잊지 말 것)

- **Score MATMUL의 ramulator 호출 = full attn block** (score + softmax + context cycle 모두 포함). [trace_gen/gen_trace_attacc_bank.py `Attention()` L171-179](../pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py#L171-L179)
- **PIM context layer time = 0** (graph order placeholder, 의도된 설계)
- **PIM softmax layer time = 0** (이중 계산 차단, P1)
- **layer.m은 ramulator wrapper가 안 봄**. m-scaling은 외부에서 곱.
- **shape contract**: score Layer는 m=L, n=L, k=dhead. wrapper는 n=L, k=dhead로 해석.
- **GQA numOp 분기 (critical)**:
  - `Layer.numOp` = num_q_heads per AttAcc (GPU compute / FLOPs 정확)
  - `Layer.pim_numOp` = num_kv_heads per AttAcc (PIM trace 정확, KV read 지배)
  - Ramulator wrapper는 `pim_numOp` 사용 (없으면 numOp fallback). PIM attention time은 K read traffic dominant → n_kv 기반이 맞음 (RESULTS_ANALYSIS L178/L599 분석 일치).
  - Q transmission (comm_x2g_q) traffic 계산은 num_q_heads 기준 (interface는 Q 단위).
  - 즉: GPU FLOPs/compute → Q heads, PIM HBM traffic → KV heads, interface → Q heads.
- corrected E2 target:
  - **S1 (TP=1)**: E2E 1.58× ±20%, interface/PIM_compute ≈ 0.5-0.7 (Q transmission 큼, all-reduce 0)
  - **S2 (TP=2)**: E2E 1.53× ±20%, interface/PIM_compute ≈ 0.2-0.4 (Q split, NVLink4 all-reduce 추가)

---

## 1. Modifications (action list)

### M0 — Repo metadata cleanup (30 min)

- README + E4 scripts: DeepStack indices [8,16,24] → [5,11,17] 통일
- Qwen3.5 loader 코멘트: text-only AutoModel 명시 (decode-loop 시뮬 X)

### M1 — H100/NVLink4 + AttAcc argparse (1 hr)

`main.py` argparse 추가:
```python
parser.add_argument("--num_attacc", type=int, default=None)  # default = ngpu
parser.add_argument("--num_hbm", type=int, default=5)
parser.add_argument("--interface", type=str, default='NVLINK3',
                    choices=['NVLINK3','NVLINK4','PCIE4','PCIE5'])
parser.add_argument("--tp", type=int, default=None)         # default = ngpu
parser.add_argument("--max_L", type=int, default=2048)
```

defaults:
- **S1**: `--gpu H100 --ngpu 1 --num_attacc 1 --num_hbm 5 --interface NVLINK4 --tp 1 --max_L 2048`
- **S2**: `--gpu H100 --ngpu 2 --num_attacc 2 --num_hbm 5 --interface NVLINK4 --tp 2 --max_L 2048`
- **Paper repro (R2)**: `--gpu A100a --ngpu 8 --num_attacc 8 --num_hbm 5 --interface NVLINK3 --tp 8`

**Assert** (M1 명시):
```python
# M1 main.py argparse 후
assert num_attacc == args.tp == args.ngpu, \
    f"Deployment supports only num_attacc == tp == ngpu (1 AttAcc per GPU). " \
    f"Got num_attacc={num_attacc}, tp={args.tp}, ngpu={args.ngpu}."
```

이 assert가 M6.3의 `n_q_per_attacc = num_q_heads // attn_tp` 단일 나눗셈 가정을 보장.

**Pass**: 3 config 모두 정상 실행, assert 통과.

### M2a — model.py Transformer 재작성 (3.5 hr)

```python
def __init__(self, modelinfos, tensor_parallel=8):
    if isinstance(modelinfos, list):
        modelinfos = self._list_to_dict(modelinfos)
    self.name = modelinfos['name']
    self.ndec = modelinfos['ndec']
    self.hdim = modelinfos['hdim']
    self.num_q_heads = modelinfos.get('num_q_heads', modelinfos.get('num_heads'))
    self.num_kv_heads = modelinfos.get('num_kv_heads', self.num_q_heads)
    self.dhead = modelinfos.get('dhead', int(self.hdim / self.num_q_heads))
    self.q_proj_out = modelinfos.get('q_proj_out', self.num_q_heads * self.dhead)
    self.kv_proj_out = modelinfos.get('kv_proj_out', self.num_kv_heads * self.dhead)
    self.qkv_proj_out_total = modelinfos.get('qkv_proj_out_total',
                                              self.q_proj_out + 2 * self.kv_proj_out)
    self.ff_scale = modelinfos['ff_scale']
    self.ffn_type = modelinfos.get('ffn_type', 'standard')
    self.ff_intermediate = modelinfos.get('ff_intermediate',
                                           int(self.ff_scale * self.hdim))
    self.activation = modelinfos.get('activation', 'gelu')
    self.dtype = modelinfos['dtype']
    # tp split
    self.tp_arg = tensor_parallel
    self.fc_tp = self.tp_arg
    self.attn_tp = min(self.tp_arg, self.num_kv_heads)
    self.ff_tp = self.tp_arg
    if self.attn_tp != self.tp_arg:
        print(f"WARNING: attn_tp clamped to num_kv_heads={self.num_kv_heads} (ngpu={self.tp_arg}). "
              f"qkv FC sharded {self.fc_tp}-way but attn only uses {self.attn_tp}-way. "
              f"Edge-case (R5): KV repartition between FC output and attention input not modeled in detail; "
              f"using analytical fallback (KV broadcast cost = 0). Use symmetric tp (attn_tp == fc_tp) for accurate analysis.")
    self.num_heads = self.num_q_heads  # back-compat alias
    self.tp = self.fc_tp
```

`build()` sum_decoder + gen_decoder 둘 다 수정:
- qkv FC: `qkv_proj_out_total / fc_tp` (was `3*hdim/tp`)
- score/softmax/context: 
  - `numOp = num_q_heads / attn_tp * batch` (GPU compute)
  - `pim_numOp = num_kv_heads / attn_tp * batch` (PIM trace, NEW field)
- prefill에 추가:
  - `comm_x2g_kv` traffic = `2 * kv_proj_out / fc_tp` (K+V)
  - `comm_x2g_q` traffic = `q_proj_out / fc_tp` (Q forward, NEW, num_q heads)
  - `comm_x2g_return` traffic = `q_proj_out / attn_tp` (return, NEW, num_q heads)
- proj: `hdim / fc_tp`
- ff: `ff_intermediate / ff_tp`

**Layer 객체에 pim_numOp 필드 추가** (model.py Layer class):
```python
class Layer:
    def __init__(self, ..., numOp=1, pim_numOp=None, ...):
        self.numOp = numOp
        self.pim_numOp = pim_numOp if pim_numOp is not None else numOp  # fallback
```

기존 GPT-175B (MHA, n_q==n_kv) entries는 pim_numOp 명시 안 해도 fallback으로 동일 → 변동 없음.

Score Layer 정의 위 docstring:
```python
# Layer shape contract for ramulator_wrapper:
#   layer.n = accumulated KV length (L)  → wrapper interprets as `l`
#   layer.k = dhead                       → wrapper interprets as `dhead`
#   layer.numOp = num_q_heads per AttAcc  → GPU/FLOPs accounting
#   layer.pim_numOp = num_kv_heads per AttAcc → Ramulator/PIM trace
#   layer.m = query count (NOT used by wrapper, multiplied externally)
```

**Pass**:
- KV cache 80.02 MB ± 1% (Qwen3-VL-4B, TP=1)
- Per-GPU KV 40 MB ± 1% (TP=2 deployment)
- TP clamp warning 발생 (n_kv < ngpu 시)
- GPT-175B legacy entry (list-style) 정상 동작

### M2b — system.py KV capacity (1 hr)

```python
# get_required_mem_capacity() 정정
n_kv = self.model.num_kv_heads
dhead = self.model.dhead
ndec = self.model.ndec
kv_per_token_per_layer = 2 * n_kv * dhead * a_byte
kv_total = ndec * batch * (lin + lout - 1) * kv_per_token_per_layer
kv_per_gpu = kv_total / self.model.attn_tp
weight_per_gpu = weight_memory / self.model.fc_tp
```

추가: `s_perf` dict에 `'x2g': 0` key + for-loop에 LayerType.X2G 케이스 (sum-stage prefill comm 집계). main.py csv col_name에 `s_x2g` 추가.

**Pass**: §0.3 per-GPU max batch 표 일치 + prefill output에 s_x2g 비-zero.

### M2c — system.py _pipeline 정정 (1 hr)

**중요**: `minimum_ratio`는 GPU baseline의 attn-FF overlap heuristic (기존 [system.py L126](../src/system.py#L126))이고, §0.4 `eff_lat`은 PIM §6.1 caveat (별도 함수). 둘은 **다른 개념** — 절대 합치지 않음.

```python
# 기존: 1 / (num_heads / num_xpu) — num_q_heads 기반
# 정정: num_kv_heads 기반으로만 변경 (단일 분모, GPU heuristic 의미 보존)
heads_per_xpu = max(1, self.model.num_kv_heads / self.GPU.num_xpu)
minimum_ratio = 1 / heads_per_xpu

# comm_x2g 매칭을 prefix로 (comm_x2g_q/kv/return 모두 인식)
elif layer.name.startswith("comm_x2g"):
    x2g_time += layer.exec_time
# (update loop에서도 동일 prefix match)
```

**검증**:
- GPT-175B (n_kv=96, num_xpu=8): 1/12 = 0.083 (기존과 동일)
- S1 Qwen3-VL (n_kv=8, num_xpu=1): 1/8 = 0.125
- S2 Qwen3-VL (n_kv=8, num_xpu=2): 1/4 = 0.25

이 값은 §0.4 eff_lat (S1 0.80, S2 0.57)과 **별개**. eff_lat은 M6.4의 `get_pipelining_efficiency_latency()`에서만 사용 (M2c와 분리).

**Pass**: GPT-175B 변동 없음, GQA 모델 fc/attn 분리 정확, comm_x2g_* prefix 인식, eff_lat은 M6.4와 독립.

### M3 — VLM configs (1-1.5 hr)

`config.py model_table`에 5 in-framework + 3 text-only baselines dict 추가. 각 dict 필드:
- LLM: `ndec, hdim, num_q_heads, num_kv_heads, gqa_size, dhead, q_proj_out, kv_proj_out, qkv_proj_out_total, ff_intermediate, ff_scale, ffn_type, activation`
- ViT: `vit_layers, vit_hidden, vit_num_heads, vit_intermediate, vit_ff_scale, vit_out_hidden, vit_activation`
- Image: `patch_size, image_size_default, spatial_merge_size, num_vis_tokens_per_image`
- Projector: `projector_type` (mlp / mlp_with_merger / pixel_shuffle_mlp)
- DeepStack: `has_deepstack, deepstack_layers`
- Anyres: `is_anyres, image_grid_pinpoints, use_image_newline_parameter`
- Flags: `is_concat_style, is_cross_attn`

GPT-175B 등 legacy entries는 list-style 유지 (M2a `_list_to_dict` fallback).

**Pass**: 5 모델 dict load 정상, GPT-175B 호환.

### M4 — ViT stage + broadcast + perf check (2.5 hr)

`model.py` VLM class:
- `build_vit_layer()`: ViT MHA (qkv/attn/proj) + FFN
- `build_projector()`: model-specific (mlp / merger / pixel-shuffle)
- TP=2 시 ViT는 single-GPU 처리 후 `vit_broadcast` Layer (LayerType.G2G) 추가

**Pass**:
- ViT cost (Qwen3-VL 2.8 / Qwen2.5-VL 7.8 / InternVL3 1.8 / LLaVA-1.5 1.1 / LLaVA-Next 4.3 ms) ±50%
- ViT prefill 1회 simulation < 30s (안 되면 layer caching 적용)

Implementation note (2026-05-05): M4 implemented as `Transformer.vision_decoder`, executed once per request outside decoder-layer group scaling. ViT is represented as one template with `numOp=vit_layers` or `vit_layers × crop_count`; TP>1 adds `vit_broadcast`. Follow-up fix: LLM visual tokens and ViT patch tokens are separated. Qwen/InternVL use raw patch tokens for linear ViT work and merged/effective tokens for attention approximation; LLaVA-Next AnyRes uses crop-wise ViT cost instead of one long global-attention sequence. Helper: `python tests/vlm_graph_sanity.py` now checks the ±50% ViT latency targets.

### M5 — Projector (30 min)

각 projector_type별 분기. mlp = 단순 2-layer FFN. mlp_with_merger = patch merge × spatial_merge_size². pixel_shuffle_mlp = pixel-shuffle ratio 적용 후 mlp.

**Pass**: projector latency < 2 ms.

Implementation note (2026-05-05): M5 implemented in `_build_projector()` for `mlp`, `mlp_with_merger`, and `pixel_shuffle_mlp`. Projector layers are part of `vision_decoder`.

### M6.1 — Prefill PIM execution path (3 hr)

**전제**: M7-pre가 먼저 group 구조 도입. M6.1은 group 단위 dispatch (per-layer 아님).

`system.py` sum stage:
```python
# After M7-pre: model.sum_decoder_groups = {name: (decoder, count, device)}
for group_name, (decoder, count, device) in self.model.sum_decoder_groups.items():
    for layer in decoder:
        if (layer.type in [LayerType.MATMUL, LayerType.SOFTMAX, LayerType.X2G]
            and device == 'pim'):
            exec_time, energy = self.devices['Acc'].get_time_and_energy(layer)
        else:
            exec_time, energy = self.devices['GPU'].get_time_and_energy(layer)
        # ... aggregate scaled by `count`
```

**Pass**: sum_decoder_groups의 'pim' device group에서 score MATMUL의 ramulator 시간이 sum_perf에 반영. M7 routing modes 모두 호환.

Implementation note (2026-05-05): M6.1 implemented. `_simulate_sum_group()` now dispatches sum-stage MATMUL/SOFTMAX/X2G to accelerator for non-GPU groups. PIM score uses chunked prefill path; CPU/GPU paths remain unchanged. Fake-PIM helper: `python tests/m6_1_prefill_fake.py`.

### M6.3 — Chunked prefill decompose (sampled extrapolation, 3-4 hr)

```python
def chunked_prefill_attn_time(score_layer, chunk_size, total_L, dhead,
                              num_q_heads, num_kv_heads, attn_tp):
    """
    Contract:
      layer.numOp     = num_q_heads per AttAcc (GPU compute용)
      layer.pim_numOp = num_kv_heads per AttAcc (PIM trace, K read 지배)
    deployment에서 num_attacc == attn_tp (M1 assert).
    Wrapper [ramulator_wrapper.py L162-163]가 pim_numOp을 num_hbm으로 추가 분할.
    """
    n_q_per_attacc = num_q_heads // attn_tp
    n_kv_per_attacc = num_kv_heads // attn_tp
    n_chunks = math.ceil(total_L / chunk_size)
    sample_indices = geometric_sample(n_chunks, k=8)
    sampled = []
    for idx in sample_indices:
        accumulated_L = (idx + 1) * chunk_size
        sub_layer = make_sub_score_layer(
            score_layer, m=1, n=accumulated_L, k=dhead,
            numOp=n_q_per_attacc,                  # GPU compute용 (Q heads)
            pim_numOp=n_kv_per_attacc)             # PIM trace용 (KV heads, dominant)
        t_per_query, _ = ramulator_call(sub_layer)
        sampled.append((accumulated_L, t_per_query * chunk_size))   # × chunk_size 외부
    fit = linear_fit(sampled)
    return sum(fit(i*chunk_size) for i in range(n_chunks))
```

**검증**:
- S1 Qwen3-VL: num_q=32, num_kv=8, attn_tp=1 → numOp=32, **pim_numOp=8**. Wrapper: 8/5 ≈ 2 heads/HBM (PIM trace).
- S2 Qwen3-VL: num_q=32, num_kv=8, attn_tp=2 → numOp=16, **pim_numOp=4**. Wrapper: 4/5 → max(1, 4/5) heads/HBM (1-head 분배 + 1 idle).
- LLaVA-1.5 MHA (num_q==num_kv): numOp == pim_numOp 자연 일치 (변동 없음).
- GPT-175B MHA: 변동 없음.

**Wrapper 변경** (M6.3 필수, R3 전 적용): `num_ops_per_attacc = layer.pim_numOp if hasattr(layer, 'pim_numOp') else layer.numOp` ([ramulator_wrapper.py L162](../src/ramulator_wrapper.py#L162))

**Pass**: per-run < 2 min. 384 sweep 백그라운드 ~13 hr.

Implementation note (2026-05-05): M6.3 implemented with chunked sampled prefill. `--prefill_chunk` and `--prefill_samples` control chunk size and sample count. For PIM sum score, sub-layers use `m=1`, accumulated `n=L`, original `numOp`, and `pim_numOp`; returned per-query PIM time/energy is multiplied externally by chunk tokens.

### M6.4 — Pipelining caveat (1 hr)

```python
# Per attn layer (NOT per chunk)
n_kv_per_gpu = self.model.num_kv_heads // self.model.attn_tp
eff_lat = get_pipelining_efficiency_latency(n_kv_per_gpu, num_hbm=5)
attn_time = attn_time / eff_lat
```

`get_pipelining_efficiency_latency` 함수: §0.4 표 reproduce (Qwen3-VL 0.57, Qwen2.5-VL 0.29 등).

Implementation note (2026-05-05): M6.4 implemented in `System.get_pipelining_efficiency_latency()`. PIM generation attention layers apply `layer.exec_time /= eff_lat` before `_pipeline()`. `tests/m6_4_eff_lat.py` reproduces §0.4 table, including Qwen3-VL S1/S2 0.80/0.57 and Qwen2.5-VL S1/S2 0.57/0.29. Batch >= 2 returns 1.0.

### M7-pre — simulator depth-aware refactor (4-5 hr) ⚠️ Day 2 risk

Group structure는 device + count + (optional) layer_indices 보존:
```python
# routing entry: (group_name, device, count, layer_indices=None)
# layer_indices is None → contiguous block scaled by count
# layer_indices = [...] → per-layer (DeepStack injection 위치 보존)
```

`model.py build()`:
```python
def build(self, batch, lin, lout, attn_on_hetero=False, routing=None):
    routing = routing or [('all', 'gpu', self.ndec, None)]
    self.sum_decoder_groups = {}
    self.gen_decoder_groups = {}
    self.routing_meta = []
    for entry in routing:
        if len(entry) == 3:
            group_name, device, count = entry
            indices = None
        else:
            group_name, device, count, indices = entry
        sum_dec = self._build_sum_one_layer(batch, lin, lout, device == 'pim')
        gen_dec = self._build_gen_one_layer(batch, lin, lout, device == 'pim')
        self.sum_decoder_groups[group_name] = (sum_dec, count, device, indices)
        self.gen_decoder_groups[group_name] = (gen_dec, count, device, indices)
        self.routing_meta.append((group_name, device, count, indices))
```

`system.py simulate()`:
```python
for group_name, (decoder, count, device, indices) in self.model.sum_decoder_groups.items():
    # M9 DeepStack: indices 사용해 layer 5/11/17 위치 inject 결정
    perf_per_layer = compute_layer_perf(decoder, device=device)
    aggregate(total_perf, perf_per_layer, count)
# (gen 동일)
```

**기존 `perf * ndec` 일괄 스케일 제거**.

**Pass**: 기존 GPT-175B 결과 ±1% (routing=[('all','gpu',96,None)] = ndec uniform 동치).

Implementation note (2026-05-05): M7-pre implemented. `Transformer.build()` now emits `sum_decoder_groups`, `gen_decoder_groups`, and `routing_meta`; `System.simulate()` aggregates per-group template perf/energy/flops with `count` scaling instead of global `perf * ndec`.

### M7 — Routing 3-mode (1.5-2 hr)

```python
class Routing:
    def __init__(self, model, mode='conservative', layer_list=None):
        self.mode = mode
        self.layer_list = layer_list or [0,8,12,13,...,29,31,33]
        self.has_deepstack = getattr(model, 'has_deepstack', False)
        # DeepStack 모델은 layer 위치 정보 필수 → list mode 강제
        if self.has_deepstack and mode == 'optimistic':
            print(f"WARNING: DeepStack model with optimistic mode loses inject positions. "
                  f"Forcing list mode.")
            self.mode = 'list'

    def to_groups(self, ndec):
        if self.mode == 'conservative':
            return [('all', 'pim', ndec, None)]
        elif self.mode == 'optimistic':
            # count-compressed (DeepStack 없는 모델만)
            pim_count = sum(1 for i in range(ndec) if i in self.layer_list)
            return [('pim', 'pim', pim_count, None),
                    ('gpu', 'gpu', ndec - pim_count, None)]
        elif self.mode == 'list':
            # per-layer (DeepStack 위치 보존)
            return [(f'l{i}',
                     'pim' if i in self.layer_list else 'gpu',
                     1, [i]) for i in range(ndec)]
```

**Pass**: 3 mode 모두 crash 없이 다른 결과. DeepStack 모델 (Qwen3-VL-4B)은 list mode forced — layer 5/11/17 inject 위치 보존.

Implementation note (2026-05-05): `Routing` class and CLI flags `--routing {default,conservative,optimistic,list}` + `--pim_layers` added. `dgx-cpu` sanity confirms all three modes run. Until M9/layer-index-sensitive graph is implemented, `optimistic` and `list` are performance-equivalent for the same selected layer set; list mode already preserves indices for DeepStack.

Follow-up fix (2026-05-05): default hetero execution now auto-forces list routing for DeepStack models even when `--routing` is omitted. To preserve default conservative semantics, omitted `--pim_layers` maps to all decoder layers (`range(ndec)`), not the 22-layer default routing list. This prevents future M9 DeepStack injection positions from being lost.

### M8 — Capacity per-GPU breakdown (45 min)

`get_required_mem_capacity()`의 기존 contract (return tuple, [system.py L384](../src/system.py#L384) `cap_usage = sum(...)` 호환) 유지. 별도 method로 breakdown 노출:

```python
def get_required_mem_capacity(self, batch_size, lin, lout):
    # 기존 return tuple 보존 (sum() 호환)
    ...
    return (weight_memory, kv_memory, activation_buffer)

def get_capacity_breakdown(self, batch_size, lin, lout):
    """v2 dict accessor for paper analysis."""
    ...
    return {
        'kv_total': kv_total, 'kv_per_gpu': kv_per_gpu,
        'weight_per_gpu': weight_per_gpu,
        'available_kv': available,
        'max_batch_at_default_L': int(available / kv_per_req_per_gpu),
    }
```

CSV/output 변경 없음. paper analysis script가 `get_capacity_breakdown()` 호출.

**Pass**: 기존 simulate() crash 없음, §0.3 per-GPU max batch 표 ±10% (script 호출 시).

### M9 — DeepStack sum_decoder injection (30 min)

`build_sum_one_layer()` after each attn block:
- if `has_deepstack` and current `layer_idx in deepstack_layers`: ADD residual layer (element-wise add)

Gen_decoder 변경 없음 (decode 단계엔 visual feature 안 들어감).

**Pass**: Qwen3-VL-4B sum_decoder의 layer 5/11/17 위치에 inject layer 존재.

Implementation note (2026-05-05): M9 implemented as `deepstack_add` ACT layer in sum decoder for layer indices 5/11/17. DeepStack models default to list routing to preserve indices. Verified by `tests/vlm_graph_sanity.py`.

### M12 — Anyres best-fit (1 hr)

LLaVA-Next 공식 알고리즘:
```python
def select_best_resolution(image_size, possible_resolutions):
    # max effective area, min wasted area tie-break
    ...
def compute_anyres_tokens(image_size, cfg):
    grid = select_best_resolution(image_size, cfg['image_grid_pinpoints'])
    n_patches = (grid[0]//cfg['patch_size']) * (grid[1]//cfg['patch_size'])
    n_base = (cfg['image_size_default']//cfg['patch_size']) ** 2
    if cfg['use_image_newline_parameter']:
        n_patches += grid[1]//cfg['patch_size']
    return n_patches + n_base
```

**Pass**: LLaVA-Next 672² input → 2880 token ±10%.

Implementation note (2026-05-05): M12 implemented in `Transformer.compute_visual_tokens()` and `select_best_resolution()`. LLaVA-Next 672² returns 2928 tokens, within ±10% of 2880, with newline token rule enabled.

### M13 — Trace_gen + wrapper extension (2 hr)

`ramulator_wrapper.py`:
```python
class Ramulator:
    def __init__(self, modelinfos, ramulator_dir, output_log='',
                 fast_mode=False, num_hbm=5, max_L=2048):       # NEW
        ...
        self.max_L = max_L

    def run_ramulator(self, ...):
        trace_args = "--dhead {} --nhead {} --seqlen {} --maxlen {} --dbyte {} --output {}".format(
            self.dhead, num_ops_per_hbm, l, self.max_L, dbyte, trace_file)

    def update_log_file(self, log):
        columns = ['L', 'max_L', 'nhead', 'dhead', 'dbyte',     # max_L 추가 → 13 columns
                   'pim_type', 'power_constraint',
                   'cycle', 'mac', 'softmax', 'mvgb', 'mvsb', 'wrgb']
        # ⚠️ L83의 `if len(df.columns) > 12: import pdb; pdb.set_trace()` 제거 필수
        # (max_L 추가 시 13 columns → pdb trap 트리거됨)

    def run(self, ...):
        file_name = "attacc_l{}_maxl{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
            l, self.max_L, num_ops_per_hbm, dhead, layer.dbyte, int(power_constraint))
        log = [l, self.max_L, num_ops_per_hbm, dhead, dbyte, pim_type.name,
               power_constraint] + result

    def output(self, ...):
        row = self.df[(self.df['L']==l) & (self.df['max_L']==self.max_L) &
                      (self.df['nhead']==num_ops_per_hbm) & ...]
```

`system.py`:
```python
def __init__(self, gpu_config, modelinfos=None, hetero_name=DeviceType.NONE,
             hetero_config=None, max_L=2048):                   # NEW
    self.max_L = max_L
    ...

def set_accelerator(self, modelinfos, name, config):
    if name == DeviceType.PIM:
        ramulator = Ramulator(modelinfos, "ramulator2", "ramulator.out",
                              max_L=self.max_L)
```

`main.py`: `system = System(xpu_config['GPU'], modelinfos, max_L=args.max_L)`.

trace_gen 파일은 `--maxlen` argparse 이미 있음 (변경 없음).

**Pass**: `--max_L 8192` 정상 흘러가서 cache row 분리.

Follow-up fix (2026-05-04 local verification): wrapper must fail fast when the
Ramulator execution environment is incomplete. `run_ramulator()` now checks that
`ramulator2/ramulator2` and `ramulator2/trace_gen/gen_trace_attacc_*.py` exist
before generating traces, uses `sys.executable` instead of bare `python`, runs
trace generation and Ramulator with `subprocess.run(..., check=True)`, and
removes generated trace/yaml files via `os.remove()` in `finally` blocks. This is
required because a missing binary or failed subprocess must not produce a
zero-cycle cache row in `ramulator.out`.

Additional wrapper consistency fixes:
- `fast_mode` execution time is scaled by `num_ops_group` on freshly generated rows, matching cached-row behavior.
- `PIMType.BUFFER` memory-access scaling is consistent between fresh and cached rows (`* 1`, not cached-only `* 2`).
- Direct `System(..., hetero_name=DeviceType.PIM, hetero_config=...)` construction now instantiates `PIM(config, scaling_factor, ramulator)` with a valid Ramulator object.

Environment gate for R2/R3:
- `git submodule status` must not show `-... ramulator2`; the submodule must be initialized.
- `ramulator2/ramulator2` must exist.
- `ramulator2/trace_gen/gen_trace_attacc_bank.py` must exist.
- If any of these are missing, R2/R3 are blocked by environment, not by simulator logic.

### M14 — Inter-GPU NVLink4 all-reduce (1.5 hr)

**기존 [devices.py](../src/devices.py#L225) G2G time model 그대로 유지** (6060 ns latency + bandwidth + traffic/num_xpu*(num_xpu-1)). M14는 다음만:

1. M1에서 `--interface NVLINK4` 지정 시 GPU config의 `INTERFACE_BW = 900 GB/s` 자동 적용 (config.py에 이미 있음)
2. S1 (num_xpu=1) 케이스: 기존 G2G 모델이 이미 `traffic / num_xpu * (num_xpu - 1) = 0` → comm_g2g layer time = 0. 별도 분기 불필요.
3. S2 (num_xpu=2): 기존 G2G 모델 + NVLink4 BW 적용 → 자연 계산.

따라서 M14는 새 코드가 거의 필요 없음. **수동 검증만 수행**:
- S1 simulation에서 comm_g2g layer.exec_time == 0
- S2 simulation에서 comm_g2g layer.exec_time > 0 (Qwen3-VL prefill 약 0.4-0.5 ms range)
- BW 변경 (NVLink3 600 → NVLink4 900 GB/s) 반영 확인

**Pass**:
- S1 comm_g2g layer time = 0 (기존 모델 자동)
- S2 comm_g2g layer time > 0, NVLink4 적용 시 NVLink3보다 0.67× 빠름

Implementation note (2026-05-05): M14 validated by `tests/m14_nvlink.py`. S1 G2G is zero, S2 G2G is non-zero, and H100/NVLink4 interface is faster than NVLink3. The helper checks both a small-message smoke (`lin=16`, latency-dominated) and a larger-message case (`lin=569`, bandwidth effect visible).

### P1 — devices.py PIM SOFTMAX zero-return (5 min)

```python
elif layer.type == LayerType.SOFTMAX:
    # Score's ramulator trace already includes softmax cycles (gen_trace Attention()).
    # Returning non-zero here would double-count.
    return 0, [0, 0, 0, 0, 0, 0]
```

**Pass**: PIM softmax exec_time = 0 in R1.

---

## 2. Modification quick reference

| # | Files | Time | Risk |
|---|---|---|---|
| M0 | README, E4 scripts | 30 min | low |
| M1 | main.py | 1 hr | low |
| M2a | model.py | 3.5 hr | medium |
| M2b | system.py + main.py | 1 hr | low |
| M2c | system.py | 1 hr | low |
| M3 | config.py | 1-1.5 hr | low |
| M4 | model.py | 2.5 hr | medium |
| M5 | model.py | 30 min | low |
| M6.1 | system.py | 3 hr | medium |
| M6.3 | model.py, ramulator_wrapper.py, devices.py | 3-4 hr | medium |
| M6.4 | system.py | 1 hr | low |
| **M7-pre** | model.py, system.py | **4-5 hr** | **high** |
| M7 | system.py | 1.5-2 hr | medium |
| M8 | system.py | 45 min | low |
| M9 | model.py | 30 min | low |
| M12 | model.py | 1 hr | low |
| M13 | ramulator_wrapper.py, system.py, main.py | 2 hr | low |
| M14 | model.py, devices.py | 1.5 hr | low |
| P1 | devices.py | 5 min | trivial |

**Total: 28-32 hr**.

---

## 3. Validation R1~R8

### R1 — Sanity (Phase 1 gate, after M0+M1+M2a+M2b+M2c+M3+P1)

Setup: Qwen3-4B (text-only) + Qwen3-VL-4B (LLM-only mode), L_in=569, batch=1, BF16, **S1 (TP=1) + S2 (TP=2) 둘 다**.

KV cache sanity는 `effective_L = L_in + L_out - 1 = 569` 기준으로 수행한다. 즉 capacity sanity는 `get_capacity_breakdown(..., L_out=1)` 또는 동등한 helper로 확인한다. `L_out=128` serving run은 시간/throughput 비교용이며 KV cache 기대값이 약 97.9 MB(S1)로 달라진다.

R1 gate에서 아래 capacity helper를 반드시 별도 호출해 S1/S2 KV 값을 확인한다. `--lin 16 --lout 2` smoke는 graph/CLI 확인용이며 80.02 MiB sanity를 대체하지 않는다.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python tests/r1_sanity.py
```

Expected:

```text
TP1 kv_per_gpu_mib=80.02
TP2 kv_per_gpu_mib=40.01
r1-sanity-ok
```

Equivalent inline check:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
from src.config import make_model_config, make_xpu_config, make_pim_config
from src.system import System
from src.type import DataType, InterfaceType, GPUType, PIMType, DeviceType

model_cfg = make_model_config('Qwen3-VL-4B', DataType.W16A16)
for tp in [1, 2]:
    xpu_cfg = make_xpu_config(GPUType.H100, num_gpu=tp)
    pim_cfg = make_pim_config(PIMType.BA, InterfaceType.NVLINK4,
                              num_attacc=tp, num_hbm=5)
    system = System(xpu_cfg['GPU'], model_cfg, max_L=2048)
    system.set_accelerator(model_cfg, DeviceType.PIM, pim_cfg)
    b = system.get_capacity_breakdown(batch_size=1, lin=569, lout=1)
    print(tp, round(b['kv_per_gpu'] / 1024 / 1024, 2))
'@ | python -
```

R1 시점에 가능한 검증 (M6.1/M13/M14 미적용):

Pass:
- KV cache total = 80.02 MB ± 1% (S1) / per-GPU = 40.0 MB ± 1% (S2)
- Decode total time ≈ 0.66 ms ± 5% (S1, S2 동일)
- Vicuna-7B (MHA) 정상
- TP clamp warning 발생 (n_kv=4 + ngpu=8 시)
- **gen path PIM softmax layer time = 0** (P1 적용)
- **gen path PIM context layer time = 0** (devices.py L349-350, 기존 design)
- prefill graph에 comm_x2g_q + comm_x2g_kv + comm_x2g_return 셋 다 존재 (layer 객체 confirmation)
- gen_decoder qkv/proj/ff dim이 fc_tp 기준
- sum path의 s_perf['x2g'] 비-zero (M2b에서 LayerType.X2G 케이스 추가 후, prefill comm_x2g_q/kv/return의 GPU 측 시간 집계)

**R1에서 보류 → 후속 gate에서 검증**:
- ~~prefill PIM context/softmax = 0~~ → R3 (M6.1 적용 후)
- ~~max_L=2048 vs 8192 cache row~~ → M13 gate
- ~~S2 comm_g2g time > 0~~ → R3 (S2 multi-GPU 시뮬 시)
- ~~PIM numOp / pim_numOp 분리 검증~~ → R3 (M6.3 sub-layer 호출 시)

### R2 — AttAcc paper repro (Phase 2 gate, after M7-pre+M7+M8)

Setup: GPT-175B, L=2048, batch=64, A100a × 8, NVLink3, num_attacc=8, num_hbm=5, TP=8.

Pass (±20%, must/should):
- DGX×AttAccs vs DGX_Base (FP16): 4.84× ± 20% → must
- DGX×AttAccs vs DGX_Large (FP16): 2.48× ± 20% → should
- INT8: 3.47× / 2.59× ± 20% → should
- must 1건 + should 3/4 이상 통과 = R2 OK.

Gate helper:
```powershell
python tests/r2_paper_repro.py
```

Precondition: `ramulator2` submodule initialized and built. The helper requires
both `ramulator2/ramulator2` and `ramulator2/trace_gen/gen_trace_attacc_bank.py`.
If either file is absent, do not interpret R2 as a failed paper repro; fix the
Ramulator environment first.

Locked Ramulator commit caveat (2026-05-04): the SHA in `set_pim_ramulator.sh`
(`b7c70275f04126c647edb989270cc429776955d1`) is no longer reachable in
`CMU-SAFARI/ramulator2` upstream. The closest commit whose tree matches every
patch's index header is `37a3fd4734e6352b03eb68fc2eae61ff113fc564` (2024-01-27,
"Merge pull request #29 from cyyself/fix_cstdint"). All 21 patches apply
cleanly there. Calibration drift between the lost SHA and `37a3fd4` should be
treated as a known unknown when interpreting absolute PIM cycles, but gain
trends should remain comparable.

### R3 — corrected E2 (Phase 3 gate, after M6+M9+M12+M14)

Setup: Qwen3-VL-4B, 672² (L_in=569, L_out=128), batch=1, full proposal, chunk=512.

**R3.S1 (H100 × 1, TP=1)** primary:
- E2E gain = **1.58× ± 20%** (1.26-1.90×)
- Interface time / PIM compute ≈ **0.5-0.7** (Q transmission 큼, all-reduce=0)
- Single-GPU prefill PIM time + interface < 6 ms

**R3.S2 (H100 × 2, TP=2)** primary:
- E2E gain = **1.53× ± 20%** (1.22-1.84×)
- Interface time / PIM compute ≈ **0.2-0.4** (Q split, NVLink4 all-reduce 추가)
- Per-GPU prefill PIM time + interface < 4 ms

Sub R3.3: A100a×8 (paper baseline) vs H100×1 vs H100×2 — gain trend 비교.

Gate helper:
```powershell
python tests/r3_gate.py
```

Precondition: same Ramulator environment as R2. A `dgx-attacc` smoke should fail
fast with `FileNotFoundError` when the trace generator or binary is missing; it
must not append zero-cycle rows to `ramulator.out`.

### R4 — Sensitivity sweep

**384 runs**: batch ∈ {1,4,8,16,32,64} × L ∈ {128,569,1024,2048} × chunk ∈ {4,16,64,256} × pim_layers ∈ {0,11,22,36}.

Pass: crash 없음, monotonic. Per-run < 2 min, 백그라운드 ~13 hr.

### R5 — Edge cases

L=1, L=2048 (max), chunk_size=L, pim_layers={}, pim_layers=all, **TP=8 vs n_kv=4 clamp** (warning + analytical fallback for KV repartition).
L=100k는 model.py-only fallback (Ramulator bypass, analytical estimate).

Pass: graceful, no crash. attn_tp != fc_tp 케이스에서 warning 출력 + 결과 일관성.

### R6 — User H100 measurement (사용자 task, S1 + S2)

H100 실측:
- **R6.S1**: H100 × 1 (single GPU 측정) → R3.S1과 ±50%
- **R6.S2**: H100 × 2 (TP=2 측정) → R3.S2와 ±50%

### R7 — Multi-VLM (after M3+M4+M5+M9+M12+M14, S1 + S2 둘 다)

4 모델 동일 methodology (Qwen3-VL/Qwen2.5-VL/InternVL3/LLaVA-1.5) + LLaVA-Next prefill-only checkpoint. **각 모델 × {S1, S2} = 8 + 2 runs**.

Pass:
- ViT cost (Qwen3-VL 2.8/Qwen2.5-VL 7.8/InternVL3 1.8/LLaVA-1.5 1.1/LLaVA-Next 4.3 ms) ±50% — S1, S2 동일
- §0.2 KV cache 표 일치
- §0.3 per-GPU max batch S1 + S2 둘 다 ±10%
- §0.4 eff_lat 표 일치:
  - **S1**: Qwen3-VL 0.80, Qwen2.5-VL/InternVL3 0.57, LLaVA-1.5 0.91, LLaVA-Next 0.80
  - **S2**: Qwen3-VL 0.57, Qwen2.5-VL/InternVL3 **0.29**, LLaVA-1.5 0.80, LLaVA-Next 0.57
  - Helper: `python tests/m6_4_eff_lat.py`
- LLaVA-Next 2880 token ±10%
- S1 vs S2 gain difference 모델별 비교 (capacity vs comm 트레이드오프)

### R8 — Quantization stability + H100 FP8

Pass:
- BF16 production safe: NaN 0회 over 100 runs
- FP16 unsafe documented: NaN ≥1회
- INT8 W-only: weight load 절반, accuracy ≤1% degradation
- W8A8 throughput + attention distortion artifact 별도 보고
- H100 FP8: FP8 attn 2× FP16 throughput (Transformer Engine spec)

---

## 4. Dependency graph

```
M0 ──► M1 ──► M2a ──► M2b ──► M2c ──► M3 ──► P1 ──► R1 (Phase 1 gate, gen-side only)

M7-pre ──► M7 ──► M8 ──► R2 (Phase 2 gate, GPT-175B paper repro)

M4 ──► M5 ──► M9 ──► (graph는 M7-pre groups 위에 얹음)

M6.1 + M6.3 + M6.4 + M14 + M12 ──► R3 (Phase 3 gate, S1+S2 prefill PIM)

M13 (max_L extension, R5 long-L 전 필수) ──► R5

──► R4 + R7 + R8
──► R6 (사용자 H100 S1+S2 실측)
```

**Implementation order (재정렬, M7-pre를 M4 앞으로)**:

1. M0, M1, M2a, M2b, M3 — graph + config
2. P1 — devices.py SOFTMAX zero (5 min)
3. M2c — _pipeline 정정 (식 검증된 후, R1 전에 들어감)
4. **R1 (Phase 1 gate)** — gen-side sanity
5. **M7-pre, M7, M8** — graph 구조 안정화 (M4 전에 처리)
6. **R2 (Phase 2 gate)** — GPT-175B paper repro
7. M4, M5, M9 — VLM/projector/DeepStack (group 위에 얹음)
8. M6.1, M6.3, M6.4 — prefill PIM path
9. M14, M12 — comm + Anyres
10. **R3 (Phase 3 gate)** — S1+S2 corrected E2
11. M13 — max_L extension
12. R4, R5, R7, R8 — validation

---

## 5. Timeline (재정렬: graph 구조 먼저)

### Day 1 — Foundation + Phase 1 gate (~9-11 hr)

- 09:00-09:30: M0
- 09:30-10:30: M1
- 10:30-14:00: M2a (3.5 hr)
- 14:00-15:00: M2b
- 15:00-16:30: M3
- 16:30-16:35: P1
- 16:35-17:35: M2c (식 검증된 후)
- 17:35-18:35: **R1 Phase 1 gate** (S1 + S2 gen-side sanity)

### Day 2 — Graph refactor + Phase 2 gate (~9-11 hr)

- 09:00-13:00: **M7-pre (4-5 hr, 위험 task)** — M4 전에 안정화
- 13:00-14:00: 점심
- 14:00-16:00: M7
- 16:00-16:45: M8
- 16:45-17:45: **R2 Phase 2 gate** (GPT-175B 4.84× must)

### Day 3 — VLM stage + PIM path + Phase 3 gate (~10-12 hr)

- 09:00-11:30: M4
- 11:30-12:00: M5
- 12:00-12:30: M9 (DeepStack sum-only)
- 12:30-13:30: 점심
- 13:30-16:30: M6.1
- 16:30-20:30: M6.3
- 20:30-21:00: M6.4 + M14 (M14는 거의 검증만)
- 21:00-22:00: M12

### Day 4 — Phase 3 gate + Validation

- 09:00-11:00: **R3 Phase 3 gate** (S1 + S2 corrected E2)
- 11:00-13:00: M13 (max_L extension)
- 13:00 ~: R4 384 runs 백그라운드 (~13 hr)
- 14:00-15:00: R5
- 15:00-17:30: R7 (5 모델 × {S1, S2})
- 17:30-18:30: R8

### Day 5+ (optional) — R6 (사용자 H100 S1+S2 실측)

**Total: 4 days core + 1 day validation**.

---

## 6. Pre-flight checklist

- [ ] §0.1 **S1 (H100×1, TP=1) + S2 (H100×2, TP=2) 두 시나리오** 동의
- [ ] §0.2 5 in-framework + 3 baselines + 2 observation 동의
- [ ] §0.3 per-GPU max batch S1 + S2 표 인정
- [ ] §0.4 eff_lat S1/S2 표 (S2 Qwen2.5-VL/InternVL3 0.29) 인정
- [ ] §0.5 critical interpretations 동의 (score = full attn / context+softmax PIM=0 / m external × / shape contract)
- [ ] §1 M0~M14 + P1 action list 동의
- [ ] §3 R1~R8 pass criteria 동의 (R3/R6/R7 S1+S2 둘 다)
- [ ] §5 4 days core + 1 day validation 동의
- [ ] Implementation 시작 OK

**ready 시 Day 1 09:00 시작 (정렬된 순서)**:
M0 → M1 → M2a → M2b → M3 → P1 → M2c (식 검증된 후) → R1 (S1+S2 gen-side sanity)

---

**문서 끝 (final_v1)**.
