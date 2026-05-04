# AttAcc Simulator Patch Implementation Report

작성일: 2026-05-04

## 1. 패치 범위

이번 패치는 `attacc_simulator_plan_final_v1.md`의 Phase 1 기반 구현을 실제 코드에 반영한 것이다. 목표는 바로 다음 구현 단계로 넘어가기 전에 simulator의 모델 형상, GQA/MQA 처리, PIM trace head 수, CLI 배포 파라미터, capacity sanity가 서로 같은 계약을 보도록 만드는 것이었다.

실제 수정 파일:

- `../main.py`
- `../src/config.py`
- `../src/model.py`
- `../src/system.py`
- `../src/devices.py`
- `../src/ramulator_wrapper.py`
- `attacc_simulator_plan_final_v1.md`
- `../../260422_verification/README.md`
- `../../260422_verification/e4_multi_model_m4.py`

핵심 설계 의도:

- GPU/FLOPs 관점의 Q head 수와 PIM/Ramulator trace 관점의 KV head 수를 분리한다.
- GQA 모델에서 `num_q_heads != num_kv_heads`일 때 qkv/proj/attention/capacity 산식이 틀어지지 않게 한다.
- H100 x 1/2, NVLink4, AttAcc 수, HBM stack 수, `max_L` 같은 배포 결정을 CLI에서 명시 가능하게 한다.
- 기존 GPT/LLAMA/OPT 계열 MHA 결과는 최대한 보존하고, 신규 VLM/modern LLM config만 확장한다.
- R1 sanity에서 기대하는 KV cache 값이 실제 코드 산식과 같은 길이 기준을 쓰도록 문서와 helper를 맞춘다.

## 2. `main.py`

### 무엇을 수정했나

- CSV header에 `s_x2g` 컬럼을 추가했다.
- CSV header의 capacity 컬럼명을 `required_cap`에서 `required_cap_per_gpu`로 바꿨다.
- CSV writer를 `open(..., newline='')` 기반 `with` 블록으로 바꿨다.
- CLI 인자를 추가했다.
  - `--tp`
  - `--num_attacc`
  - `--num_hbm`
  - `--interface`
  - `--max_L`
- `--word` 기본값을 문자열 `'2'`에서 정수 `2`로 고쳤다.
- `num_attacc == tp == ngpu` assertion을 추가했다.
- PIM interface를 하드코딩 `NVLINK3`에서 CLI 선택값으로 바꿨다.
- `make_pim_config()`에 `num_attacc`, `num_hbm`을 전달하게 했다.
- `System(..., max_L=args.max_L)`를 사용하게 했다.
- 기존 `os.system("rm output.csv")`를 `os.remove(output_path)`로 바꿨다.

### 어떻게 수정했나

- `argparse` 단계에서 deployment 관련 값들을 명시적으로 받게 했다.
- `tp`와 `num_attacc`가 주어지지 않으면 기존처럼 `ngpu`를 기본값으로 사용한다.
- 단, 현재 구현은 plan의 locked deployment와 맞춰 `num_attacc == tp == ngpu`만 허용한다.
- CSV 컬럼 순서를 `system.py`의 `s_perf` 출력 순서와 맞췄다.

### 왜 수정했나

- plan에서 H100 x 2, NVLink4, AttAcc 수, HBM stack 수가 locked decision인데 기존 CLI는 이를 재현할 수 없었다.
- `s_perf`에 X2G를 추가하면 CSV header도 같이 바뀌어야 한다. 그렇지 않으면 downstream 분석이 컬럼을 잘못 읽는다.
- capacity 산식이 TP 분할 후 per-GPU 값을 반환하므로 기존 `required_cap` 이름을 유지하면 R2 paper repro plot에서 system-total 값으로 오해할 수 있다.
- `--word default='2'`는 `args.word == 2` 비교에서 false가 되어 기본 실행이 의도치 않게 W8A8 경로로 떨어질 수 있는 실제 버그였다.
- Windows 환경에서 `csv.writer`가 빈 줄을 끼우는 문제를 막기 위해 `newline=''`가 필요했다.
- `os.system("rm ...")`는 Windows와 안전성 모두에서 좋지 않으므로 native Python 삭제로 바꿨다.

## 3. `src/config.py`

### 무엇을 수정했나

- `make_model_config()`가 기존 legacy list format과 신규 dict format을 모두 처리하게 했다.
- 신규 모델 config를 추가했다.
  - `Qwen3-4B`
  - `Qwen3-VL-4B`
  - `Qwen2.5-VL-7B`
  - `InternVL3-8B-hf`
  - `Vicuna-7B`
  - `LLaVA-1.5-7B`
  - `Mistral-7B`
  - `LLaVA-Next-Mistral-7B`
- 모든 config에 다음 파생 필드를 생성하게 했다.
  - `num_q_heads`
  - `num_kv_heads`
  - `gqa_size`
  - `q_proj_out`
  - `kv_proj_out`
  - `qkv_proj_out_total`
  - `ff_intermediate`
  - `ffn_type`
  - `activation`
- VLM 관련 metadata 기본값을 추가했다.
  - `has_deepstack`
  - `deepstack_layers`
  - `is_anyres`
  - `image_grid_pinpoints`
  - `use_image_newline_parameter`
  - `is_concat_style`
  - `is_cross_attn`

### 어떻게 수정했나

- 기존 `model_table[name] = [ndec, hdim, nheads, dhead, ff_scale, gqa_size]` 구조는 그대로 유지했다.
- 신규 모델은 dict로 정의하고, return 직전에 공통 파생 필드를 채운다.
- legacy list 모델은 dict로 변환하면서 `num_q_heads == num_kv_heads`인 MHA 호환 구조를 만든다.

### 왜 수정했나

- 기존 config는 `num_heads` 하나로 Q/KV head를 동시에 표현했다. GQA 모델에서는 이 가정이 틀린다.
- Qwen 계열처럼 Q head와 KV head가 다른 모델은 qkv FC 출력 차원, PIM trace head 수, KV cache 크기가 모두 다르다.
- VLM 구현 단계에서 ViT/projector/DeepStack/AnyRes 정보를 config에 싣기 위해 metadata 필드가 필요하다.
- legacy 모델을 깨지 않으면서 신규 dict config를 추가하는 방식이 가장 안전하다.

## 4. `src/model.py`

### 무엇을 수정했나

- `Layer`에 `pim_numOp` 필드를 추가했다.
- `Layer`의 `m`, `n`, `k`, `numOp`, `pim_numOp`를 정수로 정규화했다.
- `Transformer`가 다음 값을 명시적으로 보유하게 했다.
  - `num_q_heads`
  - `num_kv_heads`
  - `dhead`
  - `q_proj_out`
  - `kv_proj_out`
  - `qkv_proj_out_total`
  - `gqa_size`
  - `ff_intermediate`
  - `ffn_type`
  - `activation`
  - `fc_tp`
  - `attn_tp`
  - `ff_tp`
- `attn_tp = min(tp_arg, num_kv_heads)` clamp를 추가했다.
- `Routing` class를 추가했다.
  - `conservative`: 전체 decoder layer를 accelerator device로 압축 group 처리
  - `optimistic`: 선택 layer 수만 accelerator group으로 압축하고 나머지는 GPU group 처리
  - `list`: decoder layer별 group을 만들어 layer index를 보존
- 기존 monolithic `build()`를 helper 기반으로 리팩터링했다.
  - `_split()`
  - `_heads_per_attn_shard()`
  - `_activation_name()`
  - `_append_ffn()`
  - `_build_sum()`
  - `_build_gen_stage()`
- prefill/sum path X2G를 세분화했다.
  - `comm_x2g_kv`
  - `comm_x2g_q`
  - `comm_x2g_return`
- generation path X2G를 명시화했다.
  - `comm_x2g_qkv`
  - `comm_x2g_return`
- attention layer shape contract를 명시했다.
  - `layer.numOp`: GPU/FLOPs용 Q heads
  - `layer.pim_numOp`: PIM/Ramulator trace용 KV heads
  - `layer.n`: accumulated KV length
  - `layer.k`: `dhead`
- gated FFN과 standard FFN을 분리했다.
- `sum_decoder_groups`, `gen_decoder_groups`, `routing_meta`를 추가해 depth-aware group graph를 만들 수 있게 했다.

### 어떻게 수정했나

- qkv FC 출력 차원은 `qkv_proj_out_total / fc_tp`로 계산한다.
- attention score/softmax/context의 GPU 계산 head 수는 `num_q_heads / attn_tp * batch`로 둔다.
- PIM trace head 수는 `num_kv_heads / attn_tp * batch`로 따로 넘긴다.
- prefill에서 KV 전송량은 `2 * kv_proj_out / fc_tp`, Q 전송량은 `q_proj_out / fc_tp`, return은 `q_proj_out / attn_tp`로 분리했다.
- decode generation에서는 현재 qkv 결과를 한 번에 `comm_x2g_qkv`로 보낸다. 세부 q/k/v split은 다음 sub-layer routing 단계에서 더 쪼갤 수 있게 이름과 shape를 정리했다.
- `num_kv_heads < tp`인 Qwen2.5/InternVL edge case는 attention parallelism을 KV head 수로 clamp하고 warning을 출력한다.
- `build(..., routing=...)`이 routing entry `(group_name, device, count, indices)`를 받아 group별 one-layer decoder template을 생성한다.
- DeepStack 모델에서 compressed routing을 쓰면 injection 위치가 사라지므로, `Routing`은 DeepStack 모델의 non-list mode를 `list`로 강제한다.

### 왜 수정했나

- 기존 구현은 `3 * hdim / tp`와 `num_heads / tp` 기반이라 GQA 모델 qkv projection과 attention trace가 모두 틀렸다.
- AttAcc/Ramulator trace는 실제로 KV read traffic이 지배적이므로 PIM head 수는 Q head가 아니라 KV head를 기준으로 해야 한다.
- GPU FLOPs/softmax 계산량은 Q head 기준이므로 단일 `numOp`로 두 의미를 동시에 담으면 안 된다.
- prefill에서 Q와 KV 전송량을 합쳐 `comm_x2g` 하나로 두면 GQA 모델의 traffic breakdown과 R1/R3 검증이 불가능하다.
- 리팩터링은 이후 M7 sub-layer depth refactor에서 per-layer 또는 per-sub-layer routing을 안전하게 넣기 위한 선행 작업이다.

## 5. `src/system.py`

### 무엇을 수정했나

- `System.__init__()`에 `max_L`을 추가했다.
- PIM accelerator 설정 시 `Ramulator(..., num_hbm=config['NUM_HBM'], max_L=self.max_L)`를 전달하게 했다.
- pipeline overlap 계산에서 `comm_x2g` exact match를 prefix match로 바꿨다.
- pipeline minimum ratio를 `num_q_heads`가 아니라 `num_kv_heads` 기반으로 바꿨다.
- X2G layer가 여러 개일 때 X2G time을 실제 X2G layer 수로 나눠 배분하게 했다.
- `s_perf`에 `x2g` 항목을 추가했다.
- sum/prefill stage에서 `LayerType.X2G`를 `s_perf['comm']`과 `s_perf['x2g']`에 집계하게 했다.
- output config의 `gqa_size`를 `0`이 아니라 실제 `self.model.gqa_size`로 기록하게 했다.
- OPB dtype 비교를 문자열 비교에서 enum 비교로 고쳤다.
- capacity 산식을 실제 qkv/proj/ff/KV shape 기반으로 바꿨다.
- `get_capacity_breakdown()` helper를 추가했다.
- `set_routing()`과 accelerator routing name helper를 추가했다.
- `simulate()`의 기존 `perf * self.model.ndec` uniform scaling을 제거하고, group별 `count` scaling으로 바꿨다.
- generation attention device selection을 group device 기준으로 바꿨다. `gpu` group은 GPU에서 전부 실행하고, `pim`/`cpu` group은 attention/X2G를 accelerator에서 실행한다.

### 어떻게 수정했나

- weight memory:
  - qkv: `hdim * qkv_proj_out_total`
  - attention output projection: `q_proj_out * hdim`
  - gated FFN: `hdim * ff_intermediate * 3`
  - standard FFN: `hdim * ff_intermediate * 2`
- KV cache:
  - `ndec * 2 * L * num_kv_heads * dhead * activation_byte`
  - 이후 `attn_tp`로 per-GPU 분할
- weight memory는 `fc_tp`로 per-GPU 분할한다.
- temp memory는 Q heads 기반 activation 크기를 유지한다.
- temp memory의 마지막 `+ l * nhead` 항은 byte 단위 산식에 맞춰 `+ l * nhead * a_byte`로 보정했다. 원본부터 있던 작은 단위 버그였지만, per-GPU capacity label을 정리하는 김에 같이 수정했다.
- group aggregation은 각 group의 one-layer template time/energy/flops를 먼저 계산한 뒤 `count`만큼 누적한다. 이로써 default all-layer group은 기존 uniform scaling과 동치이고, list mode는 layer index를 보존한다.
- M6.4 latency-mode `eff_lat`를 `System.get_pipelining_efficiency_latency()`로 구현했다. PIM generation attention layer는 `_pipeline()` 전에 `exec_time /= eff_lat`를 적용한다.

### 왜 수정했나

- 기존 capacity 산식은 `hdim` 전체를 KV cache로 보아 GQA 모델에서 KV cache를 과대계산했다.
- weight memory도 `3 * hdim` 가정이라 Qwen/Qwen-VL의 qkv output 차원과 맞지 않았다.
- `s_x2g`가 없으면 prefill X2G traffic 개선 또는 회귀를 CSV에서 확인할 수 없다.
- pipeline code가 `comm_x2g`라는 단일 이름만 인식하면 `comm_x2g_q`, `comm_x2g_kv`, `comm_x2g_return`을 놓친다.
- dtype enum 비교 버그는 INT8 OPB 보정이 동작하지 않는 실제 오류였다.
- M7-pre 이후 DeepStack, ViT/projector, sampled prefill처럼 layer index나 group device가 중요한 기능을 올릴 수 있다.
- §0.4 latency-mode caveat를 코드에 고정해 low-KV-head GQA 모델의 PIM latency under-utilization을 반영한다.

## 6. `src/devices.py`

### 무엇을 수정했나

- PIM device에서 `LayerType.SOFTMAX` 처리 결과를 `0, [0,0,0,0,0,0]`으로 바꿨다.

### 어떻게 수정했나

- 기존 softmax compute/mem time 계산 로직을 제거하고, score Ramulator trace에 포함된 것으로 간주했다.

### 왜 수정했나

- AttAcc PIM score trace는 score와 softmax/context 관련 cycle을 함께 반영하는 구조로 사용된다.
- 별도 PIM softmax layer time을 더하면 generation PIM path에서 softmax가 double count된다.
- plan의 P1 결정값인 `SOFTMAX zero`를 코드에 고정하기 위한 변경이다.

## 7. `src/ramulator_wrapper.py`

### 무엇을 수정했나

- `pandas` import를 optional로 바꿨다.
- `Ramulator.__init__()`에 `num_hbm`, `max_L`을 받게 했다.
- 기존 `ramulator.out`에 `max_L` 컬럼이 없으면 stale cache로 보고 비우게 했다.
- trace generation command에 `--maxlen`을 전달하게 했다.
- cache/log key와 trace filename에 `max_L`을 포함했다.
- Ramulator 실행 head 수 계산에서 `layer.numOp` 대신 `layer.pim_numOp` fallback을 사용하게 했다.
- `modelinfos.get('num_kv_heads', modelinfos['num_heads'])` 형태의 KeyError 가능성을 제거했다.
- 기존 column mismatch 시 `pdb.set_trace()`로 빠지던 코드를 제거했다.
- `output()`/`update_log_file()`에서 pandas 필요성을 assert로 명확히 했다.

### 어떻게 수정했나

- `num_ops_per_attacc = getattr(layer, 'pim_numOp', layer.numOp)`로 PIM trace head 수를 읽는다.
- `num_ops_per_hbm = ceil(num_ops_per_attacc / num_hbm)` 구조는 유지했다.
- cache lookup 조건에 `max_L`을 추가했다.
- pandas가 없을 때 import 단계는 통과하지만, 실제 Ramulator output/cache execution은 명시적으로 실패하도록 했다.

### 왜 수정했나

- GQA 모델에서 PIM trace는 Q head가 아니라 KV head 기준이어야 한다.
- `max_L=2048`과 `max_L=8192`는 같은 `L`이라도 trace layout이 달라질 수 있으므로 cache key에 들어가야 한다.
- stale `ramulator.out`을 그대로 읽으면 이전 schema 결과가 새 실행에 섞일 수 있다.
- 로컬 non-PIM smoke가 pandas 부재 때문에 import 단계에서 죽는 것은 불필요한 장애였다.
- 단, 실제 PIM/Ramulator 실행은 pandas 기반 cache/logging을 쓰므로 pandas 필요성을 숨기지 않고 assert로 남겼다.

## 8. 문서 패치

### `attacc_simulator_plan_final_v1.md`

수정 내용:

- `Layer.numOp`와 `Layer.pim_numOp`의 의미를 분리해 명시했다.
- Ramulator wrapper 변경 위치를 `M13`이 아니라 `M6.3 필수, R3 전 적용`으로 바로잡았다.
- Quick reference의 M6.3 대상 파일에 `model.py`, `ramulator_wrapper.py`, `devices.py`를 명시했다.
- R1 sanity의 KV cache 기준을 `effective_L = L_in + L_out - 1 = 569`로 정리했다.
- `L_out=128` serving run은 시간/throughput 비교용이고, KV cache sanity는 `L_out=1` 또는 동등 helper 기준이라고 명시했다.

수정 의도:

- plan과 코드가 서로 다른 길이 기준을 쓰는 것을 막기 위해서다.
- `L_in=569, L_out=128`이면 KV length는 696이므로 KV cache가 약 97.9 MiB가 된다. 기존 R1의 80.02 MiB 기대값은 `effective_L=569`일 때만 맞다.
- `pim_numOp`는 코드 구현에 이미 들어간 핵심 contract이므로 문서에도 같은 용어로 고정했다.

### `../../260422_verification/README.md`

수정 내용:

- Qwen3-VL DeepStack 위치를 `[8,16,24]`에서 `[5,11,17]`로 통일했다.
- Qwen3.5 관련 설명을 simulator validation 대상이 아니라 observation-only로 명확히 했다.

수정 의도:

- DeepStack injection 위치가 plan/config/verification 문서 사이에서 어긋나면 이후 M9 구현과 결과 해석이 흔들린다.
- Qwen3.5는 hybrid linear attention 계열이라 현재 AttAcc full-attention simulator 검증군으로 넣으면 비교가 부정확하다.

### `../../260422_verification/e4_multi_model_m4.py`

수정 내용:

- Qwen3-VL notes의 DeepStack 위치를 `[5,11,17]`로 수정했다.
- Qwen3.5 loader/comment를 observation-only 성격으로 정리했다.

수정 의도:

- 실험 스크립트 주석과 README/plan이 같은 해석을 갖도록 맞췄다.
- Qwen3.5 결과가 full-attention simulator validation처럼 오해되는 것을 방지한다.

## 9. 검증 결과

실행한 검증:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
compile(main.py, src/config.py, src/model.py, src/system.py, src/devices.py, src/ramulator_wrapper.py)
```

결과:

- `compile-ok`

shape sanity:

- `Qwen3-VL-4B`, TP=1: qkv output `6144`, score `numOp=32`, `pim_numOp=8`
- `Qwen3-VL-4B`, TP=2: qkv output `3072`, score `numOp=16`, `pim_numOp=4`
- `Qwen2.5-VL-7B`, TP=8: qkv output `576`, score `numOp=7`, `pim_numOp=1`, clamp warning 발생
- `LLaVA-1.5-7B`, TP=2: qkv output `6144`, score `numOp=16`, `pim_numOp=16`
- `GPT-175B`, TP=8: qkv output `4608`, score `numOp=12`, `pim_numOp=12`

smoke 실행:

```powershell
python main.py --model GPT-13B --lin 16 --lout 2 --batch 1
```

결과:

- A100a x 8 legacy path 정상 실행
- latency 약 `5.51 ms`

```powershell
python main.py --gpu H100 --ngpu 1 --tp 1 --model Qwen3-VL-4B --lin 16 --lout 2 --batch 1 --interface NVLINK4 --num_attacc 1
```

결과:

- H100 x 1 신규 model path 정상 실행
- latency 약 `3.57 ms`
- CSV dtype이 `W16A16`으로 기록됨
- CSV header에 `s_x2g` 포함
- Windows CSV 빈 줄 문제 제거 확인

routing sanity:

```powershell
python main.py --system dgx-cpu --model GPT-13B --lin 16 --lout 2 --batch 1 --routing conservative
python main.py --system dgx-cpu --model GPT-13B --lin 16 --lout 2 --batch 1 --routing optimistic --pim_layers 0,1,2
python main.py --system dgx-cpu --model GPT-13B --lin 16 --lout 2 --batch 1 --routing list --pim_layers 0,1,2
```

결과:

- `conservative`: routing_meta `[('all', 'cpu', 40, None)]`
- `optimistic`: routing_meta `[('acc', 'cpu', 3, None), ('gpu', 'gpu', 37, None)]`
- `list`: routing_meta starts with `[('l0', 'cpu', 1, [0]), ('l1', 'cpu', 1, [1]), ('l2', 'cpu', 1, [2]), ('l3', 'gpu', 1, [3]), ...]`
- 세 mode 모두 crash 없이 실행된다.
- M9/DeepStack injection 구현 후에도 `optimistic`과 `list`는 DeepStack이 없는 모델에서 같은 선택 layer set이면 성능상 동치다. DeepStack 모델은 list mode로 강제된다.

capacity sanity:

R1 gate에서 이 검증은 smoke test와 별도로 수행해야 한다. `--lin 16 --lout 2` smoke는 graph/CLI 확인용이고, 아래 값은 `get_capacity_breakdown(batch_size=1, lin=569, lout=1)` 기준이다.

- Qwen3-VL-4B, `L=569`, TP=1:
  - weight `6930.0 MiB`
  - KV `80.02 MiB`
  - temp `0.1042 MiB`
  - total `7010.12 MiB`
- Qwen3-VL-4B, `L=569`, TP=2:
  - weight `3465.0 MiB`
  - KV `40.01 MiB`
  - temp `0.1042 MiB`
  - total `3505.11 MiB`

## 10. 현재 제한과 남은 위험

- 로컬 Python에 `pandas`가 없어 실제 `dgx-attacc` Ramulator/PIM execution path는 아직 실행 검증하지 못했다.
- `ramulator_wrapper.py`는 non-PIM import가 가능하도록 했지만, 실제 Ramulator output/cache 경로는 pandas를 요구한다.
- `ramulator_wrapper.py` 내부에는 기존 `os.system("rm ...")` 형태가 아직 남아 있다. 이번 패치의 핵심 경로는 아니지만 Windows native cleanup으로 바꾸는 것이 다음 정리 대상이다.
- `comm_x2g_qkv`는 decode generation에서 q/k/v를 아직 완전히 sub-layer 단위로 쪼개지 않는다. M7 routing refactor 전에 안정적인 중간 표현으로 둔 것이다.
- 이 때문에 corrected E2/R3에서 interface/PIM-compute ratio가 최종 M7 split 이후 값과 미세하게 다를 수 있다.
- `simulate()`의 uniform decoder scaling은 제거됐다. 대부분의 LLM decoder layer는 아직 동일 template 기반이고, DeepStack layer만 sum-stage에서 index-specific 차이를 갖는다.
- M8 capacity policy는 helper 수준으로 구현되어 있고, paper analysis용 batch policy script는 별도 작업으로 남아 있다.
- M4 follow-up으로 LLM visual token과 ViT patch token을 분리했다. 기존 구현은 Qwen/InternVL 계열 ViT cost를 과소평가하고 LLaVA-Next AnyRes를 과대평가했으므로, 이제 `tests/vlm_graph_sanity.py`가 plan의 ViT latency ±50% 조건까지 확인한다.

## 11. 다음 구현 우선순위

1. `pandas`/Ramulator 실행 환경을 맞추고 `dgx-attacc` path smoke를 수행한다.
2. `python tests/r2_paper_repro.py`와 `python tests/r3_gate.py`를 실제 Ramulator 환경에서 실행해 R2/R3 gate를 확인한다.
3. M8: routing-aware capacity policy와 paper analysis helper를 보강한다.

## 12. 요약

이번 패치는 단순 config 추가가 아니라 simulator의 핵심 shape contract를 바꾼 패치다. 가장 중요한 변경은 `numOp`를 Q head용으로 유지하고, `pim_numOp`를 KV head 기반 PIM trace용으로 분리한 것이다. 이 변경으로 GQA 모델의 GPU FLOPs, PIM trace, capacity 산식이 서로 같은 모델 구조를 바라보게 됐다.

## 13. Concern Review Addendum

- Concern 1은 타당하다. `required_cap` 의미가 system-total이 아니라 per-GPU로 바뀌었으므로 CSV label을 `required_cap_per_gpu`로 수정했다.
- Concern 2는 타당하다. 원본부터 있던 `temp_memory` 단위 버그이며 magnitude는 작지만 `a_byte` 곱셈을 추가했다.
- Concern 3은 타당하다. R1 gate는 `get_capacity_breakdown(1, 569, 1)` 호출을 smoke와 별도로 수행해야 한다고 plan/report에 명시했다.
- Concern 4는 타당하다. `comm_x2g_qkv`는 M7 전 중간 표현이며 final interface ratio는 M7 q/k/v split 후 확정된다.
- Concern 5는 현재 패치 관점에서 문제 없음으로 봤다. legacy GPT-175B FFN memory 산식은 `ff_scale * hdim` fallback으로 plausible한 값이 나온다.
- Concern 6은 반영 완료. uniform `ndec` scaling은 group-count 기반 집계로 교체했다.
- Concern 7은 확인 완료. Qwen2.5-VL official config의 `vision_config.hidden_act`가 `silu`이므로 현재 `vit_activation='silu'`는 맞다. Reference: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json

## 14. M7-pre/M7 Implementation Addendum

추가 구현 내용:

- `Routing` class를 도입해 `conservative`, `optimistic`, `list` mode를 지원한다.
- `Transformer.build()`가 routing group을 받아 `sum_decoder_groups`, `gen_decoder_groups`, `routing_meta`를 생성한다.
- `System.simulate()`가 flat decoder template을 `ndec`로 일괄 스케일하지 않고, group별 template을 실행한 뒤 `count`로 누적한다.
- `main.py`에 `--routing`과 `--pim_layers` CLI를 추가했다.
- DeepStack 모델은 layer index 보존이 필요하므로 `Routing(..., mode!='list')` 입력 시 list mode로 강제된다.
- DeepStack 모델이 hetero path에서 `--routing` 없이 실행될 때도 list routing을 자동 강제한다. 이때 omitted `--pim_layers`는 default 22-layer list가 아니라 all layers로 해석해 기존 conservative offload 의미를 보존한다.
- `self.sum_decoder` / `self.gen_decoder`는 first-group-only back-compat alias임을 코드 주석으로 명시했다. 전체 graph는 `*_decoder_groups`를 사용해야 한다.
- R1 capacity gate용 helper `tests/r1_sanity.py`를 추가했다.
- M6.4 eff_lat gate용 helper `tests/m6_4_eff_lat.py`를 추가했다.
- M4/M5 VLM graph를 `vision_decoder`로 구현했다. ViT/projector는 decoder-layer group count로 스케일하지 않고 요청당 1회 실행된다.
- M4 follow-up으로 `compute_visual_tokens()`는 LLM input visual token, `compute_vit_tokens()`는 ViT patch token, `compute_vit_attention_tokens()`는 ViT attention approximation token으로 분리했다. LLaVA-Next AnyRes는 crop-wise ViT cost로 계산한다.
- M6.1 sum-stage accelerator dispatch를 구현했다. PIM/CPU group에서는 sum-stage MATMUL/SOFTMAX/X2G가 accelerator device를 탄다.
- M6.3 chunked sampled prefill을 구현했다. PIM sum score는 `m=1` sub-layer를 chunk/sample별로 호출하고 chunk token 수를 외부에서 곱한다.
- M9 DeepStack sum-only injection을 구현했다. Qwen3-VL layer 5/11/17에 `deepstack_add`가 들어간다.
- M12 AnyRes best-fit token 계산을 구현했다.
- M14 NVLink4 allreduce sanity helper를 추가했다. Small-message regime과 `lin=569` large-message regime을 모두 확인한다.
- R2/R3 실행 gate helper `tests/r2_paper_repro.py`, `tests/r3_gate.py`를 추가했다.

검증 요약:

- 기본 GPU smoke는 기존과 동일한 latency를 유지했다.
- shape sanity와 R1 capacity sanity는 그대로 통과했다.
- `dgx-cpu` hetero path에서 3개 routing mode가 모두 crash 없이 동작했다.
- `python tests/r1_sanity.py` 통과 기준을 추가했다.
- `python tests/m6_4_eff_lat.py`가 §0.4 eff_lat 표를 재현한다.
- `python tests/vlm_graph_sanity.py`가 vision graph, DeepStack injection, LLaVA-Next token count, M4 ViT latency ±50% 조건을 확인한다.
- `python tests/m6_1_prefill_fake.py`가 Ramulator 없이 PIM prefill chunk contract를 확인한다.
- `python tests/m14_nvlink.py`가 S1/S2 G2G/NVLink behavior를 확인한다.

해석 주의:

- `optimistic`과 `list`는 현재 선택 layer set이 같으면 성능상 동치다. list mode의 가치는 layer index를 보존하는 데 있으며, DeepStack injection이나 layer별 graph 차이가 들어가는 M9 이후부터 성능 차이가 생긴다.
- M6.4 latency-mode `eff_lat`는 구현 완료다. M6.1/M6.3 prefill PIM path도 구현됐지만, R3 corrected E2 target 검증에는 실제 Ramulator 실행환경이 필요하다.
- list mode는 현재 `ndec * (lout - 1)` decoder templates를 만든다. 정확성 문제는 없지만 sweep 전 build-time profiling이 필요하다.
- sum-stage X2G stall model은 group-local이다. M9에서 layer-specific sum graph가 들어가면 group 간 X2G race/overlap 모델을 다시 점검해야 한다.
- 실제 R3 `dgx-attacc` E2E 검증은 여전히 pandas/Ramulator 실행환경이 필요하다. 로컬에서는 fake-PIM contract와 non-PIM smoke까지만 검증했다.

## 15. 실제 GPU/Ramulator 환경 실행 계획

이 섹션은 로컬 Windows sanity가 아니라, 실제 `pandas` + Ramulator + H100/A100a 실행환경에서 최종 gate를 통과시키기 위한 실행 절차다. 목적은 세 가지다. 첫째, `dgx-attacc` PIM/Ramulator path가 실제로 끝까지 도는지 확인한다. 둘째, R2 AttAcc paper repro가 기존 baseline과 같은 order의 gain을 내는지 확인한다. 셋째, Qwen3-VL corrected E2 R3가 H100 S1/S2에서 plan target 안에 들어오는지 확인한다.

### 15.1 실행 전 환경 확인

작업 디렉터리:

```powershell
cd "C:\Users\mszza\OneDrive\바탕 화면\PIM_CASL\pim_a6000\attacc_simulator"
```

필수 확인:

```powershell
python -c "import pandas; print(pandas.__version__)"
Test-Path .\ramulator2\ramulator2
Test-Path .\ramulator2\trace_gen\gen_trace_attacc_bank.py
python tests\r1_sanity.py
python tests\m6_4_eff_lat.py
python tests\vlm_graph_sanity.py
python tests\m14_nvlink.py
```

왜 돌리나:

- `pandas`는 `ramulator_wrapper.py`의 cache/log read-write에 필요하다.
- `ramulator2` binary와 trace generator가 없으면 `dgx-attacc` path가 score layer에서 실패한다.
- R1/M6.4/M4/M14 helper는 실제 긴 run 전에 shape, capacity, latency caveat, VLM graph, NVLink behavior가 깨지지 않았는지 확인하는 빠른 pre-flight다.

통과 기준:

- `r1-sanity-ok`
- `m6_4-eff-lat-ok`
- `vlm-graph-sanity-ok`
- `m14-nvlink-ok`

### 15.2 Smoke: H100 S1/S2 non-PIM baseline

먼저 Ramulator 없이 GPU-only baseline이 정상 실행되는지 확인한다.

```powershell
python main.py --system dgx --gpu H100 --ngpu 1 --tp 1 --num_attacc 1 --num_hbm 5 --interface NVLINK4 --model Qwen3-VL-4B --lin 569 --lout 128 --batch 1 --image_size 672 --prefill_chunk 512 --prefill_samples 8 --max_L 2048 --pipeopt --ffopt

python main.py --system dgx --gpu H100 --ngpu 2 --tp 2 --num_attacc 2 --num_hbm 5 --interface NVLINK4 --model Qwen3-VL-4B --lin 569 --lout 128 --batch 1 --image_size 672 --prefill_chunk 512 --prefill_samples 8 --max_L 2048 --pipeopt --ffopt
```

왜 돌리나:

- H100 S1/S2의 GPU-only 기준 시간을 확보해야 R3 E2E gain을 계산할 수 있다.
- `--lin 569`는 visual token을 이미 포함한 decoder input length로 해석한다. `--image_size 672`는 vision graph 비용 계산용이지 `lin`을 자동 변경하지 않는다.

확인할 것:

- 실행 crash 없음.
- `output.csv`가 생성되고 `required_cap_per_gpu`, `s_time`, `g_time (ms)`, `s_comm`, `s_x2g`, `g2g_comm` 컬럼이 존재.
- S2의 `required_cap_per_gpu`가 S1보다 작고, `g2g_comm`은 S2에서 non-zero.

### 15.3 Smoke: H100 S1/S2 PIM/Ramulator path

다음은 실제 `dgx-attacc` path smoke다.

```powershell
python main.py --system dgx-attacc --gpu H100 --ngpu 1 --tp 1 --num_attacc 1 --num_hbm 5 --interface NVLINK4 --model Qwen3-VL-4B --lin 569 --lout 128 --batch 1 --image_size 672 --prefill_chunk 512 --prefill_samples 8 --max_L 2048 --pipeopt --ffopt

python main.py --system dgx-attacc --gpu H100 --ngpu 2 --tp 2 --num_attacc 2 --num_hbm 5 --interface NVLINK4 --model Qwen3-VL-4B --lin 569 --lout 128 --batch 1 --image_size 672 --prefill_chunk 512 --prefill_samples 8 --max_L 2048 --pipeopt --ffopt
```

왜 돌리나:

- 로컬에서는 `pandas` 부재로 실제 PIM/Ramulator path를 검증하지 못했다.
- M6.1/M6.3/M6.4의 핵심은 sum-stage PIM score가 Ramulator trace를 타고, chunked sampled prefill과 `eff_lat`가 실제 E2E에 반영되는지 확인하는 것이다.

확인할 것:

- `ramulator.out`가 생성 또는 갱신된다.
- `ramulator.out`에 `max_L` 컬럼이 존재한다.
- `output.csv`에서 PIM run의 `s_matmul`이 0이 아니고, `s_comm`도 0이 아니다.
- PIM run에서 `s_softmax`는 0 또는 매우 작아야 한다. PIM softmax는 score trace에 포함되므로 별도 double-count하면 안 된다.
- S1은 all-reduce가 없어야 하므로 `g2g_comm`/prefill G2G가 S2보다 작다.

### 15.4 R3 corrected E2 gate

자동 gate helper:

```powershell
python tests\r3_gate.py
```

왜 돌리나:

- R3는 final_v1의 핵심 주장인 Qwen3-VL corrected E2 speedup을 검증한다.
- helper는 H100 S1/S2 각각에 대해 GPU-only baseline과 `dgx-attacc` proposal을 모두 실행하고 E2E gain과 interface/PIM ratio를 계산한다.

검증 기준:

- R3.S1 H100 x1: E2E gain `1.58x ± 20%`, 즉 `1.26x ~ 1.90x`.
- R3.S1 interface/PIM compute ratio: `0.5 ~ 0.7`.
- R3.S2 H100 x2: E2E gain `1.53x ± 20%`, 즉 `1.22x ~ 1.84x`.
- R3.S2 interface/PIM compute ratio: `0.2 ~ 0.4`.

결과 해석:

- gain이 target보다 낮으면 먼저 `s_comm / s_matmul`이 과도한지 본다. 과도하면 X2G/Q transmission 또는 NVLink 설정 문제다.
- gain이 target보다 높으면 `s_matmul`이 비정상적으로 작게 나온 것인지 확인한다. 이 경우 Ramulator cache가 잘못 재사용됐거나 `pim_numOp`/`max_L` row key가 어긋났을 수 있다.
- S2가 S1보다 무조건 크게 좋아져야 한다고 보면 안 된다. S2는 KV/PIM compute는 줄지만 all-reduce와 synchronization cost가 추가된다.

### 15.5 R2 AttAcc paper repro gate

자동 gate helper:

```powershell
python tests\r2_paper_repro.py
```

왜 돌리나:

- 새 GQA/VLM 패치가 legacy GPT-175B AttAcc paper reproduction을 깨지 않았는지 확인한다.
- R2는 modern VLM 결과를 주장하기 전에 simulator가 원 논문 기준 동작을 유지하는지 확인하는 regression gate다.

검증 기준:

- GPT-175B, A100a x8, TP=8, NVLink3, L=2048, batch=64 기준.
- DGX x AttAccs vs DGX_Base FP16 gain `4.84x ± 20%`가 must-pass다.
- helper 기본 target은 `4.84`, tolerance는 `0.20`이다.

결과 해석:

- R2가 실패하면 R3 결과를 논문 주장에 사용하면 안 된다.
- R2 gain이 낮으면 legacy MHA path, PIM softmax zero-return, Ramulator cache, `--pipeopt --ffopt` 적용 여부를 우선 확인한다.
- R2 gain이 높게 튀면 baseline GPU time이 과대평가됐거나 PIM trace cache가 잘못 매칭됐는지 확인한다.

### 15.6 결과 정리 형식

실행 결과는 raw `output.csv`를 보존하고, 별도 요약 표를 만든다. 파일명은 실행 날짜와 gate를 포함한다.

권장 파일:

- `results_h100_s1_baseline_YYYYMMDD.csv`
- `results_h100_s1_attacc_YYYYMMDD.csv`
- `results_h100_s2_baseline_YYYYMMDD.csv`
- `results_h100_s2_attacc_YYYYMMDD.csv`
- `results_r2_paper_repro_YYYYMMDD.md`
- `results_r3_corrected_e2_YYYYMMDD.md`

R3 요약 표:

```markdown
| Scenario | System | TP | Lin | Lout | Batch | s_time ms | g_time ms/token | E2E ms | Gain | s_matmul ms | s_comm ms | s_x2g ms | interface/PIM | required_cap_per_gpu GB | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S1 | dgx | 1 | 569 | 128 | 1 | | | | 1.00x | | | | | | baseline |
| S1 | dgx-attacc | 1 | 569 | 128 | 1 | | | | | | | | | | |
| S2 | dgx | 2 | 569 | 128 | 1 | | | | 1.00x | | | | | | baseline |
| S2 | dgx-attacc | 2 | 569 | 128 | 1 | | | | | | | | | | |
```

계산식:

```text
E2E ms = s_time + g_time(ms/token) * (Lout - 1)
Gain = E2E_baseline / E2E_attacc
interface/PIM = s_comm / s_matmul
required_cap_per_gpu GB = required_cap_per_gpu / 1024^3
```

R2 요약 표:

```markdown
| Model | System | GPU | TP | Lin | Lout | Batch | Word | E2E ms | Gain vs DGX | Target | Pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GPT-175B | dgx | A100a x8 | 8 | 2048 | 128 | 64 | 2 | | 1.00x | baseline | |
| GPT-175B | dgx-attacc | A100a x8 | 8 | 2048 | 128 | 64 | 2 | | | 4.84x ±20% | |
```

반드시 같이 기록할 로그:

- 실행 command 전체.
- git diff 또는 commit hash.
- Python version, pandas version.
- Ramulator binary 존재 여부와 `ramulator.out` row 수.
- `ramulator.out`를 새로 만든 run인지, 기존 cache를 재사용한 run인지.
- 실패 시 traceback 전체와 마지막으로 생성된 `output.csv`.
