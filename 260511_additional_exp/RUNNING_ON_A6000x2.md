# Running Guide: RTX A6000 x 2

본 fork의 실험을 RTX A6000 x 2 환경에서 돌리는 step-by-step 가이드.
H100 deployment guide와의 차이점만 정리.

---

## 1. A6000 vs H100 (실제 차이)

| 항목 | RTX A6000 (Ampere) | H100 SXM5 (Hopper) | 비고 |
|---|---|---|---|
| Memory | 48 GB GDDR6 | 80 GB HBM3 | A6000 GDDR6 (non-HBM) |
| Memory BW | 768 GB/s | 3,352 GB/s | A6000가 **4.4x 느림** |
| FP16 Tensor (dense) | 309.7 TFLOPS | 989.4 TFLOPS | A6000가 **3.2x 느림** |
| FP8 Tensor Engine | 없음 | 1979 TFLOPS | A6000 = Ampere, FP8 없음 (이미 scope 제외) |
| NVLink | NVLink Bridge 112.5 GB/s (양방향) | NVLink4 900 GB/s | A6000가 **8x 느림** |
| CUDA Cores / SMs | 10752 / 84 | 16896 / 132 | |
| L2 cache | 6 MB | 50 MB | |
| Driver 535+ 호환 | OK | OK | |
| vLLM 0.7.3 | OK | OK | |
| NCCL TP=2 | OK (NVLink Bridge) | driver 545+ 필요 | A6000은 NCCL 안정 |

**Paper-relevant 함의**:

- A6000 GPU 자체가 H100보다 느림 -> **AttAcc PIM offload의 상대적 gain은
  더 커짐** (memory-bound 부분이 더 dominant)
- AttAcc PIM은 여전히 HBM3 기반 hypothetical 가속기로 simulator에서 모델링.
  GPU baseline만 A6000으로 교체
- TP=2가 driver 535에서도 NCCL 정상 동작 -> **S2 measurement도 실측 가능**
  (H100 + driver 535에서 못 했던 측정)

---

## 2. Simulator 변경 (필수) — *완료, 참고용*

> **STATUS (2026-05-11):** 본 섹션의 모든 패치는 이미 코드베이스에 적용되어
> 있다. `--gpu A6000`, `--interface NVLINK_BRIDGE` 가 그대로 동작하며,
> `sim_runner.py` 기본값도 A6000/NVLink Bridge 로 변경됨. `--gmemcap`
> 기본값은 `None` 으로 바뀌어 A6000 의 48 GB spec 이 그대로 사용됨
> (override 시 명시적으로 `--gmemcap 48` 등 지정). 본 섹션은 변경 내역을
> 추적할 수 있도록 그대로 남겨 둠 (historical reference).
>
> 본문에서 "필요" / "추가해야 함" 같은 표현은 패치 적용 *이전* 시점의
> 설명이다. 이미 적용 완료된 상태에서는 단순히 어떤 변경이 들어갔는지를
> 보여 주는 changelog 로 읽으면 된다.

원래 simulator는 `--gpu A100a` / `--gpu H100`만 지원했으며, 아래 변경들로
`--gpu A6000` 경로가 추가되었다.

### 2-1. Patch: `src/type.py`

```python
class GPUType(Enum):
    A100a = 0
    H100  = 1
    A6000 = 2     # NEW
```

### 2-2. Patch: `src/config.py` `make_xpu_config()`

기존 `elif gpu_type == GPUType.H100:` 블록 다음에 추가:

```python
    elif gpu_type == GPUType.A6000:
        # Ref: NVIDIA RTX A6000 datasheet (Ampere GA102)
        config['GPU']["NUM_CORE"] = 84               # SMs
        config['GPU']["FLOPS_PER_DEVICE"] = 309.7 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 48 * 1024 * 1024 * 1024 \
                                                   if mem_cap is None else mem_cap
        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 768 * 1000 * 1000 * 1000 \
                                                 if mem_bw is None else mem_bw
        config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        config['GPU']["L1_CAP_PER_CORE"] = 128 * 1024
        config['GPU']["L2_CAP_PER_DEVICE"] = 6 * 1024 * 1024
        # NVLink Bridge: 112.5 GB/s aggregate, 56.25 GB/s per-direction
        config['GPU']["INTERFACE_BW"] = 112 * 1000 * 1000 * 1000
        config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        # Typical workstation CPU (not central; use Sapphire-Rapids analog)
        config['CPU']["NUM_DEVICE"] = 2
        config['CPU']["NUM_CORE"] = 32
        config['CPU']["FLOPS_PER_DEVICE"] = 4 * 1000 * 1000 * 1000 * 1000
        config['CPU']["MEM_CAPACITY_PER_DEVICE"] = 512 * 1024 * 1024 * 1024
        config['CPU']["OFF_MEM_BW_PER_DEVICE"] = 200 * 1000 * 1000 * 1000
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        config['CPU']["L1_CAP_PER_CORE"] = 64 * 1024
        config['CPU']["L2_CAP_PER_DEVICE"] = 64 * 1024 * 1024
        config['CPU']["INTERFACE_BW"] = 4 * 64 * 1000 * 1000 * 1000
        config['CPU']["ENERGY_TABLE"] = ENERGY_TABLE['CPU']
```

### 2-3. Patch: `main.py` argparse

기존:
```python
    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    else:
        assert 0
```

다음으로 교체:
```python
    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    elif args.gpu == 'A6000':
        gpu_device = GPUType.A6000
    else:
        assert 0, "Unsupported --gpu: {}".format(args.gpu)
```

### 2-4. New Interface enum (적용됨)

A6000 NVLink Bridge는 NVLink3/4와 다른 등급이므로 별도 enum으로 적용되어 있음:

```python
# src/type.py
class InterfaceType(Enum):
    NVLINK4 = 0
    NVLINK3 = 1
    PCIE4 = 2
    PCIE5 = 3
    NVLINK_BRIDGE = 4    # NEW (A6000 등 workstation NVLink)
```

`src/config.py` `make_pim_config()`:
```python
    elif interface_type == InterfaceType.NVLINK_BRIDGE:
        config["INTERFACE_BW"] = 112 * 1000 * 1000 * 1000
```

(생략 시 기존 `--interface NVLINK3 600 GB/s`도 운영 가능. 단 실제 g2g
overhead가 1/5 정도로 underestimate됨.)

---

## 3. Deployment Scenarios for A6000 x 2

| Scenario | GPUs | TP | NUM_ATTACC | NUM_HBM | Inter-GPU | PIM aggregate |
|---|---|---|---|---|---|---|
| **A1: A6000 x 1** | 1 | 1 | 1 | 5 | none | 18.1 TB/s |
| **A2: A6000 x 2** | 2 | 2 | 2 | 5 | NVLink Bridge 112 GB/s | 36.2 TB/s |

PIM aggregate는 H100 시나리오와 동일 (PIM은 hypothetical HBM3 모듈이므로
host GPU와 무관). GPU baseline의 BW/FLOPS가 약해지는 만큼 **PIM relative
gain은 H100보다 더 큼**.

main.py assert `num_attacc == tp == ngpu` 동일 적용.

---

## 4. Setup (한 번만)

```bash
# 1. Repo clone
cd /your/workspace
git clone https://github.com/minsik-choi126/attacc_simulator_vlm.git
cd attacc_simulator_vlm
git submodule update --init --recursive

# 2. Simulator patches -- 이미 적용되어 있음 (sec.2 STATUS 참고).
#    별도 수정 없이 다음 단계로 진행.

# 3. Python deps
pip install -r requirements.txt

# 4. Ramulator2 build
bash set_pim_ramulator.sh
cd ramulator2 && mkdir -p build && cd build
cmake .. && make -j$(nproc)
cp ramulator2 ../ramulator2 && cd ../..

# 5. vLLM (measurement용)
pip install vllm==0.7.3 torch==2.5.1 transformers==4.49.0 mistral_common==1.4.4

# 6. 환경 검증
nvidia-smi -L                              # 두 GPU 인식 확인
python -c "import torch; print(torch.cuda.device_count())"   # 2
python -c "import vllm; print('vllm OK')"
test -x ramulator2/ramulator2 && echo "ramulator OK"
```

NCCL TP=2 sanity check (A6000 x 2 NVLink Bridge 정상 동작 확인):
```bash
python -c "
import torch
import torch.distributed as dist
import os
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '29500')
# Single-process check
print('CUDA devices:', torch.cuda.device_count())
print('NCCL available:', dist.is_nccl_available())
print('Per-GPU memory:', [torch.cuda.get_device_properties(i).total_memory / 1024**3
                          for i in range(torch.cuda.device_count())])
"
```

---

## 5. 실행 명령 (A6000-specific)

### 5-1. Tier 1 simulator gate (A6000 baseline 기반)

```bash
cd 260511_additional_exp

# R2 paper repro -- simulator-only, 하드웨어 무관 (DGX-A100 simulated)
python tier1_simulator/r2_paper_repro.py

# Upstream legacy LLM regression (DGX-A100 simulated)
python tier1_simulator/upstream_baseline.py

# Multi-VLM matrix -- 스크립트는 이미 A6000/NVLink Bridge 기본값 사용
python tier1_simulator/multi_vlm_full_sim.py

# Ablation (Qwen3-VL-4B, A6000 deployment)
python tier1_simulator/ablation_contribution.py

# ViT calibration
python tier1_simulator/vit_recalibration.py
```

### 5-2. Tier 2 simulator (대부분 default 그대로)

```bash
python tier2_simulator/chunk_size_sweep.py        # 5 모델 x 8 chunk
python tier2_simulator/routing_mode_compare.py    # 4 모델 x 3 mode
python tier2_simulator/eff_lat_ablation.py
python tier2_simulator/nvlink_compare.py          # NVLink_Bridge 옵션 추가 권장
python tier2_simulator/roofline_per_vlm.py        # 분석적, GPU type 무관
python tier2_simulator/capacity_regime.py         # 분석적, GPU type 무관
python tier2_simulator/pim_mode_compare.py
python tier2_simulator/slo_throughput.py
python tier2_simulator/w4a16_pim_sim.py
python tier2_simulator/sensitivity_sweep.py       # 240 configs, 가장 김
```

### 5-3. Tier 2 measurement (A6000 x 2, vLLM 0.7.3)

```bash
HF_HOME=/your/cache python tier2_measurement/w4a16_awq_measure.py \
    --image_size 672 --lout 128 --repeats 4

HF_HOME=/your/cache python tier2_measurement/w8a16_gptq_measure.py \
    --image_size 672 --lout 128 --repeats 4

HF_HOME=/your/cache python tier2_measurement/quant_stability_test.py --n_runs 50

HF_HOME=/your/cache python tier2_measurement/image_size_sweep.py \
    --sizes 336 448 672 1008 --repeats 4

HF_HOME=/your/cache python tier2_measurement/prompt_pattern_matrix.py \
    --repeats 3

HF_HOME=/your/cache python tier2_measurement/vllm_bf16_baseline_aligned.py \
    --repeats 3

# A6000 x 2 TP=2 BF16 baseline (driver 535 + NVLink Bridge 조합에서
# H100 stack이 못 했던 측정. --tp 2 를 명시적으로 넘겨 줘야 한다.
# 기본값은 --tp 1; A2 시나리오는 항상 --tp 2 로 명시.)
HF_HOME=/your/cache python tier2_measurement/vllm_bf16_baseline_aligned.py \
    --models Qwen/Qwen2.5-VL-7B-Instruct \
    --tp 2 \
    --repeats 3
```

---

## 6. 스크립트 정합 상태 (참고)

Tier 1 / Tier 2 simulator 스크립트 + Tier 2 measurement 스크립트는
**이미** A6000 / NVLink Bridge / `--tp` 인자를 지원하도록 패치 완료.

### 6-1. Simulator scripts (`tier1_simulator/`, `tier2_simulator/`)

`shared/sim_runner.py` 의 기본값이 `gpu="A6000"`, `interface="NVLINK_BRIDGE"`
이며, 모든 스크립트가 이 sim_runner 를 통해 호출되므로 별도 수정 없이
바로 A6000 deployment 로 실행된다.

A100a / H100 에서 돌리고 싶을 때는 `sr.run(..., gpu="H100", interface="NVLINK4")`
처럼 명시적으로 override 한다 (`r2_paper_repro.py`, `upstream_baseline.py`
가 이 패턴을 유지).

### 6-2. Measurement scripts (`tier2_measurement/`)

6개 스크립트 모두 `--tp` argparse 인자를 가진다 (기본 `1`). A6000 x 2
실측 시에는 다음과 같이 호출:

```bash
python tier2_measurement/vllm_bf16_baseline_aligned.py --tp 2 --repeats 3
python tier2_measurement/w4a16_awq_measure.py            --tp 2 --repeats 4
python tier2_measurement/w8a16_gptq_measure.py           --tp 2 --repeats 4
python tier2_measurement/image_size_sweep.py             --tp 2
python tier2_measurement/prompt_pattern_matrix.py        --tp 2
python tier2_measurement/quant_stability_test.py         --tp 2
```

> `--tp 2` 는 현재 시점에서는 *test / validation* 용도이며, 기본 측정
> 흐름은 `--tp 1` 로 잡혀 있다. TP=2 결과의 paper 채택 여부는 NCCL
> 안정성 + 분산 capture 일치 여부 확인 후 결정.

---

## 7. Paper-grade Pass Criteria 조정

A6000 baseline 약하므로 일부 target 재계산:

### 7-1. R2 paper repro (변동 없음)

`r2_paper_repro.py`는 DGX-A100 x 8 simulated. 실행 환경 (A6000) 무관.
target 4.84x must-pass 그대로.

### 7-2. R3 corrected E2 (A6000 deployment)

기존 H100 target (S1 1.58x, S2 1.53x)은 H100 baseline 가정.
A6000 baseline일 때 simulator 결과는 더 큰 speedup 예상 (GPU가 느리므로 PIM 상대 이득 ↑).

권장 절차:
1. A6000 deployment 적용한 simulator로 corrected E2 측정
2. 측정된 simulator gain을 paper target으로 documented (예: "A1 baseline:
   1.8-2.5x ±20%, A2: 1.7-2.4x ±20%" — 실측 후 narrow)

### 7-3. eff_lat 표

`paper sec.6.1` eff_lat은 model의 n_kv 와 num_hbm만 사용. GPU type 무관.
sec.0.4 표 그대로 유효:
```
Qwen3-VL-4B (n_kv=8): A1 0.80 / A2 0.57
Qwen2.5-VL-7B (n_kv=4): A1 0.57 / A2 0.29
LLaVA-1.5-7B (n_kv=32): A1 0.91 / A2 0.80
```

### 7-4. Capacity regime (변경 거의 없음)

`capacity_regime.py`는 `MEM_CAPACITY_PER_DEVICE`만 의존. A6000 48GB
대비 H100 80GB:

| 모델 | H100 80GB max_batch | A6000 48GB max_batch (대략) |
|---|---|---|
| Qwen3-VL-4B S1 | ~875 | ~520 |
| Qwen3-VL-4B S2 | ~1850 | ~1100 |
| LLaVA-1.5-7B S1 | ~184 | ~110 |
| LLaVA-Next S1 | ~170 | ~100 |

capacity_regime.py를 A6000 spec으로 재실행 후 paper Tab.2 갱신.

---

## 8. 권장 실행 순서 (A6000 x 2 환경)

```bash
# 0. Setup (한 번)
# - 위 sec.4 단계 완료

# 1. Simulator patch 적용 (위 sec.2)
# - vim src/type.py src/config.py main.py
# - patches 적용

# 2. Sim-runner default A6000 전환 (위 sec.6-2)
# - vim shared/sim_runner.py

# 3. Simulator gate run
cd 260511_additional_exp
bash run_all_h100x1.sh --tier 1     # multi_vlm_full_sim 등 A6000 deployment로 실행됨
bash run_all_h100x1.sh --tier 2sim  # 모든 simulator 분석

# 4. Tier 2 measurement (TP=1)
HF_HOME=/your/cache bash run_all_h100x1.sh --tier meas

# 5. TP=2 measurement (A6000 x 2, 신규)
# - vllm_bf16_baseline_aligned.py에 --tp 인자 추가 후
HF_HOME=/your/cache python tier2_measurement/vllm_bf16_baseline_aligned.py \
    --tp 2 --repeats 3
HF_HOME=/your/cache python tier2_measurement/w4a16_awq_measure.py \
    --tp 2 --repeats 4   # 동일 패턴으로 인자 추가 후

# 6. 결과 종합
ls results/*.json | head -30
cat results/r2_paper_repro.json | python -m json.tool | head -50
```

---

## 9. 주의사항 / Caveat (paper에 명시)

- **A6000은 GDDR6, AttAcc는 HBM3 hypothetical** -- AttAcc는 host GPU와
  독립 모델이고, GPU baseline만 A6000으로 측정. "A6000 + AttAcc"는
  hypothetical projection (실제 hardware로 존재하지 않음).
- **AttAcc PIM aggregate BW는 변하지 않음** (18.1 TB/s for 1 AttAcc),
  단 GPU baseline이 약해 **relative gain은 H100보다 큼** -- paper에서
  "lower-tier GPU에서 AttAcc 이득이 더 큼" 식 narrative 가능.
- **NVLink Bridge 112 GB/s vs NVLink4 900 GB/s** -- TP=2 comm 비중이
  훨씬 큼. eff_lat S2 0.29 (Qwen2.5-VL) 같은 시나리오에서 NVLink Bridge
  bottleneck 추가 영향 있음. nvlink_compare.py 결과로 quantify.
- **DGX_Large baseline R2 skip** 그대로 (CLI 미모델링) -- A6000 환경 영향 없음.
- **FP8 out of scope** (이미 결정) -- Ampere는 FP8 transformer engine
  없으므로 자연스럽게 제외.
- **A6000 x 2 NVLink Bridge는 펌웨어/케이블 의존** -- 일부 보드는 NVLink
  미장착. `nvidia-smi topo --matrix` 로 확인 후 NVLink 표시 없으면 PCIe
  fallback (64 GB/s). PCIe fallback인 경우 `--interface PCIE4` 사용.

```bash
nvidia-smi topo --matrix
# GPU0 -- NV1 -- GPU1   : NVLink OK
# GPU0 -- PIX -- GPU1   : PCIe만 (NVLink Bridge 미장착)
```

---

## 10. Sanity Test (먼저)

simulator patch 적용 후 5분 sanity:

```bash
# patches 적용했는지 검증
python -c "from src.type import GPUType; assert hasattr(GPUType, 'A6000'); print('GPUType.A6000 OK')"

# Smoke run: small GPT-13B A6000 simulator
python main.py --gpu A6000 --ngpu 1 --tp 1 --num_attacc 1 --num_hbm 5 \
    --interface NVLINK_BRIDGE \
    --model GPT-13B --lin 16 --lout 4 --batch 1 \
    --system dgx 2>&1 | tail -3

# capacity_regime: simulator-only, A6000 spec로 max_batch 재계산
python 260511_additional_exp/tier2_simulator/capacity_regime.py
```

기대 결과:
- GPT-13B smoke: latency 출력 (H100 5.51ms -> A6000 ~15-20ms 예상)
- capacity_regime: per-GPU max_batch가 H100 대비 약 60% (48/80) 수준

---

## 11. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `assert 0` in main.py | `--gpu` 값이 enum 에 없음 (예: 오타) | `--gpu A6000` / `A100a` / `H100` 중 하나로 지정 |
| `AttributeError: GPUType.A6000` | 구버전 체크아웃 (패치 이전) | `git pull` 로 본 commit 동기화 |
| `InterfaceType.NVLINK_BRIDGE` 없음 | 구버전 체크아웃 | `git pull`. 임시 우회는 `--interface NVLINK3` |
| vLLM `RuntimeError: NCCL` on TP=2 | NVLink Bridge 미장착 또는 NCCL 버전 | `NCCL_P2P_DISABLE=1` 환경변수, 또는 `--tp 1` 로 fallback |
| capacity 예상보다 큼 (80 GB로 잡힘) | `--gmemcap 80` 가 명시적으로 들어감 | `--gmemcap` 생략 (default=None → 48 GB) |
| nvlink_compare 결과에 NVLink Bridge 행이 없음 | 구버전 체크아웃 | `git pull` (현재 `INTERFACES`에 `NVLINK_BRIDGE` 포함) |

---

## 12. Minimum Paper-grade Set (A6000 x 2 기준)

reviewer 대응에 충분한 최소 결과:

- [ ] R2 paper repro PASS (simulator-only, hardware 무관)
- [ ] Multi-VLM simulator matrix (5 VLM x dgx vs dgx-attacc, A6000 baseline)
- [ ] Ablation contribution
- [ ] Roofline per VLM
- [ ] Capacity regime (A6000 48GB)
- [ ] vLLM TP=1 measurement (5 measurement scripts)
- [ ] **vLLM TP=2 measurement (A6000 x 2 신규!)** -- driver 535 H100에서 못 했던 측정
- [ ] eff_lat caveat (paper sec.6.1)
- [ ] W4A16 + W8A16 quant measurement

위 9 항목으로 paper의 main figure / table 데이터 다 확보 가능.

H100 측 결과와 비교용으로 prior `results/r6_*` `r9_*` 그대로 reuse 가능.
