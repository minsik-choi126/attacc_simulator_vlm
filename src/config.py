import os

from src.type import *


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        return float(raw)
    except ValueError:
        return default


SCALING_FACTOR = {}
SCALING_FACTOR['MAX_COMPUTE_UTIL'] = _env_float('ATTACC_MAX_COMPUTE_UTIL', 0.8)
SCALING_FACTOR['MAX_OFF_MEM_BW_UTIL'] = _env_float('ATTACC_MAX_OFF_MEM_BW_UTIL',
                                                   0.85)

# ENERGY_TABLE: pJ per byte
# Cache info: https://core.ac.uk/download/pdf/232142915.pdf
ENERGY_TABLE = {
    'GPU': {},
    'CPU': {},
    'PIM': {
        PIMType.BA: {},
        PIMType.BG: {},
        PIMType.BUFFER: {}
    }
}
ENERGY_TABLE['GPU']['reg'] = 0.0675
#4-way cache, ref: https://arxiv.org/pdf/1509.02308v1.pdf
ENERGY_TABLE['GPU'][ 'l1'] = 0.16 * 8
ENERGY_TABLE['GPU']['l2'] = 0.3 * 8
ENERGY_TABLE['GPU']['alu'] = 0.32
ENERGY_TABLE['GPU']['mem'] = (0.11 + 0.44 + 1.01 + 1.23 + 0.5 + 0.3) * 8
# ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
ENERGY_TABLE['GPU'][ 'comm'] = 1.3 * 8

## TODO: Add energy of CPU (pJ per byte)
ENERGY_TABLE['CPU']['reg'] = 0
ENERGY_TABLE['CPU']['l1'] = 0
ENERGY_TABLE['CPU']['l2'] = 0
ENERGY_TABLE['CPU']['alu'] = 0
ENERGY_TABLE['CPU']['mem'] = 0
ENERGY_TABLE['CPU']['comm'] = 0

## 2017 MICRO FGDRAM
## https://www.cs.utexas.edu/users/skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf
## Cell (ACT/PRE) energy: 0.11pJ/b,
## Cell (RD/WRT) energy: 0.44pJ/b,

## RD/WR Energy (column decoder to BG MUX): 1.01 pJ/b
## RD/WR Energy (BG Mux to GIO Mux): 1.23 pJ/b
## TSV energy : 0.5 pJ/b
## Silicon interposer IO energy : 0.3 pJ/b

## energy_table = [energy between DRAM cell and PE, energy between PE and buffer die

ENERGY_TABLE['PIM'][PIMType.BA]['mem'] = (0.11 +
                                          0.44) * 8  #, (1.01 + 1.23 + 0.5) * 8]
ENERGY_TABLE['PIM'][PIMType.BG]['mem'] = (0.11 + 0.44 +
                                          1.01) * 8  #, (1.23 + 0.5) * 8]
ENERGY_TABLE['PIM'][PIMType.BUFFER]['mem'] = (0.11 + 0.44 + 1.01 + 1.23 +
                                              0.5) * 8  #, 0]

ENERGY_TABLE['PIM'][PIMType.BA]['sram'] = 0.0034
ENERGY_TABLE['PIM'][PIMType.BG]['sram'] = 0.0034
ENERGY_TABLE['PIM'][PIMType.BUFFER]['sram'] = 0.0034

ENERGY_TABLE['PIM'][PIMType.BA]['alu'] = 0.32
ENERGY_TABLE['PIM'][PIMType.BG]['alu'] = 0.32
ENERGY_TABLE['PIM'][PIMType.BUFFER]['alu'] = 0.32

ENERGY_TABLE['PIM'][PIMType.BA]['io'] = [0.3, 0.5, 1.23, 1.01]
ENERGY_TABLE['PIM'][PIMType.BG]['io'] = [0.3, 0.5, 1.23, 1.01]
ENERGY_TABLE['PIM'][PIMType.BUFFER]['io'] = [0.3, 0.5, 1.23, 1.01]

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
ENERGY_TABLE['PIM'][PIMType.BA]['comm'] = 10.4
ENERGY_TABLE['PIM'][PIMType.BG]['comm'] = 10.4
ENERGY_TABLE['PIM'][PIMType.BUFFER]['comm'] = 10.4


def make_xpu_config(gpu_type: GPUType,
                    num_gpu=None,
                    flops=None,
                    mem_cap=None,
                    mem_bw=None,
                    power_constraint=True):
    config = {'GPU': {}, 'CPU': {}}
    config['GPU']["GPUTYPE"] = gpu_type
    config['GPU']["NUM_DEVICE"] = 8 if num_gpu is None else num_gpu

    if gpu_type == GPUType.A100a:
        # Ref: DGX-A100 whitepaper
        config['GPU']["NUM_CORE"] = 108
        config['GPU']["FLOPS_PER_DEVICE"] = 312 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 80 * 1024 * 1024 * 1024 \
                                                    if mem_cap is None else mem_cap

        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 3352 * 1000 * 1000 * 1000 \
                                                  if mem_bw is None else mem_bw
        config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        #config['GPU']["L2_MEM_BW_PER_DEVICE"] = 3.8 * 1000 * 1000 * 1000 * 1000
        config['GPU']["L1_CAP_PER_CORE"] = 192 * 1024
        config['GPU']["L2_CAP_PER_DEVICE"] = 40 * 1024 * 1024
        config['GPU']["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
        config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        config['CPU']["NUM_DEVICE"] = 2
        config['CPU']["NUM_CORE"] = 64
        config['CPU']["FLOPS_PER_DEVICE"] = 4 * 1000 * 1000 * 1000 * 1000
        config['CPU']["MEM_CAPACITY_PER_DEVICE"] = 1024 * 1024 * 1024 * 1024
        config['CPU']["OFF_MEM_BW_PER_DEVICE"] = 200 * 1000 * 1000 * 1000
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # TODO: Modify it
        config['CPU']["L1_CAP_PER_CORE"] = 96 * 1024
        config['CPU']["L2_CAP_PER_DEVICE"] = 256 * 1024 * 1024
        config['CPU']["INTERFACE_BW"] = 4 * 64 * 1000 * 1000 * 1000
        config['CPU']["ENERGY_TABLE"] = ENERGY_TABLE['CPU']

    elif gpu_type == GPUType.H100:
        # Ref: DGX-H100 whitepaper
        config['GPU']["NUM_CORE"] = 132
        config['GPU']["FLOPS_PER_DEVICE"] = 989.4 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 80 * 1024 * 1024 * 1024 \
                                                   if mem_cap is None else mem_cap
        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 3352 * 1000 * 1000 * 1000 \
                                                 if mem_bw is None else mem_bw
        config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # 5.5TB/s, https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
        #config['GPU']["L2_MEM_BW_PER_DEVICE"] = 5.5 * 1000 * 1000 * 1000 * 1000
        config['GPU']["L1_CAP_PER_CORE"] = 256 * 1024
        config['GPU']["L2_CAP_PER_DEVICE"] = 50 * 1024 * 1024
        # NVLINK: 900GB/s (Read 450GB/s Write 450GB/s)
        config['GPU']["INTERFACE_BW"] = 900 * 1000 * 1000 * 1000
        config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        # H100 DGX CPU configuration sapphire-rapids
        # https://www.servethehome.com/4th-gen-intel-xeon-scalable-sapphire-rapids-leaps-forward/7/
        config['CPU']["NUM_DEVICE"] = 2
        config['CPU']["NUM_CORE"] = 56
        # 4TFLOPS per CPU (half precision)
        config['CPU']["FLOPS_PER_DEVICE"] = 4 * 1000 * 1000 * 1000 * 1000
        # (2TB, dual processors)
        config['CPU']["MEM_CAPACITY_PER_DEVICE"] = 1024 * 1024 * 1024 * 1024
        # channels x dpc x 4400 MT/s  https://www.intel.com/content/www/us/en/products/sku/231746/intel-xeon-platinum-8480-processor-105m-cache-2-00-ghz/specifications.html
        config['CPU']["OFF_MEM_BW_PER_DEVICE"] = 8 * 2 * 4400 * (
            64 / 8) * 1000 * 1000
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # 5.5TB/s, https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = 5.5 * 1000 * 1000 * 1000 * 1000
        # TODO: Modify it
        config['CPU']["L1_CAP_PER_CORE"] = 48 * 1024
        config['CPU']["L2_CAP_PER_DEVICE"] = 2 * 1024 * 1024
        config['CPU']["INTERFACE_BW"] = 4 * 128 * 1000 * 1000 * 1000
        config['CPU']["ENERGY_TABLE"] = ENERGY_TABLE['CPU']

    elif gpu_type == GPUType.A6000:
        # Ref: NVIDIA RTX A6000 datasheet (Ampere GA102, GDDR6 48 GB)
        config['GPU']["NUM_CORE"] = 84
        config['GPU']["FLOPS_PER_DEVICE"] = 309.7 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 48 * 1024 * 1024 * 1024 \
                                                    if mem_cap is None else mem_cap
        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 768 * 1000 * 1000 * 1000 \
                                                  if mem_bw is None else mem_bw
        config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        config['GPU']["L1_CAP_PER_CORE"] = 128 * 1024
        config['GPU']["L2_CAP_PER_DEVICE"] = 6 * 1024 * 1024
        # NVLink Bridge (A6000): 112 GB/s aggregate, ~56 GB/s per direction
        config['GPU']["INTERFACE_BW"] = 112 * 1000 * 1000 * 1000
        config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        # Workstation CPU placeholder (Sapphire-Rapids analog)
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

    return config


# Rank x BG x BA / 2 (tCCD)
BW_SCALE = {
    False: {
        PIMType.BA: 2 * 4 * 4 / 2,
        PIMType.BG: 2 * 4,
        PIMType.BUFFER: 1
    },
    True: {
        PIMType.BA: 9,
        PIMType.BG: 3,
        PIMType.BUFFER: 1
    }
}


def make_pim_config(pim_type: PIMType,
                    interface_type: InterfaceType,
                    opb=1,
                    num_attacc=8,
                    num_hbm=5,
                    bw_scale=None,
                    power_constraint=False):
    config = {}
    config["PIM_TYPE"] = pim_type
    config["POWER_CONSTRAINT"] = power_constraint
    config["ENERGY_TABLE"] = ENERGY_TABLE['PIM'][pim_type]

    internal_bandwidth_scale =  BW_SCALE[power_constraint][pim_type] \
                                if bw_scale is None else bw_scale
    config["NUM_ATTACC"] = num_attacc
    config["NUM_HBM"] = num_hbm
    config["MEM_CAPACITY_PER_HBM"] = 16 * 1024 * 1024 * 1024
    config[
        "MEM_BW_PER_HBM"] = 670.4 * 1000 * 1000 * 1000 * internal_bandwidth_scale
    config["FLOPS_PER_HBM"] = config["MEM_BW_PER_HBM"] * opb
    config["SOFTMAX_MEM_BW"] = 670.4 * 1000 * 1000 * 1000 * num_hbm
    config["SOFTMAX_FLOPS"] = config["SOFTMAX_MEM_BW"]

    if interface_type == InterfaceType.NVLINK3:
        config["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.NVLINK4:
        config["INTERFACE_BW"] = 900 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.PCIE4:
        config["INTERFACE_BW"] = 64 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.PCIE5:
        config["INTERFACE_BW"] = 128 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.NVLINK_BRIDGE:
        # RTX A6000 NVLink Bridge: 112 GB/s aggregate
        config["INTERFACE_BW"] = 112 * 1000 * 1000 * 1000
    else:
        assert 0, "Invalid interface type"

    return config


def make_model_config(name, dtype):
    model_table = {}
    model_table['GPT-175B'] = [96, 12288, 96, 128, 4, 1]
    model_table['GPT-89B'] = [48, 12288, 96, 128, 4, 1]
    model_table['GPT-13B'] = [40, 5120, 40, 128, 4, 1]
    model_table['LLAMA-7B'] = [32, 4096, 32, 128, 8 / 3, 1]
    model_table['LLAMA-65B'] = [80, 8192, 64, 128, 8 / 3, 1]
    model_table['MT-76B'] = [60, 10240, 40, 128, 4, 1]
    model_table['MT-146B'] = [80, 12288, 80, 128, 4, 1]
    model_table['MT-310B'] = [96, 16384, 128, 128, 4, 1]
    model_table['MT-530B'] = [105, 20480, 128, 160, 4, 1]
    model_table['MT-1008B'] = [128, 25600, 160, 160, 4, 1]
    model_table['OPT-66B'] = [64, 9216, 72, 128, 4, 1]

    model_table['Qwen3-4B'] = {
        'ndec': 36,
        'hdim': 2560,
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'dhead': 128,
        'ff_intermediate': 9728,
        'ffn_type': 'gated',
        'activation': 'silu',
    }
    model_table['Qwen3-VL-4B'] = {
        'ndec': 36,
        'hdim': 2560,
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'dhead': 128,
        'ff_intermediate': 9728,
        'ffn_type': 'gated',
        'activation': 'silu',
        'vit_layers': 24,
        'vit_hidden': 1024,
        'vit_num_heads': 16,
        'vit_intermediate': 4096,
        'vit_out_hidden': 2560,
        'vit_activation': 'gelu_pytorch_tanh',
        'patch_size': 16,
        'image_size_default': 672,
        'spatial_merge_size': 2,
        'num_vis_tokens_per_image': 441,
        'projector_type': 'mlp_with_merger',
        'has_deepstack': True,
        'deepstack_layers': [5, 11, 17],
        'is_anyres': False,
        'image_grid_pinpoints': [],
        'use_image_newline_parameter': False,
        'is_concat_style': True,
        'is_cross_attn': False,
        # Fix B: VLM-side floor overhead -- preprocessing, RoPE,
        # CUDA kernel launches, vLLM scheduler.  Calibrated from
        # LLaVA-1.5 (measured 41.2 - simulated 18.7 = 22.5 ms).
        # Treat as HW-independent first-order constant; tune per-model
        # after Fix A baseline if cross-HW calibration shows drift.
        'vlm_floor_overhead_ms': 22.0,
    }
    model_table['Qwen2.5-VL-7B'] = {
        'ndec': 28,
        'hdim': 3584,
        'num_q_heads': 28,
        'num_kv_heads': 4,
        'dhead': 128,
        'ff_intermediate': 18944,
        'ffn_type': 'gated',
        'activation': 'silu',
        'vit_layers': 32,
        'vit_hidden': 1280,
        'vit_num_heads': 16,
        'vit_intermediate': 3420,
        'vit_out_hidden': 3584,
        'vit_activation': 'silu',
        'patch_size': 14,
        'image_size_default': 672,
        'spatial_merge_size': 2,
        'num_vis_tokens_per_image': 576,
        'projector_type': 'mlp_with_merger',
        'has_deepstack': False,
        'deepstack_layers': [],
        'is_anyres': False,
        'image_grid_pinpoints': [],
        'use_image_newline_parameter': False,
        'is_concat_style': True,
        'is_cross_attn': False,
        # Fix B: VLM-side floor overhead -- preprocessing, RoPE,
        # CUDA kernel launches, vLLM scheduler.  Calibrated from
        # LLaVA-1.5 (measured 41.2 - simulated 18.7 = 22.5 ms).
        # Treat as HW-independent first-order constant; tune per-model
        # after Fix A baseline if cross-HW calibration shows drift.
        'vlm_floor_overhead_ms': 22.0,
    }
    model_table['InternVL3-8B-hf'] = {
        'ndec': 28,
        'hdim': 3584,
        'num_q_heads': 28,
        'num_kv_heads': 4,
        'dhead': 128,
        'ff_intermediate': 18944,
        'ffn_type': 'gated',
        'activation': 'silu',
        'vit_layers': 24,
        'vit_hidden': 1024,
        'vit_num_heads': 16,
        'vit_intermediate': 4096,
        'vit_out_hidden': 3584,
        'vit_activation': 'gelu',
        'patch_size': 14,
        'image_size_default': 448,
        'spatial_merge_size': 2,
        'num_vis_tokens_per_image': 256,
        'projector_type': 'pixel_shuffle_mlp',
        'has_deepstack': False,
        'deepstack_layers': [],
        'is_anyres': False,
        'image_grid_pinpoints': [],
        'use_image_newline_parameter': False,
        'is_concat_style': True,
        'is_cross_attn': False,
        # Fix B: VLM-side floor overhead -- preprocessing, RoPE,
        # CUDA kernel launches, vLLM scheduler.  Calibrated from
        # LLaVA-1.5 (measured 41.2 - simulated 18.7 = 22.5 ms).
        # Treat as HW-independent first-order constant; tune per-model
        # after Fix A baseline if cross-HW calibration shows drift.
        'vlm_floor_overhead_ms': 22.0,
    }
    model_table['Vicuna-7B'] = {
        'ndec': 32,
        'hdim': 4096,
        'num_q_heads': 32,
        'num_kv_heads': 32,
        'dhead': 128,
        'ff_intermediate': 11008,
        'ffn_type': 'gated',
        'activation': 'silu',
    }
    model_table['LLaVA-1.5-7B'] = {
        **model_table['Vicuna-7B'],
        'vit_layers': 24,
        'vit_hidden': 1024,
        'vit_num_heads': 16,
        'vit_intermediate': 4096,
        'vit_out_hidden': 4096,
        'vit_activation': 'gelu',
        'patch_size': 14,
        'image_size_default': 336,
        'spatial_merge_size': 1,
        'num_vis_tokens_per_image': 576,
        'projector_type': 'mlp',
        'has_deepstack': False,
        'deepstack_layers': [],
        'is_anyres': False,
        'image_grid_pinpoints': [],
        'use_image_newline_parameter': False,
        'is_concat_style': True,
        'is_cross_attn': False,
        # Fix B: VLM-side floor overhead -- preprocessing, RoPE,
        # CUDA kernel launches, vLLM scheduler.  Calibrated from
        # LLaVA-1.5 (measured 41.2 - simulated 18.7 = 22.5 ms).
        # Treat as HW-independent first-order constant; tune per-model
        # after Fix A baseline if cross-HW calibration shows drift.
        'vlm_floor_overhead_ms': 22.0,
    }
    model_table['Mistral-7B'] = {
        'ndec': 32,
        'hdim': 4096,
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'dhead': 128,
        'ff_intermediate': 14336,
        'ffn_type': 'gated',
        'activation': 'silu',
    }
    model_table['LLaVA-Next-Mistral-7B'] = {
        **model_table['Mistral-7B'],
        'vit_layers': 24,
        'vit_hidden': 1024,
        'vit_num_heads': 16,
        'vit_intermediate': 4096,
        'vit_out_hidden': 4096,
        'vit_activation': 'gelu',
        'patch_size': 14,
        'image_size_default': 336,
        'spatial_merge_size': 1,
        'num_vis_tokens_per_image': 2880,
        'projector_type': 'mlp',
        'has_deepstack': False,
        'deepstack_layers': [],
        'is_anyres': True,
        'image_grid_pinpoints': [[336, 672], [672, 336], [672, 672],
                                 [1008, 336], [336, 1008]],
        'use_image_newline_parameter': True,
        'is_concat_style': True,
        'is_cross_attn': False,
        # Fix B: VLM-side floor overhead -- preprocessing, RoPE,
        # CUDA kernel launches, vLLM scheduler.  Calibrated from
        # LLaVA-1.5 (measured 41.2 - simulated 18.7 = 22.5 ms).
        # Treat as HW-independent first-order constant; tune per-model
        # after Fix A baseline if cross-HW calibration shows drift.
        'vlm_floor_overhead_ms': 22.0,
    }

    entry = model_table[name]
    if isinstance(entry, list):
        ndec, hdim, nheads, dhead, ff_scale, gqa_size = entry
        config = {
            'name': name,
            'ndec': ndec,
            'hdim': hdim,
            'num_heads': nheads,
            'num_q_heads': nheads,
            'num_kv_heads': int(nheads / gqa_size),
            'dhead': dhead,
            'ff_scale': ff_scale,
            'gqa_size': gqa_size,
            'dtype': dtype,
            'ffn_type': 'gated' if 'LLAMA' in name else 'standard',
            'activation': 'silu' if 'LLAMA' in name else 'gelu',
        }
        llama_ff_intermediate = {
            'LLAMA-7B': 11008,
            'LLAMA-65B': 22016,
        }
        if name in llama_ff_intermediate:
            config['ff_intermediate'] = llama_ff_intermediate[name]
    else:
        config = dict(entry)
        config['name'] = name
        config['num_heads'] = config['num_q_heads']
        config['gqa_size'] = int(config['num_q_heads'] / config['num_kv_heads'])
        config['ff_scale'] = config['ff_intermediate'] / config['hdim']
        config['dtype'] = dtype

    config['q_proj_out'] = config['num_q_heads'] * config['dhead']
    config['kv_proj_out'] = config['num_kv_heads'] * config['dhead']
    config['qkv_proj_out_total'] = config['q_proj_out'] + 2 * config[
        'kv_proj_out']
    config.setdefault('has_deepstack', False)
    config.setdefault('deepstack_layers', [])
    config.setdefault('is_anyres', False)
    config.setdefault('image_grid_pinpoints', [])
    config.setdefault('use_image_newline_parameter', False)
    config.setdefault('is_concat_style', False)
    config.setdefault('is_cross_attn', False)
    return config
