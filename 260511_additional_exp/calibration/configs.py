"""Shared (model, image_size, lin) matrix for calibration.

Same matrix used on A6000 and H100 so cross-HW comparison aligns 1:1.

Each entry produces one (simulator, vLLM measurement) pair per batch.
"""

LOUT = 128
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
# Capacity-bound models (LLaVA-1.5 MHA, LLaVA-Next anyres) will OOM on
# A6000 48GB above ~80 batch; vLLM side catches CUDA OOM and marks the
# cell. Simulator side has no OOM enforcement so it runs all configs
# (which is informative anyway -- shows scaling beyond physical limit).

# (sim_model, hf_model_id, label, image_size, lin)
VLM_CONFIGS = [
    # Qwen3-VL-4B -- requires driver 545+; vLLM 0.7.3 may not load it.
    ("Qwen3-VL-4B",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen3-VL-4B",     336, 569),
    ("Qwen3-VL-4B",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen3-VL-4B",     672, 569),
    # Qwen2.5-VL-7B -- main calibration target (s_corr 7.9x today).
    ("Qwen2.5-VL-7B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-7B",   336, 704),
    ("Qwen2.5-VL-7B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-7B",   672, 704),
    # LLaVA-1.5-7B -- fixed 336 res (no anyres).
    ("LLaVA-1.5-7B",
        "llava-hf/llava-1.5-7b-hf",
        "LLaVA-1.5-7B",    336, 704),
    # LLaVA-Next-Mistral-7B -- anyres path, multi-tile.
    ("LLaVA-Next-Mistral-7B",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "LLaVA-Next-Mistral-7B", 336, 704),
    ("LLaVA-Next-Mistral-7B",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "LLaVA-Next-Mistral-7B", 672, 3008),
    # InternVL3-8B-hf -- requires driver 545+; may not load on vLLM 0.7.3.
    ("InternVL3-8B-hf",
        "OpenGVLab/InternVL3-8B-hf",
        "InternVL3-8B-hf", 448, 704),
]


# Simulator GPU/interface mapping per detected hardware.
HW_TO_SIM = {
    "A6000":  {"gpu": "A6000",  "interface": "NVLINK_BRIDGE"},
    "H100":   {"gpu": "H100",   "interface": "NVLINK4"},
    "A100":   {"gpu": "A100a",  "interface": "NVLINK3"},
}


def grid():
    """Yield (sim_model, hf_id, label, image_size, lin, batch) tuples."""
    for cfg in VLM_CONFIGS:
        for b in BATCHES:
            yield (*cfg, b)
