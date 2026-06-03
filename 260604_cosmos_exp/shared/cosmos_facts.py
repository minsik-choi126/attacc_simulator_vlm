"""Cosmos 3 architecture / inference constants.

Cross-checked sources:
- HF config.json (nvidia/Cosmos-Reason1-7B → reasoner, Cosmos-Predict2 → DM)
- NVIDIA Cosmos repo `inference_benchmarks.md`
- Cosmos 3 tech report (research.nvidia.com)

All numbers are paper / config anchors used by analysis scripts.  Do not
edit without re-checking against the upstream.
"""

# ---- Nano (16 B = 8 B reasoner + 8 B generator) ----
NANO = {
    "name": "Cosmos3-Nano",
    "backbone": "Qwen3-VL-8B",
    "params_billion": 16,
    "weight_bytes_bf16": 16e9 * 2,           # 32 GB
    # Text tower (AR side)
    "hidden": 4096,
    "n_layers": 36,
    "n_q_heads": 32,
    "n_kv_heads": 8,
    "d_head": 128,
    "ffn": 12288,
    "max_position_embeddings": 262144,        # 256 K
    "vocab_size": 151936,
    # Vision tower
    "vit_layers": 27,
    "vit_hidden": 1152,
    "vit_patch": 16,
    "vit_heads": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "deepstack_layers": [8, 16, 24],
    "out_hidden_size": 4096,
}

# ---- Super (64 B = 32 B reasoner + 32 B generator) ----
SUPER = {
    "name": "Cosmos3-Super",
    "backbone": "Qwen3-VL-32B",
    "params_billion": 64,
    "weight_bytes_bf16": 64e9 * 2,           # 128 GB
    "hidden": 5120,
    "n_layers": 64,
    "n_q_heads": 64,
    "n_kv_heads": 8,
    "d_head": 128,
    "ffn": 25600,
    "max_position_embeddings": 262144,
    "vocab_size": 151936,
    "vit_layers": 27,
    "vit_hidden": 1152,
    "vit_patch": 16,
    "vit_heads": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "deepstack_layers": [8, 16, 24],
    "out_hidden_size": 5120,
}

ALL_MODELS = {"Cosmos3-Nano": NANO, "Cosmos3-Super": SUPER}

# ---- Official HF repos (confirmed against
# https://huggingface.co/nvidia/Cosmos3-Nano|Super, fetched 2026-06-04) ----
HF_REPOS = {
    "Cosmos3-Nano":  "nvidia/Cosmos3-Nano",
    "Cosmos3-Super": "nvidia/Cosmos3-Super",
}

# ---- Diffusers pipeline (single class for ALL tasks; differentiated by
# kwargs: num_frames=1 -> t2i, image= -> i2v, enable_sound=True -> +audio) ----
DIFFUSERS_PIPELINE_CLASS = "Cosmos3OmniPipeline"

# ---- Generation defaults (from official model card examples) ----
DEFAULT_DENOISE_STEPS = 35
DEFAULT_GUIDANCE_SCALE = 6.0
DEFAULT_FLOW_SHIFT = 10.0
DEFAULT_FRAMES = 189                  # ~ 7.875 s @ 24 fps
DEFAULT_FPS = 24.0
DEFAULT_RES = "720p"
RESOLUTIONS = {
    "256p": (448, 256),
    "480p": (832, 480),
    "720p": (1280, 720),
}

# NOTE: NVIDIA's inference_benchmarks.md uses a DIFFERENT internal
# resolution for the Diffusers engine's "256p" column -- it generates
# at 320x192, not 448x256.  When comparing measurements at engine=
# Diffusers and column starts with "256p/", apply this override.
# PyTorch / vLLM-Omni keep the canonical 256p = 448x256.
ENGINE_RESOLUTION_OVERRIDES = {
    "Diffusers": {"256p": (320, 192)},
}

# Approximate text-token budget for our standard t2v prompt (cosmos_facts
# uses the official quality-control negative prompt + a ~200-char positive
# prompt).  Used by Phase 3.2 to derive an actual_context_tokens count
# that Phase 3.5 can match against.  Real exact count requires the actual
# Qwen2 tokenizer; this is a conservative upper estimate.
TEXT_TOKENS_APPROX = 256


def actual_context_tokens(model, resolution, frames,
                            text_tokens=TEXT_TOKENS_APPROX,
                            engine="PyTorch"):
    """Compute the *actual* context-window the workload will produce.

    visual_tokens (from estimate_visual_tokens) + text approx.  Used as
    the matching key when correlating Phase 3.2 measured KV per step
    with Phase 3.5 theoretical grid.
    """
    if engine in ENGINE_RESOLUTION_OVERRIDES and \
            resolution in ENGINE_RESOLUTION_OVERRIDES[engine]:
        W, H = ENGINE_RESOLUTION_OVERRIDES[engine][resolution]
        p = model["vit_patch"]
        sm = model["spatial_merge_size"]
        tp = model["temporal_patch_size"]
        tokens_per_frame = (W // p // sm) * (H // p // sm)
        # ceil(frames / tp) -- match estimate_visual_tokens()
        temporal_groups = max(1, -(-frames // tp))
        visual = tokens_per_frame * temporal_groups
    else:
        visual = estimate_visual_tokens(model, resolution, frames)
    return visual + text_tokens

# Official quality-control negative prompts from the diffusers docs page.
NEGATIVE_PROMPT_T2V = (
    "The video captures a series of frames showing ugly scenes, static "
    "with no motion, motion blur, over-saturation, shaky footage, low "
    "resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out "
    "colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, "
    "jump cuts, visual noise, and flickering. Overall, the video is of "
    "poor quality."
)
NEGATIVE_PROMPT_I2V = (
    "The video captures a series of frames showing macroblocking "
    "artifacts, chromatic aberration, high-frequency noise, and rolling "
    "shutter distortion. It includes static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed "
    "scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, bit-depth compression artifacts, "
    "color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump "
    "cuts, visual noise, and flickering. Avoid moire patterns, edge "
    "halos, and temporal aliasing. Furthermore, the content defies "
    "common sense, generating illogical scenarios, nonsensical entities, "
    "absurd character behaviors, and conceptual paradoxes that violate "
    "basic human reasoning and everyday reality. The video looks like a "
    "surreal or glitchy hallucination. Overall, the video is of poor "
    "quality."
)

# ---- Modality token rates (used by Phase 1.8 + theoretical_pim_gain) ----
# Audio: Cosmos 3 audio tokenizer (placeholder until Phase 1.8 confirms).
# Treat as 50 tokens / sec (≈ Encodec / SpeechTokenizer @ 50 Hz).
AUDIO_TOKENS_PER_SEC = 50
# Action: 32D humanoid joint state at 30 Hz ≈ 1 token (latent compressed).
ACTION_TOKENS_PER_SEC = 30
# Image: a single 720p image after ViT patchify + spatial_merge.
# Used by t2i task to estimate token budget.


# ---- Omni-modal task matrix ----
TASKS = (
    "t2v", "t2i", "i2v", "v2v", "t2a", "multi2v", "multi2action",
)
TASK_REQUIRES = {
    # inputs / outputs in {"text", "image", "video", "audio", "action"}
    "t2v":           {"inputs": ["text"],
                       "outputs": ["video"]},
    "t2i":           {"inputs": ["text"],
                       "outputs": ["image"]},
    "i2v":           {"inputs": ["text", "image"],
                       "outputs": ["video"]},
    "v2v":           {"inputs": ["text", "video"],
                       "outputs": ["video"]},
    "t2a":           {"inputs": ["text"],
                       "outputs": ["audio"]},
    "multi2v":       {"inputs": ["text", "image", "audio"],
                       "outputs": ["video"]},
    "multi2action":  {"inputs": ["text", "image", "video"],
                       "outputs": ["action"]},
}

# ---- NVIDIA inference_benchmarks.md table structure ----
# Confirmed against
# https://github.com/NVIDIA/cosmos/blob/main/inference_benchmarks.md
# (fetched 2026-06-04).  Tables are organized as:
#   per (model, task, GPU) -> rows = inference engine (PyTorch /
#   vLLM-Omni / Diffusers), columns = "resolution/TP" pairs:
#     256p/1, 256p/4, 256p/8, 480p/1, 480p/4, 480p/8,
#     720p/1, 720p/4, 720p/8
# An empty cell means "not measured" (not "unsupported").
NVIDIA_BENCHMARK_COLUMNS = [
    "256p/1", "256p/4", "256p/8",
    "480p/1", "480p/4", "480p/8",
    "720p/1", "720p/4", "720p/8",
]
NVIDIA_BENCHMARK_GPUS = [
    "RTX_PRO_6000_Blackwell",
    "H20",
    "H100_NVL",
    "H200_NVL",
    "H100_80GB_HBM3_SXM",
    "H200_141GB_HBM3",
    "B200",
    "B300",
]
NVIDIA_BENCHMARK_ENGINES = ["PyTorch", "vLLM-Omni", "Diffusers"]
NVIDIA_BENCHMARK_TASKS = ["t2v", "i2v", "t2i"]

# Cells transcribed VERBATIM from raw markdown table at
# https://raw.githubusercontent.com/NVIDIA/cosmos/main/inference_benchmarks.md
# (fetched 2026-06-04).  Empty source cells -> key omitted.
# Schema: NVIDIA_BENCHMARK[(model, task, gpu, engine)] = {col: seconds}.
#
# (*) annotations in the source (vLLM-Omni 88.25(*), 54.01(*) etc.) are
# captured by NVIDIA_BENCHMARK_NOTES below.
NVIDIA_BENCHMARK = {
    # ---- Cosmos3-Nano t2v H100 NVL ----
    ("Cosmos3-Nano", "t2v", "H100_NVL", "PyTorch"): {
        "256p/8": 3.95, "480p/1": 84.12,
        "720p/1": 297.27, "720p/4": 94.15, "720p/8": 61.63,
    },
    ("Cosmos3-Nano", "t2v", "H100_NVL", "vLLM-Omni"): {
        "720p/1": 311.13, "720p/4": 88.25, "720p/8": 54.01,
    },
    ("Cosmos3-Nano", "t2v", "H100_NVL", "Diffusers"): {
        "256p/1": 11.00, "480p/1": 90.00, "720p/1": 324.20,
    },
    # ---- Cosmos3-Nano i2v H100 NVL ----
    ("Cosmos3-Nano", "i2v", "H100_NVL", "PyTorch"): {
        "256p/8": 3.99, "480p/1": 84.50, "480p/4": 28.69,
        "720p/1": 298.57, "720p/4": 95.76, "720p/8": 60.58,
    },
    ("Cosmos3-Nano", "i2v", "H100_NVL", "vLLM-Omni"): {
        "720p/1": 286.33, "720p/4": 92.23, "720p/8": 58.02,
    },
    ("Cosmos3-Nano", "i2v", "H100_NVL", "Diffusers"): {
        "256p/1": 11.00, "480p/1": 91.00, "720p/1": 325.20,
    },
    # ---- Cosmos3-Nano t2i H100 NVL ----
    ("Cosmos3-Nano", "t2i", "H100_NVL", "PyTorch"): {
        "256p/4": 2.45, "720p/1": 4.21, "720p/4": 2.57, "720p/8": 2.64,
    },
    ("Cosmos3-Nano", "t2i", "H100_NVL", "vLLM-Omni"): {
        "720p/1": 3.44, "720p/4": 1.83, "720p/8": 1.90,
    },
    ("Cosmos3-Nano", "t2i", "H100_NVL", "Diffusers"): {
        "256p/1": 3.00, "480p/1": 3.00, "720p/1": 4.00,
    },
    # ---- Cosmos3-Super t2v H100 NVL ----
    ("Cosmos3-Super", "t2v", "H100_NVL", "PyTorch"): {
        "480p/4": 101.27, "720p/4": 330.04, "720p/8": 186.19,
    },
    # ---- Cosmos3-Super i2v H100 NVL ----
    ("Cosmos3-Super", "i2v", "H100_NVL", "PyTorch"): {
        "256p/8": 16.96, "720p/4": 331.40, "720p/8": 186.47,
    },
    # ---- Cosmos3-Super t2i H100 NVL ----
    ("Cosmos3-Super", "t2i", "H100_NVL", "PyTorch"): {
        "256p/4": 19.73, "256p/8": 19.86, "720p/4": 20.68, "720p/8": 19.87,
    },
    # ---- Cosmos3-Nano t2v H100 80GB HBM3 (SXM) ----
    ("Cosmos3-Nano", "t2v", "H100_80GB_HBM3_SXM", "PyTorch"): {
        "256p/1": 7.61, "480p/1": 59.83, "720p/1": 207.78,
    },
    ("Cosmos3-Nano", "t2v", "H100_80GB_HBM3_SXM", "Diffusers"): {
        "256p/1": 9.00, "480p/1": 68.00, "720p/1": 240.00,
    },
    # ---- Cosmos3-Nano i2v H100 80GB HBM3 (SXM) ----
    ("Cosmos3-Nano", "i2v", "H100_80GB_HBM3_SXM", "PyTorch"): {
        "256p/1": 7.64, "720p/1": 207.87,
    },
    ("Cosmos3-Nano", "i2v", "H100_80GB_HBM3_SXM", "Diffusers"): {
        "256p/1": 9.00, "480p/1": 68.00, "720p/1": 239.80,
    },
    # ---- Cosmos3-Nano t2i H100 80GB HBM3 (SXM) ----
    ("Cosmos3-Nano", "t2i", "H100_80GB_HBM3_SXM", "PyTorch"): {
        "256p/1": 3.01, "720p/1": 3.45,
    },
    ("Cosmos3-Nano", "t2i", "H100_80GB_HBM3_SXM", "Diffusers"): {
        "256p/1": 3.00, "480p/1": 3.00, "720p/1": 4.00,
    },
    # ---- Cosmos3-Nano t2v H200 141GB HBM3 ----
    ("Cosmos3-Nano", "t2v", "H200_141GB_HBM3", "PyTorch"): {
        "256p/4": 3.34, "256p/8": 3.19, "480p/8": 13.97,
        "720p/1": 214.28, "720p/4": 67.48, "720p/8": 41.26,
    },
    ("Cosmos3-Nano", "t2v", "H200_141GB_HBM3", "Diffusers"): {
        "256p/1": 9.00, "480p/1": 67.00, "720p/1": 239.60,
    },
}

NVIDIA_BENCHMARK_NOTES = {
    # vLLM-Omni Nano t2v / i2v H100 NVL 720p/4 + 720p/8 carry an (*) in
    # the source page (likely "preliminary / scheduler under tuning").
    ("Cosmos3-Nano", "t2v", "H100_NVL", "vLLM-Omni"): {
        "720p/4": "(*) page annotation -- verify against latest release",
        "720p/8": "(*) page annotation -- verify against latest release",
    },
    ("Cosmos3-Nano", "i2v", "H100_NVL", "vLLM-Omni"): {
        "720p/4": "(*) page annotation -- verify against latest release",
        "720p/8": "(*) page annotation -- verify against latest release",
    },
}

# ---- AttAcc paper claim BW (used for theoretical lower-bound) ----
ATTACC_BW_TBS = 242.0      # 4-stack HBM-PIM: paper Fig. 14 BW peak
H100_BW_TBS = 3.35         # H100 80GB SXM5 spec
H100_NVL_BW_TBS = 3.9
A6000_BW_TBS = 0.768


def kv_bytes_per_token(model):
    """BF16 GQA: 2 (K+V) * n_kv_heads * d_head * n_layers * 2 bytes."""
    return 2 * model["n_kv_heads"] * model["d_head"] * model["n_layers"] * 2


def kv_bytes_per_request(model, context_tokens=None):
    if context_tokens is None:
        context_tokens = model["max_position_embeddings"]
    return kv_bytes_per_token(model) * context_tokens


def estimate_visual_tokens(model, resolution=DEFAULT_RES,
                            n_frames=DEFAULT_FRAMES):
    """Rough visual token count after patchify + spatial/temporal merge.

    tokens_per_frame = (W/patch/spatial_merge) * (H/patch/spatial_merge)
    total = ceil(frames / temporal_patch) * tokens_per_frame

    Use CEIL: the trailing partial group still produces a temporal
    token group (padded internally).  Floor undercounts by one group's
    worth at every odd frame count, which is exactly what 189-frame
    (default Cosmos workload) hits.
    """
    W, H = RESOLUTIONS[resolution]
    p = model["vit_patch"]
    sm = model["spatial_merge_size"]
    tp = model["temporal_patch_size"]
    tokens_per_frame = (W // p // sm) * (H // p // sm)
    # ceil(n_frames / tp) via Python idiom -(-a // b)
    temporal_groups = max(1, -(-n_frames // tp))
    return tokens_per_frame * temporal_groups


def denoise_traffic_bytes(model, context_tokens=None,
                           steps=DEFAULT_DENOISE_STEPS):
    """Worst-case lower bound: each denoise step reads entire KV once."""
    return kv_bytes_per_request(model, context_tokens) * steps


if __name__ == "__main__":
    for name, m in ALL_MODELS.items():
        kvpt = kv_bytes_per_token(m)
        kvpr = kv_bytes_per_request(m)
        traf = denoise_traffic_bytes(m)
        print(f"{name:18s}  KV/tok={kvpt/1024:6.1f} KB  "
              f"KV/req(256K)={kvpr/1e9:5.1f} GB  "
              f"35step traffic={traf/1e12:5.2f} TB")
        for res in ("256p", "480p", "720p"):
            vt = estimate_visual_tokens(m, res, DEFAULT_FRAMES)
            print(f"    {res:5s} (189f) visual_tokens ~= {vt:>7d}")
