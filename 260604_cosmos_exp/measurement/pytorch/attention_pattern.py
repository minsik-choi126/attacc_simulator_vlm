"""Phase 2.1 — Sparse vs dense attention pattern check.

Wraps Cosmos 3 attention forward with a hook that records the attention
softmax distribution per layer (sampled at one denoise step) and reports:

    sparsity_fraction = fraction of attn_weights < THRESHOLD (1e-4)
    effective_keys    = mean number of keys with attn > THRESHOLD per query
    layer_pattern     = per-layer histogram

This identifies whether Cosmos 3 already uses sliding window or
sparse attention (would make Topic A weaker -- per R-CO1).

Output: results/cosmos_attention_pattern.json
"""
import argparse
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))

from hw_detect import detect_host
from result_aggregator import save


HF_REPOS = {
    "Cosmos3-Nano":  "nvidia/Cosmos3-Nano",
    "Cosmos3-Super": "nvidia/Cosmos3-Super",
}
RES = {"256p": (448, 256), "480p": (832, 480), "720p": (1280, 720)}
THRESHOLD = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--resolution", default="480p")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--denoise-steps", type=int, default=5)
    ap.add_argument("--prompt", default="a self-driving car turning right")
    ap.add_argument("--max-layers-recorded", type=int, default=4)
    args = ap.parse_args()

    host = detect_host()
    print(f"[Phase 2.1] attention_pattern on host={host}")
    try:
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )
        sys.path.insert(0, str(HERE.parents[1] / "shared"))
        from cosmos_facts import (NEGATIVE_PROMPT_T2V, DEFAULT_FLOW_SHIFT,
                                  DEFAULT_FPS)

        repo = HF_REPOS[args.model]
        pipe = Cosmos3OmniPipeline.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map="cuda",
            enable_safety_checker=False)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=DEFAULT_FLOW_SHIFT)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        captured = {}      # layer_idx -> list of sparsity stats per call

        def _rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        class SparsityProcessor:
            """Cosmos3PackedMoTAttention wrapper: recompute generation-pathway
            softmax to measure attention sparsity, then delegate to base."""
            def __init__(self, base, idx):
                self.base = base
                self.idx = idx

            def __call__(self, attn, und_seq, gen_seq, rotary_emb):
                try:
                    with torch.no_grad():
                        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
                        hd = attn.head_dim
                        Hq, Hkv = attn.num_attention_heads, attn.num_key_value_heads
                        groups = Hq // Hkv
                        q_gen = attn.add_q_proj(gen_seq).view(-1, Hq, hd)
                        k_und = attn.to_k(und_seq).view(-1, Hkv, hd)
                        k_gen = attn.add_k_proj(gen_seq).view(-1, Hkv, hd)
                        q_gen = attn.norm_added_q(q_gen)
                        k_und = attn.norm_k(k_und); k_gen = attn.norm_added_k(k_gen)
                        cg, sg = cos_gen.unsqueeze(1), sin_gen.unsqueeze(1)
                        cu, su = cos_und.unsqueeze(1), sin_und.unsqueeze(1)
                        q_gen = q_gen * cg + _rotate_half(q_gen) * sg
                        k_gen = k_gen * cg + _rotate_half(k_gen) * sg
                        k_und = k_und * cu + _rotate_half(k_und) * su
                        all_k = torch.cat([k_und, k_gen], dim=0).repeat_interleave(
                            groups, dim=1)
                        K = all_k.shape[0]; scale = hd ** -0.5
                        sp = 0.0; eff = 0.0
                        for h in range(Hq):
                            s = torch.matmul(q_gen[:, h, :],
                                             all_k[:, h, :].transpose(0, 1)).float() * scale
                            w = torch.softmax(s, dim=-1)
                            sp += (w < THRESHOLD).float().mean().item()
                            eff += (w > THRESHOLD).float().sum(-1).mean().item()
                        captured.setdefault(self.idx, []).append(
                            {"sparsity_frac": round(sp / Hq, 4),
                             "effective_keys": round(eff / Hq, 2),
                             "seqlen": int(K)})
                except Exception as ex:
                    captured.setdefault("diag", []).append(str(ex)[:200])
                return self.base(attn, und_seq, gen_seq, rotary_emb)

        attn_mods = [(n, m) for n, m in pipe.transformer.named_modules()
                     if m.__class__.__name__.endswith("Attention")
                     and hasattr(m, "add_q_proj")]
        if not attn_mods:
            raise RuntimeError("No Cosmos3PackedMoTAttention in transformer")
        # sample evenly across the stack
        step = max(1, len(attn_mods) // args.max_layers_recorded)
        sampled = attn_mods[::step][:args.max_layers_recorded]
        for i, (n, m) in enumerate(sampled):
            m.processor = SparsityProcessor(m.processor, i)

        W, H = RES[args.resolution]
        gen = torch.Generator(device="cuda").manual_seed(123)
        _ = pipe(prompt=[args.prompt] * args.batch,
                 negative_prompt=[NEGATIVE_PROMPT_T2V] * args.batch,
                 num_frames=args.frames, height=H, width=W,
                 num_inference_steps=args.denoise_steps,
                 guidance_scale=6.0, fps=DEFAULT_FPS, generator=gen)

        out = {"sampled_attn_layers": [n for n, _ in sampled],
               "per_layer_stats": {str(k): v for k, v in captured.items()},
               "threshold": THRESHOLD}
        for k, v in captured.items():
            print(f"  layer[{k}] {v[:2] if isinstance(v,list) else v}")
        save("cosmos_attention_pattern",
              {"phase": "2.1", "host": host, "platform": host, **vars(args)},
              out)
    except Exception as e:
        traceback.print_exc()
        save("cosmos_attention_pattern",
              {"phase": "2.1", "host": host, "platform": host, **vars(args)},
              {"status": "fail", "error": str(e)})


if __name__ == "__main__":
    main()
