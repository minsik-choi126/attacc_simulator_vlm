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
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo = HF_REPOS[args.model]
        mdl = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to("cuda")
        mdl.eval()
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

        captured = {}      # layer_idx -> sparsity stats

        def make_hook(idx):
            def hook(module, args_, output):
                # output may be (attn_out, attn_weights) — depends on model
                if isinstance(output, tuple) and len(output) > 1:
                    w = output[1]
                    if w is not None and w.dim() >= 3:
                        flat = w.float().reshape(-1, w.shape[-1])
                        sparsity = (flat < THRESHOLD).float().mean().item()
                        eff = (flat > THRESHOLD).float().sum(-1).mean().item()
                        captured.setdefault(idx, []).append(
                            {"sparsity_frac": round(sparsity, 4),
                             "effective_keys": round(eff, 2),
                             "seqlen": int(flat.shape[-1])})
            return hook

        # Try to find attention modules
        attn_layers = []
        for name, m in mdl.named_modules():
            if m.__class__.__name__.endswith("Attention"):
                attn_layers.append((name, m))
        sampled = attn_layers[:args.max_layers_recorded]
        handles = [m.register_forward_hook(make_hook(i))
                   for i, (n, m) in enumerate(sampled)]

        ids = tok([args.prompt] * args.batch,
                   return_tensors="pt").input_ids.to("cuda")
        with torch.inference_mode():
            _ = mdl(ids, output_attentions=True)

        for h in handles:
            h.remove()

        out = {"sampled_attn_layers": [n for n, _ in sampled],
               "per_layer_stats": captured,
               "threshold": THRESHOLD}
        for k, v in captured.items():
            print(f"  layer[{k}] {v}")
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
