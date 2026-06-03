"""Phase 2.2 — Temporal KV locality histogram.

Monkey-patches an Attention processor inside Cosmos3OmniPipeline so that
on every attention call we capture the per-query attention weights, then
bucket attended positions by their *token age* (= step at which the
position was first touched).  Output is a histogram per denoise step:
"of all queries at step k, what fraction of attention mass falls on
tokens introduced at step >= k - age".

This is what Topic A (tiered KV) needs as foundational evidence: if
old-frame KV is rarely attended (cold KV), tiered placement makes
sense.  If recent + old are attended equally, Topic A is weakened.

Output: results/cosmos_kv_temporal_locality.json
"""
import argparse
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))

from hw_detect import detect_host
from result_aggregator import save
from cosmos_facts import (
    HF_REPOS, NEGATIVE_PROMPT_T2V, DEFAULT_FLOW_SHIFT, DEFAULT_FPS,
    RESOLUTIONS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cosmos3-Nano",
                    choices=list(HF_REPOS))
    ap.add_argument("--resolution", default="480p",
                    choices=list(RESOLUTIONS))
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--bucket-size", type=int, default=256,
                    help="token-position bucket size")
    ap.add_argument("--prompt",
                    default="a self-driving car turning right at sunset")
    args = ap.parse_args()

    host = detect_host()
    print(f"[Phase 2.2] kv_temporal_locality on host={host}")
    try:
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )

        repo = HF_REPOS[args.model]
        pipe = Cosmos3OmniPipeline.from_pretrained(
            repo, torch_dtype=torch.bfloat16,
            device_map="cuda",
            enable_safety_checker=False)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=DEFAULT_FLOW_SHIFT)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        # Monkey-patch the joint Cosmos3 transformer attention to capture
        # softmax weights.  The Cosmos3OmniTransformer's attention layer
        # uses processor pattern; we swap one layer's processor with a
        # capturing version.  This avoids the diffusers
        # output_attentions= flag (which the Cosmos pipeline does not
        # forward to the transformer).
        captured_per_call = []   # list[ tensor (..., seqlen) of softmax ]

        class CapturingProcessor:
            def __init__(self, base):
                self.base = base

            def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, **kw):
                # Compute Q, K, V via the base attn module's projections so
                # we get the exact tensors the layer would have used.
                B = hidden_states.shape[0]
                q = attn.to_q(hidden_states)
                k_src = (encoder_hidden_states if encoder_hidden_states
                         is not None else hidden_states)
                k = attn.to_k(k_src)
                v = attn.to_v(k_src)
                head_dim = q.shape[-1] // attn.heads
                q = q.view(B, -1, attn.heads, head_dim).transpose(1, 2)
                k = k.view(B, -1, attn.heads, head_dim).transpose(1, 2)
                v = v.view(B, -1, attn.heads, head_dim).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
                if attention_mask is not None:
                    scores = scores + attention_mask
                w = torch.softmax(scores.float(), dim=-1)
                # Sample: average across batch + head to keep memory low.
                captured_per_call.append(
                    w.mean(dim=(0, 1)).detach().cpu())
                out = torch.matmul(w.to(v.dtype), v)
                out = out.transpose(1, 2).reshape(B, -1, attn.heads * head_dim)
                return attn.to_out[0](out) if isinstance(attn.to_out,
                                                          torch.nn.ModuleList) \
                    else attn.to_out(out)

        # Find a mid-layer attention module to capture (one is enough
        # for the locality story; capturing all is too much memory).
        target_attn = None
        for name, m in pipe.transformer.named_modules():
            if m.__class__.__name__.endswith("Attention"):
                target_attn = m
                break
        if target_attn is None:
            raise RuntimeError("No Attention module in pipe.transformer")
        if hasattr(target_attn, "processor"):
            target_attn.processor = CapturingProcessor(target_attn.processor)
        else:
            raise RuntimeError("Attention module has no processor slot")

        W, H = RESOLUTIONS[args.resolution]
        gen = torch.Generator(device="cuda").manual_seed(123)
        _ = pipe(
            prompt=[args.prompt],
            negative_prompt=[NEGATIVE_PROMPT_T2V],
            num_frames=args.frames, height=H, width=W,
            num_inference_steps=args.denoise_steps,
            guidance_scale=6.0, fps=DEFAULT_FPS, generator=gen,
        )

        # captured_per_call[i] has shape (Q_len, K_len) (averaged over
        # batch + head).  Bucket K dim by position groups.
        out_rows = []
        for step_idx, w in enumerate(captured_per_call):
            if w.dim() < 2:
                continue
            klen = w.shape[-1]
            bs = max(1, args.bucket_size)
            n_buckets = (klen + bs - 1) // bs
            mass_per_bucket = []
            for b in range(n_buckets):
                lo, hi = b * bs, min((b + 1) * bs, klen)
                mass_per_bucket.append(float(w[..., lo:hi].sum()))
            total = sum(mass_per_bucket) or 1.0
            frac = [m / total for m in mass_per_bucket]
            out_rows.append({"call_idx": step_idx,
                              "key_seqlen": klen,
                              "bucket_size": bs,
                              "mass_fraction_per_bucket": frac})

        save("cosmos_kv_temporal_locality",
              {"phase": "2.2", "host": host, "platform": host, **vars(args)},
              {"per_call_buckets": out_rows,
               "note": "mass_fraction_per_bucket[i] = fraction of softmax "
                        "mass landing on token positions [i*bucket, "
                        "(i+1)*bucket).  Old / new mapping requires the "
                        "Cosmos3 step -> token-position layout; if KV is "
                        "appended in order, bucket 0 is the oldest "
                        "(first-introduced) tokens."})
    except Exception as e:
        traceback.print_exc()
        save("cosmos_kv_temporal_locality",
              {"phase": "2.2", "host": host, "platform": host, **vars(args)},
              {"status": "fail", "error": str(e)[:500]})


if __name__ == "__main__":
    main()
