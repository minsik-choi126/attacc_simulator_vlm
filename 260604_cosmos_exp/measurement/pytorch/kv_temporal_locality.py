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
        captured_per_call = []   # list[ tensor (1, K) of mean softmax mass ]
        capture_diag = []

        def _rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        class CapturingProcessor:
            """Cosmos3PackedMoTAttention dual-pathway processor wrapper.

            Cosmos3 uses a packed Mixture-of-Transformers attention whose
            processor signature is (attn, und_seq, gen_seq, rotary_emb) and
            whose real kernel (dispatch_attention_fn) does not return softmax
            weights.  We mirror the *generation* pathway's Q/K projection +
            RMSNorm + rotary to recompute scores -> softmax for the locality
            histogram (gen queries attend to all = und(old context) + gen
            keys, so bucket 0 = oldest tokens), then delegate to the base
            processor for the actual (und_out, gen_out) so the pipeline output
            is unchanged.
            """

            def __init__(self, base):
                self.base = base

            def __call__(self, attn, und_seq, gen_seq, rotary_emb):
                try:
                    with torch.no_grad():
                        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
                        hd = attn.head_dim
                        Hq = attn.num_attention_heads
                        Hkv = attn.num_key_value_heads
                        groups = Hq // Hkv
                        # generation queries, und+gen keys (the "all keys"
                        # the full pathway cross-attends to)
                        q_gen = attn.add_q_proj(gen_seq).view(-1, Hq, hd)
                        k_und = attn.to_k(und_seq).view(-1, Hkv, hd)
                        k_gen = attn.add_k_proj(gen_seq).view(-1, Hkv, hd)
                        q_gen = attn.norm_added_q(q_gen)
                        k_und = attn.norm_k(k_und)
                        k_gen = attn.norm_added_k(k_gen)
                        cg, sg = cos_gen.unsqueeze(1), sin_gen.unsqueeze(1)
                        cu, su = cos_und.unsqueeze(1), sin_und.unsqueeze(1)
                        q_gen = q_gen * cg + _rotate_half(q_gen) * sg
                        k_gen = k_gen * cg + _rotate_half(k_gen) * sg
                        k_und = k_und * cu + _rotate_half(k_und) * su
                        all_k = torch.cat([k_und, k_gen], dim=0)  # (K,Hkv,hd)
                        # GQA expand kv heads to query heads
                        all_k = all_k.repeat_interleave(groups, dim=1)  # (K,Hq,hd)
                        K = all_k.shape[0]
                        scale = hd ** -0.5
                        # accumulate mean softmax mass per key position,
                        # head-by-head to bound memory on long sequences
                        mass = torch.zeros(K, dtype=torch.float32,
                                           device=q_gen.device)
                        for h in range(Hq):
                            s = torch.matmul(q_gen[:, h, :],
                                             all_k[:, h, :].transpose(0, 1))
                            s = s.float() * scale          # (Qg, K)
                            w = torch.softmax(s, dim=-1)
                            mass += w.sum(dim=0)
                        mass /= (Hq * q_gen.shape[0])       # mean over heads+queries
                        captured_per_call.append(mass.unsqueeze(0).cpu())  # (1,K)
                except Exception as ex:  # never break the pipeline
                    capture_diag.append(str(ex)[:200])
                return self.base(attn, und_seq, gen_seq, rotary_emb)

        # Patch a mid-stack attention layer (one is enough for the locality
        # story; capturing all is too much memory).  Cosmos3's class is
        # Cosmos3PackedMoTAttention -> endswith("Attention").
        attn_mods = [m for _, m in pipe.transformer.named_modules()
                     if m.__class__.__name__.endswith("Attention")
                     and hasattr(m, "to_q") and hasattr(m, "add_q_proj")]
        if not attn_mods:
            raise RuntimeError(
                "No Cosmos3PackedMoTAttention module found in pipe.transformer")
        target_attn = attn_mods[len(attn_mods) // 2]
        if not hasattr(target_attn, "processor"):
            raise RuntimeError("Attention module has no processor slot")
        target_attn.processor = CapturingProcessor(target_attn.processor)

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
               "n_calls_captured": len(captured_per_call),
               "capture_diag": capture_diag[:5],
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
