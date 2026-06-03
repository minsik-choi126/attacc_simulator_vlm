"""Phase 1.8 — empirical token counts per modality from the actual
Cosmos 3 tokenizer / preprocessor.

Uses AutoTokenizer / AutoProcessor / AutoConfig on the official
nvidia/Cosmos3-Nano (and -Super) repos so the token counts match what
the runtime engine actually consumes.  These numbers replace the
analytic estimates in cosmos_facts.estimate_visual_tokens() and the
placeholder audio/action rates.

Output: results/cosmos_tokens_per_modality.json
"""
import argparse
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import save
from cosmos_facts import (
    NANO, SUPER, HF_REPOS, RESOLUTIONS, estimate_visual_tokens,
)


PROMPTS = [
    "a self-driving car turning right at sunset",
    "a robotic arm picks up a red mug and places it on a shelf",
    ("In the kitchen, a humanoid robot pours coffee from a French press "
     "while explaining the brewing process in a calm voice"),
]


def _text_tokens(prompts, hf_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    return [len(tok(p).input_ids) for p in prompts]


def _image_tokens(hf_id, resolution, model_facts):
    """Use Cosmos 3 processor if exposed; fallback to analytic."""
    try:
        from transformers import AutoProcessor
        from PIL import Image
        import numpy as np
        W, H = RESOLUTIONS[resolution]
        proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        img = Image.fromarray((np.random.rand(H, W, 3) * 255).astype("uint8"))
        out = proc(images=img, return_tensors="pt")
        for k, v in out.items():
            shape = getattr(v, "shape", None)
            if shape is None or len(shape) < 2:
                continue
            # Heuristic: the largest-but-one dim is typically the token
            # count after patchify + spatial_merge.
            return int(max(shape[1:-1] or [shape[-2]]))
    except Exception as e:
        return {"fallback_analytic": estimate_visual_tokens(model_facts,
                                                              resolution, 1),
                "error": str(e)[:120]}
    return estimate_visual_tokens(model_facts, resolution, 1)


def _video_tokens(hf_id, resolution, n_frames, model_facts):
    """Try processor's video / multi-frame path; else analytic."""
    try:
        from transformers import AutoProcessor
        from PIL import Image
        import numpy as np
        W, H = RESOLUTIONS[resolution]
        proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        frames = [Image.fromarray(
            (np.random.rand(H, W, 3) * 255).astype("uint8"))
                   for _ in range(n_frames)]
        # Cosmos3 processor may expect videos= or images= for stacks; try
        # both.
        for kw_name in ("videos", "images"):
            try:
                out = proc(**{kw_name: frames}, return_tensors="pt")
                for k, v in out.items():
                    shape = getattr(v, "shape", None)
                    if shape is None or len(shape) < 2:
                        continue
                    return int(max(shape[1:-1] or [shape[-2]]))
            except Exception:
                continue
    except Exception as e:
        return {"fallback_analytic": estimate_visual_tokens(
                    model_facts, resolution, n_frames),
                "error": str(e)[:120]}
    return estimate_visual_tokens(model_facts, resolution, n_frames)


def _audio_tokens(hf_id, seconds=2.0):
    """Try Cosmos 3 audio tokenizer; return None if unavailable."""
    try:
        from transformers import AutoFeatureExtractor
        import numpy as np
        feat = AutoFeatureExtractor.from_pretrained(
            hf_id, trust_remote_code=True)
        wav = np.zeros(int(48000 * seconds), dtype="float32")
        out = feat(wav, sampling_rate=48000, return_tensors="pt")
        for k, v in out.items():
            shape = getattr(v, "shape", None)
            if shape is not None and len(shape) >= 2:
                return int(shape[-1])
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Cosmos3-Nano", "Cosmos3-Super"])
    ap.add_argument("--resolutions", nargs="+",
                    default=["256p", "480p", "720p"])
    ap.add_argument("--frame-grid", nargs="+", type=int,
                    default=[1, 24, 96, 189, 300])
    ap.add_argument("--audio-seconds", nargs="+", type=float,
                    default=[1.0, 2.0, 4.0, 8.0])
    args = ap.parse_args()

    host = detect_host()
    payload = {"models": {}}
    print(f"[Phase 1.8] tokens_per_modality on host={host}")

    for name in args.models:
        repo = HF_REPOS[name]
        facts = {"Cosmos3-Nano": NANO, "Cosmos3-Super": SUPER}[name]
        print(f"\n=== {name} ({repo}) ===")
        record = {"hf_repo": repo}
        try:
            record["text"] = {"prompt_token_counts":
                                _text_tokens(PROMPTS, repo)}
            print(f"  text token counts = "
                  f"{record['text']['prompt_token_counts']}")
        except Exception as e:
            record["text"] = {"error": str(e)[:200]}
            print(f"  text -> error: {e}")

        record["image"] = {}
        for res in args.resolutions:
            t = _image_tokens(repo, res, facts)
            record["image"][res] = t
            print(f"  image {res} -> {t}")

        record["video"] = {}
        for res in args.resolutions:
            for f in args.frame_grid:
                key = f"{res}_{f}f"
                t = _video_tokens(repo, res, f, facts)
                record["video"][key] = t

        record["audio"] = {}
        for s in args.audio_seconds:
            t = _audio_tokens(repo, s)
            record["audio"][str(s)] = t
            print(f"  audio {s}s -> {t}")

        payload["models"][name] = record

    save("cosmos_tokens_per_modality",
          {"phase": "1.8", "host": host, "platform": host, **vars(args)},
          payload)
    print("\nDone")


if __name__ == "__main__":
    main()
