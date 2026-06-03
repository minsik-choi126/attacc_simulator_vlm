"""Phase 2.5 — Per-modality KV cost breakdown (analysis only).

Combines Phase 1.8 (measured tokens / modality) with cosmos_facts
(KV bytes / token) to produce, for each (model, scenario):

    tokens_text, tokens_image, tokens_video, tokens_audio
    kv_bytes_text, kv_bytes_image, ...
    share_of_total_kv_per_modality

Scenarios are realistic input bundles (e.g. "drive_assist": 1 prompt +
1 image + 8s audio + 24f video).

Output: results/cosmos_modality_kv_breakdown.json
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from hw_detect import detect_host
from result_aggregator import load, save
from cosmos_facts import (
    NANO, SUPER, kv_bytes_per_token, AUDIO_TOKENS_PER_SEC,
    ACTION_TOKENS_PER_SEC, estimate_visual_tokens,
)


SCENARIOS = {
    "drive_assist":   {"text": 64, "image": ("720p", 1),
                        "video": ("720p", 24), "audio_s": 8},
    "robot_planner":  {"text": 128, "image": ("480p", 4),
                        "video": ("480p", 48), "audio_s": 4},
    "video_gen_only": {"text": 32, "image": None,
                        "video": ("720p", 189), "audio_s": 0},
    "audio_caption":  {"text": 32, "image": None,
                        "video": None, "audio_s": 32},
    "world_model":    {"text": 256, "image": ("720p", 4),
                        "video": ("720p", 96), "audio_s": 16},
}


def _measured_tokens(payload, model_name, modality, key):
    if payload is None:
        return None
    rec = payload.get("results", {}).get("models", {}).get(model_name, {})
    return rec.get(modality, {}).get(key)


def main():
    host = detect_host()
    print(f"[Phase 2.5] modality_kv_breakdown on host={host}")
    measured = load("cosmos_tokens_per_modality")

    rows = []
    for model in (NANO, SUPER):
        kvpt = kv_bytes_per_token(model)
        for scen, spec in SCENARIOS.items():
            tt = spec["text"]
            # image
            if spec["image"]:
                res, n = spec["image"]
                im = (_measured_tokens(measured, model["name"], "image", res)
                      or estimate_visual_tokens(model, res, 1)) * n
            else:
                im = 0
            # video
            if spec["video"]:
                res, n = spec["video"]
                key = f"{res}_{n}f"
                vid = (_measured_tokens(measured, model["name"],
                                          "video", key)
                       or estimate_visual_tokens(model, res, n))
            else:
                vid = 0
            au = AUDIO_TOKENS_PER_SEC * spec["audio_s"]

            tokens = {"text": tt, "image": im, "video": vid, "audio": au}
            kv = {k: v * kvpt for k, v in tokens.items()}
            total_kv = sum(kv.values())
            share = ({k: round(v / total_kv, 4) for k, v in kv.items()}
                     if total_kv > 0 else {})
            row = {"model": model["name"], "scenario": scen,
                   "tokens": tokens, "kv_bytes": kv,
                   "total_kv_bytes": total_kv,
                   "share_of_total_kv": share}
            rows.append(row)
            print(f"  {model['name']:18s} {scen:18s} total KV={total_kv/1e9:5.2f} GB "
                  f"share={share}")

    save("cosmos_modality_kv_breakdown",
          {"phase": "2.5", "host": host, "platform": host,
           "scenarios": SCENARIOS,
           "audio_tokens_per_sec": AUDIO_TOKENS_PER_SEC,
           "action_tokens_per_sec": ACTION_TOKENS_PER_SEC},
          {"rows": rows})


if __name__ == "__main__":
    main()
