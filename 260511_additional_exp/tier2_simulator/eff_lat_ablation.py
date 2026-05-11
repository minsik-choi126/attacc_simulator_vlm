"""eff_lat (paper sec.6.1 pipelining caveat) on/off ablation.

eff_lat 자체는 simulator 내부적으로 항상 적용되며, runtime toggle이 없음.
이 script는 이론적 eff_lat 값을 모델별로 계산 + simulator latency와
eff_lat 미적용 시 expected latency 비교 (gain/loss quantification).

paper sec.0.4 expected eff_lat:
  Qwen3-VL-4B  (n_kv=8):  S1 0.80 / S2 0.57
  Qwen2.5-VL-7B (n_kv=4): S1 0.57 / S2 0.29
  InternVL3-hf  (n_kv=4): S1 0.57 / S2 0.29
  LLaVA-1.5-7B (n_kv=32): S1 0.91 / S2 0.80
  LLaVA-Next   (n_kv=8):  S1 0.80 / S2 0.57
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import sim_runner as sr
from result_aggregator import save
from src.system import System


MODELS = [
    {"model": "Qwen3-VL-4B",           "n_kv": 8,  "lin": 569,  "img": 672},
    {"model": "Qwen2.5-VL-7B",         "n_kv": 4,  "lin": 704,  "img": 672},
    {"model": "InternVL3-8B-hf",       "n_kv": 4,  "lin": 384,  "img": 448},
    {"model": "LLaVA-1.5-7B",          "n_kv": 32, "lin": 704,  "img": 336},
    {"model": "LLaVA-Next-Mistral-7B", "n_kv": 8,  "lin": 3008, "img": 672},
]


def main():
    print("eff_lat caveat ablation -- paper sec.6.1")
    rows = []
    for cfg in MODELS:
        eff_s1 = System.get_pipelining_efficiency_latency(
            cfg["n_kv"], num_hbm=5, batch_size=1)
        eff_s2 = System.get_pipelining_efficiency_latency(
            cfg["n_kv"] // 2, num_hbm=5, batch_size=1)
        # Run S1 with full eff_lat applied
        m = sr.run(
            model=cfg["model"], system="dgx-attacc", gpu="A6000",
            ngpu=1, tp=1, num_attacc=1, num_hbm=5, interface="NVLINK_BRIDGE",
            pim="bank", lin=cfg["lin"], lout=128, batch=1,
            image_size=cfg["img"],
            prefill_chunk=512, prefill_samples=8, max_L=4096,
            powerlimit=True, ffopt=True, pipeopt=True, word=2,
            routing="default",
        )
        s = m.get("s_time") if m else None
        g = m.get("g_time") if m else None
        # Without eff_lat penalty: attn portion / eff_lat
        # (rough estimate: full attn portion would be faster by 1/eff_s1)
        rows.append({
            "model": cfg["model"], "n_kv": cfg["n_kv"],
            "eff_lat_S1": round(eff_s1, 3),
            "eff_lat_S2": round(eff_s2, 3),
            "s_time_ms_S1_with_eff": s,
            "g_time_ms_S1_with_eff": g,
            "note": "eff_lat ablated externally -- paper analysis only",
        })
        print("  {:25s}  n_kv={:>2d}  eff_lat S1={:.2f} / S2={:.2f}  | "
              "s={:.1f}ms g={:.2f}ms".format(
                  cfg["model"], cfg["n_kv"], eff_s1, eff_s2, s or -1, g or -1))

    save("eff_lat_ablation",
         {"platform": "A6000 x 1 A1 dgx-attacc",
          "method": "paper sec.6.1 latency-mode caveat quantification"},
         {"models": rows})
    print("Done")


if __name__ == "__main__":
    main()
