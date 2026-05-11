"""Capacity-bound regime validation.

For each VLM, sweep batch and find theoretical max batch from
get_capacity_breakdown() vs actual simulator. Demonstrate that
LLaVA-1.5/Next are capacity-bound at moderate batch while
Qwen3-VL/InternVL3 are not (paper C3 argument).
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import make_model_config, make_xpu_config
from src.system import System
from src.type import DataType, DeviceType, GPUType, InterfaceType, PIMType
from src.config import make_pim_config

from result_aggregator import save


MODELS = [
    ("Qwen3-VL-4B",            672,  569),
    ("Qwen2.5-VL-7B",          672,  704),
    ("InternVL3-8B-hf",        448,  384),
    ("LLaVA-1.5-7B",           336,  704),
    ("LLaVA-Next-Mistral-7B",  672, 3008),
]


def make_system(tp):
    model_cfg = make_model_config("Qwen3-VL-4B", DataType.W16A16)
    xpu_cfg = make_xpu_config(GPUType.H100, num_gpu=tp)
    pim_cfg = make_pim_config(PIMType.BA, InterfaceType.NVLINK4,
                               num_attacc=tp, num_hbm=5)
    system = System(xpu_cfg["GPU"], model_cfg, max_L=2048)
    system.set_accelerator(model_cfg, DeviceType.PIM, pim_cfg)
    return system


def estimate_max_batch(model_name, lin, tp):
    """Use get_capacity_breakdown() with batch=1 -> derive max."""
    model_cfg = make_model_config(model_name, DataType.W16A16)
    xpu_cfg = make_xpu_config(GPUType.H100, num_gpu=tp)
    pim_cfg = make_pim_config(PIMType.BA, InterfaceType.NVLINK4,
                               num_attacc=tp, num_hbm=5)
    system = System(xpu_cfg["GPU"], model_cfg, max_L=2048)
    system.set_accelerator(model_cfg, DeviceType.PIM, pim_cfg)
    bd = system.get_capacity_breakdown(batch_size=1, lin=lin, lout=128)
    return bd


def main():
    print("Capacity regime validation -- per-GPU max batch")
    rows = []
    for model, img, lin in MODELS:
        for tp_label, tp in [("S1 (TP=1)", 1), ("S2 (TP=2)", 2)]:
            bd = estimate_max_batch(model, lin, tp)
            weight_mib = bd["weight_per_gpu"] / 1024 / 1024
            kv_mib = bd["kv_per_gpu"] / 1024 / 1024
            avail_mib = bd["available_kv"] / 1024 / 1024
            max_batch = bd["max_batch_at_default_L"]
            rows.append({
                "model": model, "deployment": tp_label, "lin": lin,
                "weight_per_gpu_mib": round(weight_mib, 1),
                "kv_per_gpu_mib": round(kv_mib, 2),
                "available_kv_mib": round(avail_mib, 1),
                "max_batch_estimate": max_batch,
            })
            print("  {:25s} {:12s}  weight={:>7.0f} MiB  kv/req={:>6.2f} MiB  "
                  "max_batch={:>5d}".format(
                      model, tp_label, weight_mib, kv_mib, max_batch))

    save("capacity_regime",
         {"platform": "H100 80 GB simulator capacity breakdown",
          "note": "max_batch_estimate from get_capacity_breakdown()"},
         {"rows": rows,
          "interpretation": {
              "capacity_bound": ["LLaVA-1.5-7B", "LLaVA-Next-Mistral-7B"],
              "throughput_bound": ["Qwen3-VL-4B", "Qwen2.5-VL-7B",
                                     "InternVL3-8B-hf"],
          }})
    print("Done")


if __name__ == "__main__":
    main()
