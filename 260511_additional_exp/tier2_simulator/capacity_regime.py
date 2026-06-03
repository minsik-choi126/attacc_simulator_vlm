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
from hw_detect import detect_host, gputype_enum

HOST = detect_host()
HOST_GPUTYPE = gputype_enum(HOST)
HOST_INTERFACE = {
    "A6000": InterfaceType.NVLINK_BRIDGE,
    "H100":  InterfaceType.NVLINK4,
    "A100":  InterfaceType.NVLINK3,
}[HOST]


MODELS = [
    ("Qwen3-VL-4B",            672,  569),
    ("Qwen2.5-VL-7B",          672,  704),
    ("InternVL3-8B-hf",        448,  384),
    ("LLaVA-1.5-7B",           336,  704),
    ("LLaVA-Next-Mistral-7B",  672, 3008),
]
DEPLOYMENTS = [("A1 (TP=1)", 1)]
# Future hook, intentionally not used in the first paper pass.
FUTURE_DEPLOYMENTS = [("A2 (TP=2)", 2)]


def make_system(tp):
    model_cfg = make_model_config("Qwen3-VL-4B", DataType.W16A16)
    xpu_cfg = make_xpu_config(HOST_GPUTYPE, num_gpu=tp)
    pim_cfg = make_pim_config(PIMType.BA, HOST_INTERFACE,
                               num_attacc=tp, num_hbm=5)
    system = System(xpu_cfg["GPU"], model_cfg, max_L=2048)
    system.set_accelerator(model_cfg, DeviceType.PIM, pim_cfg)
    return system


def estimate_max_batch(model_name, lin, tp):
    """Use get_capacity_breakdown() with batch=1 -> derive max."""
    model_cfg = make_model_config(model_name, DataType.W16A16)
    xpu_cfg = make_xpu_config(HOST_GPUTYPE, num_gpu=tp)
    pim_cfg = make_pim_config(PIMType.BA, HOST_INTERFACE,
                               num_attacc=tp, num_hbm=5)
    system = System(xpu_cfg["GPU"], model_cfg, max_L=2048)
    system.set_accelerator(model_cfg, DeviceType.PIM, pim_cfg)
    bd = system.get_capacity_breakdown(batch_size=1, lin=lin, lout=128)
    return bd


def main():
    print("Capacity regime validation -- per-system max batch")
    print("(Post-Fix-C: dgx-attacc moves KV to AttAcc HBM -- both sides "
          "reported)")
    rows = []
    for model, img, lin in MODELS:
        for tp_label, tp in DEPLOYMENTS:
            bd = estimate_max_batch(model, lin, tp)
            weight_mib = bd["weight_per_gpu"] / 1024 / 1024
            kv_gpu_mib = bd["kv_per_gpu"] / 1024 / 1024
            avail_gpu_mib = bd["available_kv"] / 1024 / 1024
            # Fix C exposes the AttAcc-side breakdown when KV moves there.
            kv_attacc_mib = bd.get("kv_per_attacc", 0) / 1024 / 1024
            avail_attacc_mib = (
                bd.get("available_kv_attacc", 0) / 1024 / 1024)
            attacc_cap_mib = (
                bd.get("attacc_capacity_total", 0) / 1024 / 1024)
            max_batch = bd["max_batch_at_default_L"]
            max_batch_attacc = bd.get("max_batch_at_default_L_attacc")
            kv_side = "AttAcc" if attacc_cap_mib > 0 else "GPU"

            rows.append({
                "model": model, "deployment": tp_label, "lin": lin,
                "kv_resident_side": kv_side,
                "weight_per_gpu_mib": round(weight_mib, 1),
                "kv_per_gpu_mib": round(kv_gpu_mib, 2),
                "available_kv_gpu_mib": round(avail_gpu_mib, 1),
                "kv_per_attacc_mib": round(kv_attacc_mib, 2),
                "available_kv_attacc_mib": round(avail_attacc_mib, 1),
                "attacc_capacity_total_mib": round(attacc_cap_mib, 1),
                "max_batch_estimate": max_batch,
                "max_batch_attacc_side": max_batch_attacc,
            })
            if kv_side == "AttAcc":
                print(("  {:25s} {:12s}  weight={:>7.0f} MiB  "
                       "kv_side=AttAcc  kv/req(AttAcc)={:>6.2f} MiB  "
                       "max_batch={:>5d}").format(
                          model, tp_label, weight_mib,
                          kv_attacc_mib, max_batch))
            else:
                print(("  {:25s} {:12s}  weight={:>7.0f} MiB  "
                       "kv_side=GPU     kv/req(GPU)={:>6.2f} MiB  "
                       "max_batch={:>5d}").format(
                          model, tp_label, weight_mib,
                          kv_gpu_mib, max_batch))

    # Save twice so users get a stable filename (capacity_regime.json,
    # whatever ran last) and a per-host filename for cross-HW reference.
    meta = {"platform": f"{HOST} GPU + AttAcc HBM (post-Fix-C)",
            "host_detected": HOST,
            "deployment_scope": "A1 TP=1 only",
            "future_hooks": [label for label, _ in FUTURE_DEPLOYMENTS],
            "note": ("max_batch_estimate = system limiting batch; "
                     "kv_resident_side tells which device holds KV "
                     "(GPU under dgx, AttAcc under dgx-attacc).  After "
                     "Fix C the AttAcc-side fields are populated for "
                     "dgx-attacc, replacing the prior buggy GPU-only "
                     "accounting (LLaVA 88/91 -> 197/209 corrected)")}
    payload = {"rows": rows,
               "interpretation": {
                   "capacity_bound": ["LLaVA-1.5-7B", "LLaVA-Next-Mistral-7B"],
                   "throughput_bound": ["Qwen3-VL-4B", "Qwen2.5-VL-7B",
                                          "InternVL3-8B-hf"],
               }}
    save("capacity_regime", meta, payload)
    save(f"capacity_regime_{HOST.lower()}", meta, payload)
    print("Done")


if __name__ == "__main__":
    main()
