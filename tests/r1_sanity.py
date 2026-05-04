import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import make_model_config, make_pim_config, make_xpu_config
from src.system import System
from src.type import DataType, DeviceType, GPUType, InterfaceType, PIMType


def make_system(tp):
    model_cfg = make_model_config('Qwen3-VL-4B', DataType.W16A16)
    xpu_cfg = make_xpu_config(GPUType.H100, num_gpu=tp)
    pim_cfg = make_pim_config(PIMType.BA,
                              InterfaceType.NVLINK4,
                              num_attacc=tp,
                              num_hbm=5)
    system = System(xpu_cfg['GPU'], model_cfg, max_L=2048)
    system.set_accelerator(model_cfg, DeviceType.PIM, pim_cfg)
    return system


def assert_close(name, actual, expected, rel_tol=0.01):
    if abs(actual - expected) > expected * rel_tol:
        raise AssertionError("{}: got {:.4f}, expected {:.4f}".format(
            name, actual, expected))


def main():
    expected = {1: 80.02, 2: 40.0}
    for tp in [1, 2]:
        system = make_system(tp)
        breakdown = system.get_capacity_breakdown(batch_size=1,
                                                  lin=569,
                                                  lout=1)
        kv_mib = breakdown['kv_per_gpu'] / 1024 / 1024
        assert_close("TP{} kv_per_gpu_mib".format(tp), kv_mib, expected[tp])
        print("TP{} kv_per_gpu_mib={:.2f}".format(tp, kv_mib))
    print("r1-sanity-ok")


if __name__ == '__main__':
    main()
