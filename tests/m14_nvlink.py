import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import make_model_config, make_xpu_config
from src.system import System
from src.type import DataType, GPUType


def comm_time(num_gpu, bandwidth, lin=16, stage='gen'):
    model_cfg = make_model_config('Qwen3-VL-4B', DataType.W16A16)
    xpu_cfg = make_xpu_config(GPUType.H100, num_gpu=num_gpu)
    xpu_cfg['GPU']['INTERFACE_BW'] = bandwidth
    system = System(xpu_cfg['GPU'], model_cfg)
    perfs = []
    system.simulate(1, lin, 2, perfs=perfs)
    if stage == 'sum':
        return perfs[0][2][3] - perfs[0][2][4]
    return perfs[0][2][16]


def main():
    s1 = comm_time(1, 900 * 1000 * 1000 * 1000)
    s2_nv4 = comm_time(2, 900 * 1000 * 1000 * 1000)
    s2_nv3 = comm_time(2, 600 * 1000 * 1000 * 1000)
    assert s1 == 0
    assert s2_nv4 > 0
    assert s2_nv4 < s2_nv3

    s2_nv4_large = comm_time(2,
                              900 * 1000 * 1000 * 1000,
                              lin=569,
                              stage='sum')
    s2_nv3_large = comm_time(2,
                              600 * 1000 * 1000 * 1000,
                              lin=569,
                              stage='sum')
    assert s2_nv4_large < 0.85 * s2_nv3_large

    print("s1_g2g={:.6f}".format(s1))
    print("s2_nvlink4_g2g={:.6f}".format(s2_nv4))
    print("s2_nvlink3_g2g={:.6f}".format(s2_nv3))
    print("s2_nvlink4_large_g2g={:.6f}".format(s2_nv4_large))
    print("s2_nvlink3_large_g2g={:.6f}".format(s2_nv3_large))
    print("m14-nvlink-ok")


if __name__ == '__main__':
    main()
