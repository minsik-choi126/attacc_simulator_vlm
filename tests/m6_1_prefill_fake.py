import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import make_model_config, make_xpu_config
from src.system import System
from src.type import DataType, DeviceType, GPUType, LayerType, PIMType


class FakePIM:
    name = DeviceType.PIM
    num_hbm = 5
    pim_type = PIMType.BA
    aggregate_memory_capacity = 0
    peak_memory_bandwidth = 1

    def get_time_and_energy(self, layer):
        if layer.type == LayerType.MATMUL and layer.name == 'score':
            return layer.n * 1e-9, [layer.n, 0, 0, 0, 0, 0]
        return 0, [0, 0, 0, 0, 0, 0]


def main():
    model_cfg = make_model_config('GPT-13B', DataType.W16A16)
    xpu_cfg = make_xpu_config(GPUType.H100, num_gpu=1)
    system = System(xpu_cfg['GPU'], model_cfg)
    system.hetero_name = DeviceType.PIM
    system.devices['Acc'] = FakePIM()
    system.set_routing('conservative')
    system.set_prefill_config(chunk_size=4, sample_count=8)
    perfs = []
    system.simulate(1, 16, 2, perfs=perfs)
    score = next(layer for layer in system.model.sum_decoder
                 if layer.name == 'score')
    assert getattr(score, 'prefill_chunked', False)
    assert score.prefill_chunk_size == 4
    assert score.prefill_sample_count == 4
    assert hasattr(score, 'eff_lat')
    print("prefill_chunk_size={}".format(score.prefill_chunk_size))
    print("prefill_eff_lat={:.2f}".format(score.eff_lat))
    print("m6_1-prefill-fake-ok")


if __name__ == '__main__':
    main()
