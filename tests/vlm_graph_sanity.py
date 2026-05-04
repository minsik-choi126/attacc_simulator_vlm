import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import make_model_config
from src.config import make_xpu_config
from src.config import SCALING_FACTOR
from src.devices import xPU
from src.model import Transformer
from src.type import DataType, DeviceType, GPUType


def vision_time_ms(model_name, image_size):
    model = Transformer(make_model_config(model_name, DataType.W16A16),
                        tensor_parallel=1)
    model.build(1, 569, 2, False, image_size=image_size)
    gpu = xPU(DeviceType.GPU,
              make_xpu_config(GPUType.H100, num_gpu=1)['GPU'],
              SCALING_FACTOR)
    return sum(gpu.get_time_and_energy(layer)[0]
               for layer in model.vision_decoder) * 1000


def main():
    qwen = Transformer(make_model_config('Qwen3-VL-4B', DataType.W16A16),
                       tensor_parallel=1)
    qwen.build(1, 569, 2, False)
    assert len(qwen.vision_decoder) > 0
    assert len(qwen.sum_decoder_groups) == qwen.ndec
    for idx in [5, 11, 17]:
        group = qwen.sum_decoder_groups['l{}'.format(idx)][0]
        assert any(layer.name == 'deepstack_add' for layer in group)

    llava_next = Transformer(
        make_model_config('LLaVA-Next-Mistral-7B', DataType.W16A16),
        tensor_parallel=1)
    tokens = llava_next.compute_visual_tokens((672, 672))
    assert abs(tokens - 2880) <= 288
    llava_next.build(1, tokens, 2, False, image_size=(672, 672))
    assert len(llava_next.vision_decoder) > 0

    expected_ms = {
        'Qwen3-VL-4B': (672, 2.8),
        'Qwen2.5-VL-7B': (672, 7.8),
        'InternVL3-8B-hf': (448, 1.8),
        'LLaVA-1.5-7B': (336, 1.1),
        'LLaVA-Next-Mistral-7B': (672, 4.3),
    }
    for model_name, (image_size, expected) in expected_ms.items():
        actual = vision_time_ms(model_name, image_size)
        assert expected * 0.5 <= actual <= expected * 1.5, \
            "{} vision_ms {:.3f} outside target {:.3f}".format(
                model_name, actual, expected)

    print("qwen3_vl_vision_layers={}".format(len(qwen.vision_decoder)))
    print("llava_next_tokens={}".format(tokens))
    print("vlm-graph-sanity-ok")


if __name__ == '__main__':
    main()
