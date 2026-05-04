import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.system import System


def assert_close(name, actual, expected, abs_tol=0.01):
    if abs(actual - expected) > abs_tol:
        raise AssertionError("{}: got {:.4f}, expected {:.4f}".format(
            name, actual, expected))


def main():
    cases = [
        ('qwen3_vl_s1', 8, 1, 0.80),
        ('qwen3_vl_s2', 4, 1, 0.57),
        ('qwen25_vl_s1', 4, 1, 0.57),
        ('qwen25_vl_s2', 2, 1, 0.29),
        ('llava15_s1', 32, 1, 0.91),
        ('llava15_s2', 16, 1, 0.80),
        ('batch_ge_2', 2, 2, 1.00),
    ]
    for name, n_kv, batch_size, expected in cases:
        actual = System.get_pipelining_efficiency_latency(
            n_kv, num_hbm=5, batch_size=batch_size)
        assert_close(name, actual, expected)
        print("{}={:.2f}".format(name, actual))
    print("m6_4-eff-lat-ok")


if __name__ == '__main__':
    main()
