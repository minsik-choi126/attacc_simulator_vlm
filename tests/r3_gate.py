import argparse
import csv
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def run_case(label, args):
    output_csv = os.path.join(ROOT, 'output.csv')
    if os.path.exists(output_csv):
        os.remove(output_csv)
    cmd = [sys.executable, 'main.py'] + args
    print("[{}] {}".format(label, ' '.join(cmd)))
    subprocess.run(cmd, cwd=ROOT, check=True)
    with open(output_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, "Expected one output row for {}".format(label)
    return rows[0]


def as_float(row, key):
    return float(row[key])


def e2e_ms(row):
    lout = int(row['Lout'])
    return as_float(row, 's_time') + as_float(row, 'g_time (ms)') * max(
        0, lout - 1)


def run_scenario(tp, target_gain, ratio_range, args):
    common = [
        '--gpu', 'H100', '--ngpu', str(tp), '--tp', str(tp), '--num_attacc',
        str(tp), '--num_hbm', str(args.num_hbm), '--interface', 'NVLINK4',
        '--model', 'Qwen3-VL-4B', '--lin', str(args.lin), '--lout',
        str(args.lout), '--batch', str(args.batch), '--image_size',
        str(args.image_size), '--prefill_chunk', str(args.prefill_chunk),
        '--prefill_samples', str(args.prefill_samples), '--max_L',
        str(args.max_l), '--pipeopt', '--ffopt'
    ]
    base = run_case('R3.S{} baseline'.format(tp),
                    ['--system', 'dgx'] + common)
    pim = run_case('R3.S{} proposal'.format(tp),
                   ['--system', 'dgx-attacc'] + common)

    base_e2e = e2e_ms(base)
    pim_e2e = e2e_ms(pim)
    gain = base_e2e / pim_e2e
    interface_ratio = as_float(pim, 's_comm') / max(1e-9,
                                                    as_float(pim, 's_matmul'))
    lo = target_gain * 0.8
    hi = target_gain * 1.2
    assert lo <= gain <= hi, \
        "R3.S{} gain {:.3f} outside [{:.3f}, {:.3f}]".format(
            tp, gain, lo, hi)
    assert ratio_range[0] <= interface_ratio <= ratio_range[1], \
        "R3.S{} interface/PIM {:.3f} outside [{:.3f}, {:.3f}]".format(
            tp, interface_ratio, ratio_range[0], ratio_range[1])
    print("R3.S{} baseline_e2e_ms={:.3f} proposal_e2e_ms={:.3f} "
          "gain={:.3f} interface_over_pim={:.3f}".format(
              tp, base_e2e, pim_e2e, gain, interface_ratio))


def main():
    parser = argparse.ArgumentParser(
        description="Run R3 corrected-E2 gate. Requires pandas + Ramulator.")
    parser.add_argument('--lin', type=int, default=569)
    parser.add_argument('--lout', type=int, default=128)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=672)
    parser.add_argument('--prefill_chunk', type=int, default=512)
    parser.add_argument('--prefill_samples', type=int, default=8)
    parser.add_argument('--max_l', type=int, default=2048)
    parser.add_argument('--num_hbm', type=int, default=5)
    args = parser.parse_args()

    run_scenario(1, 1.58, (0.5, 0.7), args)
    run_scenario(2, 1.53, (0.2, 0.4), args)
    print("r3-gate-ok")


if __name__ == '__main__':
    main()
