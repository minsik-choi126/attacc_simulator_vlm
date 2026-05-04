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


def e2e_ms(row):
    lout = int(row['Lout'])
    return float(row['s_time']) + float(row['g_time (ms)']) * max(0, lout - 1)


def main():
    parser = argparse.ArgumentParser(
        description="Run R2 GPT-175B paper repro gate. Requires Ramulator.")
    parser.add_argument('--lin', type=int, default=2048)
    parser.add_argument('--lout', type=int, default=128)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--target', type=float, default=4.84)
    parser.add_argument('--tolerance', type=float, default=0.20)
    parser.add_argument('--word', type=int, default=2)
    args = parser.parse_args()

    common = [
        '--gpu', 'A100a', '--ngpu', '8', '--tp', '8', '--num_attacc', '8',
        '--num_hbm', '5', '--interface', 'NVLINK3', '--model', 'GPT-175B',
        '--lin', str(args.lin), '--lout', str(args.lout), '--batch',
        str(args.batch), '--word', str(args.word), '--max_L', str(args.lin),
        '--pipeopt', '--ffopt'
    ]
    base = run_case('R2 baseline', ['--system', 'dgx'] + common)
    pim = run_case('R2 dgx-attacc',
                   ['--system', 'dgx-attacc', '--powerlimit'] + common)
    base_e2e = e2e_ms(base)
    pim_e2e = e2e_ms(pim)
    gain = base_e2e / pim_e2e
    lo = args.target * (1 - args.tolerance)
    hi = args.target * (1 + args.tolerance)
    assert lo <= gain <= hi, \
        "R2 gain {:.3f} outside [{:.3f}, {:.3f}]".format(gain, lo, hi)
    print("R2 baseline_e2e_ms={:.3f} attacc_e2e_ms={:.3f} gain={:.3f}".
          format(base_e2e, pim_e2e, gain))
    print("r2-paper-repro-ok")


if __name__ == '__main__':
    main()
