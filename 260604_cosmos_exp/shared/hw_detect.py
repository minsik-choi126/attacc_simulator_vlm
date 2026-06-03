"""Host-GPU auto-detection (copy of 260511_additional_exp/shared/hw_detect.py
trimmed for Cosmos experiments where the simulator integration is deferred).

Phase 1/3/4 measurement scripts only need detect_host() to tag JSON outputs
with a per-host suffix.  Sim-related helpers are kept for forward-compat.
"""
import subprocess


HW_TO_SIM_GPU = {
    "A6000": "A6000",
    "H100":  "H100",
    "A100":  "A100a",
}

HW_TO_INTERFACE = {
    "A6000": "NVLINK_BRIDGE",
    "H100":  "NVLINK4",
    "A100":  "NVLINK3",
}

HW_TO_PEAK_BW_TBS = {
    "A6000": 0.768,   # GDDR6 768 GB/s
    "H100":  3.35,    # HBM3 80GB ~ 3.35 TB/s (NVL: 3.9 TB/s SXM)
    "A100":  2.04,    # HBM2e 80GB ~ 2.04 TB/s
}


def detect_host():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=4).decode().strip()
        name = out.splitlines()[0].upper()
    except Exception:
        return "A6000"
    for tag in ("A6000", "H100", "A100"):
        if tag in name:
            return tag
    return "A6000"


def sim_gpu_tag(host=None):
    return HW_TO_SIM_GPU[host or detect_host()]


def sim_interface_tag(host=None):
    return HW_TO_INTERFACE[host or detect_host()]


def peak_bw_tbs(host=None):
    return HW_TO_PEAK_BW_TBS[host or detect_host()]


def gpu_count():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=4).decode().strip()
        return len([l for l in out.splitlines() if l.strip()])
    except Exception:
        return 0


if __name__ == "__main__":
    host = detect_host()
    print(f"detected host = {host}")
    print(f"  sim gpu tag = {sim_gpu_tag(host)}")
    print(f"  interface   = {sim_interface_tag(host)}")
    print(f"  peak BW     = {peak_bw_tbs(host)} TB/s")
    print(f"  gpu count   = {gpu_count()}")
