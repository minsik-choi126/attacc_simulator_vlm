"""Host-GPU auto-detection + simulator config mapping.

Shared by Phase 1/2/3 experiment scripts so they automatically pick the
right simulator GPU type (`gpu=` for sim_runner.run) and interconnect
when invoked on either an A6000 or H100 node, without manual flags.

If `nvidia-smi` is unavailable (e.g. CI / Windows laptop), `detect_host()`
falls back to "A6000" so local syntax/sanity runs still produce
well-formed configs.

Usage:
    from hw_detect import detect_host, sim_gpu_tag, sim_interface_tag
    HOST = detect_host()
    sr.run(model=m, gpu=sim_gpu_tag(HOST),
           interface=sim_interface_tag(HOST), ...)
"""
import subprocess


# host GPU label -> (sim_runner gpu tag, sim_runner interface tag)
# sim_runner's gpu accepts "A100a", "H100", "A6000".
# interface accepts "NVLINK3", "NVLINK4", "NVLINK_BRIDGE".
HW_TO_SIM_GPU = {
    "A6000": "A6000",
    "H100":  "H100",
    "A100":  "A100a",
}

HW_TO_INTERFACE = {
    # A6000 workstation: NVLink Bridge ~ 112 GB/s aggregate
    "A6000": "NVLINK_BRIDGE",
    # H100 SXM5: NVLink 4 ~ 900 GB/s aggregate
    "H100":  "NVLINK4",
    # A100 SXM4: NVLink 3 ~ 600 GB/s aggregate (paper's DGX-A100 default)
    "A100":  "NVLINK3",
}


def detect_host():
    """Return one of 'A6000', 'H100', 'A100'.

    Falls back to 'A6000' if nvidia-smi is missing or returns an
    unrecognised name -- this keeps Windows / CI environments
    syntax-checkable without crashing.
    """
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
    """Map host GPU -> sim_runner gpu argument."""
    return HW_TO_SIM_GPU[host or detect_host()]


def sim_interface_tag(host=None):
    """Map host GPU -> sim_runner interface argument."""
    return HW_TO_INTERFACE[host or detect_host()]


def gputype_enum(host=None):
    """Return src.type.GPUType enum value for the detected host.

    Useful for scripts that build System() / make_xpu_config() directly.
    """
    from src.type import GPUType
    host = host or detect_host()
    return {
        "A6000": GPUType.A6000,
        "H100":  GPUType.H100,
        "A100":  GPUType.A100a,
    }[host]


if __name__ == "__main__":
    host = detect_host()
    print(f"detected host = {host}")
    print(f"  sim gpu tag   = {sim_gpu_tag(host)}")
    print(f"  interface tag = {sim_interface_tag(host)}")
