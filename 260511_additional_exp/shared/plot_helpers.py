"""Matplotlib helpers for paper-style figures."""
import pathlib

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

COLORS = ["#2E86C1", "#E84D3D", "#27AE60", "#F39C12", "#8E44AD", "#1ABC9C"]


def setup():
    if not HAVE_PLT:
        return False
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })
    return True


def figdir():
    p = pathlib.Path(__file__).resolve().parents[1] / "results" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, name):
    p = figdir() / "{}.png".format(name)
    fig.savefig(str(p), bbox_inches="tight", facecolor="white")
    return p
