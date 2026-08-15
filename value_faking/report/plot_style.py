"""Shared plotting style utilities for publication-ready figures."""

import os

import matplotlib
matplotlib.use("Agg")  # headless rendering — avoids X11 pixmap allocation failures on large figures

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# SciencePlots' IEEE style asks for "Times", which is almost never installed under
# that exact family name on Linux. Matplotlib then warns once per text object --
# hundreds of lines for a single figure -- and silently falls back. Rather than
# suppress the warning, resolve to the best Times-metric font actually present.
# Nimbus Roman and Liberation Serif are metric-compatible Times clones, so the
# figure keeps its intended proportions; DejaVu Serif ships with matplotlib and
# therefore always terminates the list.
_SERIF_PREFERENCE = [
    "Times New Roman",   # the real thing, if the user has msttcorefonts
    "Nimbus Roman",      # URW Times clone, metric-compatible
    "Liberation Serif",  # Red Hat Times clone, metric-compatible
    "STIXGeneral",       # Times-like, bundled with matplotlib
    "DejaVu Serif",      # always present; last resort
]


def _available_serif() -> list[str]:
    from matplotlib import font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    found = [n for n in _SERIF_PREFERENCE if n in installed]
    return found or ["DejaVu Serif"]


def apply_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee"])
    except (ImportError, OSError):
        plt.rcParams.update({
            "font.family":       "serif",
            "axes.linewidth":    0.8,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "xtick.direction":   "in",
            "ytick.direction":   "in",
            "figure.dpi":        150,
        })

    # SciencePlots' "science" style turns text.usetex on, which routes every label
    # through LaTeX and hard-fails on any character TeX has no glyph for (U+2248,
    # U+2212, arrows) as well as on machines with no TeX installed at all. Figures
    # must render from a bare `pip install -r requirements.txt`, so usetex is
    # opt-in: set VALET_USETEX=1 when producing camera-ready figures on a machine
    # that has LaTeX, and keep labels ASCII-safe so both paths agree.
    plt.rcParams["text.usetex"] = os.environ.get("VALET_USETEX", "") == "1"

    # Point font.serif at fonts that exist on this machine (see _available_serif).
    # Under usetex the family is chosen by the TeX preamble instead, so this only
    # governs the default renderer -- which is the one most users will hit.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = _available_serif() + plt.rcParams["font.serif"]

    # IEEE's default font.size (~7pt) reads fine in a two-column layout but is too
    # small once these figures are placed in a single-column context; bump it (and
    # everything that inherits from it) regardless of which branch above ran.
    plt.rcParams.update({
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
    })


# ── Paul Tol colorblind-safe diverging colormap ───────────────────────────────
# Blue (#0077BB) → light grey (#EEEEEE) → Orange (#EE7733)
# Safe for deuteranopia, protanopia, and tritanopia.

_TOL_DIV_COLORS = ["#0077BB", "#DDDDDD", "#EE7733"]
TOL_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "tol_div", _TOL_DIV_COLORS, N=512
)

# Darker orange endpoint for higher-contrast positive values
_TOL_DIV_STRONG_COLORS = ["#0077BB", "#F5F5F5", "#CC3311"]
TOL_DIVERGING_STRONG = mcolors.LinearSegmentedColormap.from_list(
    "tol_div_strong", _TOL_DIV_STRONG_COLORS, N=512
)


def gap_norm(vmin: float = -0.25, vmax: float = 0.9) -> mcolors.TwoSlopeNorm:
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def _luminance(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def cell_text_color(bg_rgba) -> str:
    return "white" if _luminance(bg_rgba[:3]) < 0.50 else "#1A252F"


def savefig(fig, path: str, dpi: int = 300):
    import os
    stem, ext = os.path.splitext(path)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # always save PDF (vector) first
    pdf_path = stem + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"saved -> {pdf_path}")

    # also save PNG at requested DPI
    png_path = stem + ".png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"saved -> {png_path}")
