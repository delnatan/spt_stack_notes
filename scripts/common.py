"""Shared plotting style and paths for the explainer figures.

Run any script from this folder with spotsolve 0.6.0 and diffusionkit (main) installed:
    python fig_01_pipeline.py
Figures are written to ../figures.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

# A small, colour-blind-safe palette (Okabe-Ito), used consistently across figures.
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7",
         sky="#56B4E9", yellow="#F0E442", grey="#777777", black="#222222")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "image.cmap": "magma",
})


def save(fig, name):
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
