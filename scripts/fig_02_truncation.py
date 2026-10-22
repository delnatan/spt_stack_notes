"""Figure 2.3: the implied radius of a link and what it does to steps (analytic, from track.rs's gain).

With a track's posterior settled on one D, the step density is a 2-D Gaussian of per-axis variance v, and the gain
-ln(2 pi v) - d^2 / (2 v) - ln(lambda) - c is positive only for d^2 < 2 v ln(1 / (2 pi v lambda e^c)). The share
of true steps cut is f = 2 pi v lambda e^c, and the kept steps' mean square is reduced by 1 - x0 f / (1 - f),
x0 = -ln f. (Breaks also end the track, so short tracks of fast particles are over-represented downstream.)
"""
import numpy as np

from common import C, plt, save

c = 2.5
se2 = 0.09
D = np.geomspace(0.01, 10, 300)              # px^2 per frame
v = 2 * D + 2 * se2
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
for q, rho, col in [(0.9, 0.002, C["blue"]), (0.8, 0.004, C["orange"]), (0.7, 0.01, C["red"])]:
    lam = (1 - q) ** 2 * rho / q
    f = np.clip(2 * np.pi * v * lam * np.exp(c), 1e-12, 1)
    x0 = -np.log(f)
    bias = np.where(f < 1, 1 - x0 * f / (1 - f), 0)
    lab = f"q = {q}, ρ = {rho} /px²  (λe^c = {lam*np.exp(c):.1e})"
    ax[0].plot(D, f, color=col, label=lab)
    ax[1].plot(D, bias, color=col)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 1)
ax[0].set_xlabel("track's D (px²/frame)"); ax[0].set_ylabel("share of true steps that cannot link")
ax[0].set_title("A. Steps cut by the gain threshold: f = 2π v λ e^c")
ax[0].legend(frameon=False, fontsize=7)
ax[1].set_xscale("log"); ax[1].set_ylim(0.5, 1.01)
ax[1].set_xlabel("track's D (px²/frame)"); ax[1].set_ylabel("⟨d²⟩ kept / ⟨d²⟩ true")
ax[1].set_title("B. Mean-square step of the steps that do link")
save(fig, "fig_02_truncation.png")
