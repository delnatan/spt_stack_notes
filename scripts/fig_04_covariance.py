"""Figure 4.1: how the fBm displacement covariance is built (diffusionkit.fbmpost.covariance).

Row A: the fGn correlation matrix (no blur) at alpha = 0.5, 1 and 1.5, and the one number per lag, gamma(k)/gamma(0),
that fills each diagonal. Row B: the full Sigma = K C_alpha + B for one track (alpha = 0.6, 20 ms exposure, frame 3
localized twice as poorly), laid out like figure 3.1. Uses the real `fgn_autocovariance`, `fgn_motion_covariance` and
gridpost's `localization_covariance`, and checks that alpha = 1 reproduces gridpost's `motion_covariance`.
"""
import numpy as np

from diffusionkit import Acquisition
from diffusionkit.fbmpost.covariance import fgn_autocovariance, fgn_motion_covariance
from diffusionkit.gridpost.likelihood import localization_covariance, motion_covariance
from common import C, plt, save

acq = Acquisition(dt_s=0.035, exposure_s=0.020)
m = 8
alphas = [0.5, 1.0, 1.5]
cols = [C["blue"], C["grey"], C["red"]]

# alpha = 1 with blur is gridpost's Brownian matrix A (figure 3.1)
assert np.allclose(fgn_motion_covariance(m, acq.dt_s, 1.0, acq.exposure_s),
                   motion_covariance(m, acq.dt_s, acq.exposure_s))

fig = plt.figure(figsize=(11, 6.2))
gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, .05, .32, 1.2], hspace=.45, wspace=.3,
                      left=.05, right=.99, top=.9, bottom=.08)

# ---- row A: one number per lag fills each diagonal
for j, a in enumerate(alphas):
    ax = fig.add_subplot(gs[0, j])
    Ca = fgn_motion_covariance(m, acq.dt_s, a)
    R = Ca / Ca[0, 0]
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(m):
        for k in range(m):
            if abs(i - k) <= 1 and k >= i:
                ax.text(k, i, f"{R[i, k]:.2f}".replace("-0", "−0").replace("0.00", "0"), ha="center", va="center",
                        fontsize=6.5, color="w" if abs(R[i, k]) > .6 else "k")
    ax.set_xticks(range(m)); ax.set_yticks(range(m))
    ax.set_xlabel("displacement j")
    if j == 0:
        ax.set_ylabel("displacement i")
    kind = "subdiffusion" if a < 1 else ("Brownian" if a == 1 else "superdiffusion")
    ax.set_title(f"{'ABC'[j]}. α = {a:g} ({kind}), no blur")
cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]))
cb.set_label("correlation")

ax = fig.add_subplot(gs[0, 5])
k = np.arange(m)
w = .26
for j, (a, col) in enumerate(zip(alphas, cols)):
    g = fgn_autocovariance(m, acq.dt_s, a)
    ax.bar(k + (j - 1) * w, g / g[0], width=w, color=col, label=f"α = {a:g}")
ax.axhline(0, color="k", lw=.6)
ax.set_xticks(k)
ax.set_xlabel("lag k = |i − j| (frames)")
ax.set_ylabel("γ(k) / γ(0)")
ax.set_ylim(-.4, 1.05)
ax.set_title("D. One number per lag fills each diagonal")
ax.legend(frameon=False, loc="upper right")
ax.text(1.3, -.36, "lag 1: 2^(α−1) − 1", fontsize=8, color="0.3")

# ---- row B: Sigma = K C_alpha + B for one track, as in figure 3.1
alpha, D_ref, tau_ref, s = 0.5, 0.05, 0.1, 0.03
K = D_ref * tau_ref ** (1 - alpha)                 # D_ref = K tau_ref^(alpha - 1)
sd = np.full((m + 1, 2), s)
sd[3] = 0.06                                       # one poorly localized frame
KC = K * fgn_motion_covariance(m, acq.dt_s, alpha, acq.exposure_s)
B = localization_covariance(sd)[0]
vmax = np.abs(KC + B).max()
titles = [f"E. K·C_α: blurred fBm (α = {alpha:g})", "F. B: localization (frame 3 noisy)", "G. Σ = K·C_α + B"]
for j, (M, t) in enumerate(zip([KC, B, KC + B], titles)):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(t)
    ax.set_xticks(range(m)); ax.set_yticks(range(m))
    ax.set_xlabel("displacement j")
    if j == 0:
        ax.set_ylabel("displacement i")
        for c in range(m):                          # first row, in units of 1e-4 um^2
            ax.text(c, 0, f"{1e4 * M[0, c]:.0f}".replace("-0", "0").replace("-", "−"), ha="center", va="center", fontsize=6.5)
cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 3]))
cb.set_label("covariance (µm²), per axis")
ax = fig.add_subplot(gs[1, 5])
ax.axis("off")
ax.text(0, 1, "E: every diagonal is filled, so all\ndisplacements are correlated (Brownian,\nfigure 3.1: only the "
        "first off-diagonal,\nfrom blur). Row 0 is labelled in 10⁻⁴ µm².\n"
        f"Blur lifts the lag-1 correlation from\n{2 ** (alpha - 1) - 1:.2f} (panel A) to {KC[0, 1] / KC[0, 0]:.2f}.\n\n"
        "K = D_ref · τ_ref^(1−α), with\n"
        f"D_ref = {D_ref:g} µm²/s at τ_ref = {tau_ref:g} s.\n\n"
        "Δt = 35 ms, exposure 20 ms,\ns = 30 nm (60 nm at frame 3).", va="top", fontsize=8.5, linespacing=1.4)
save(fig, "fig_04_covariance.png")
print("lag-1 correlations", {a: round(float(fgn_autocovariance(3, acq.dt_s, a)[1] / fgn_autocovariance(3, acq.dt_s, a)[0]), 3)
                             for a in alphas})
print(f"blurred lag-1 at alpha={alpha}:", round(float(KC[0, 1] / KC[0, 0]), 3))
