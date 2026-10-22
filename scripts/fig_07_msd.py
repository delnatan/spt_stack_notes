"""Figure 7: the classic MSD estimates beside the likelihood (diffusionkit.classic vs diffusionkit.gridpost).

Uses the real `classic.compute_msd`, `fit_linear_msd`, `fit_loglog_msd`, `classic.analyze_tracks` (default
window: 3 lags, known-SD offset subtracted, OLS through the origin) and `gridpost.analyze_tracks`.
"""
import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.classic import MSDOptions, analyze_tracks as msd_tracks, compute_msd, fit_linear_msd, fit_loglog_msd
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import analyze_tracks as grid_tracks
from common import C, plt, save

rng = np.random.default_rng(7)
s = 0.03
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)

# A. one 60-frame track's MSD (no exposure blur here: compute_msd has no blur model)
acq0 = Acquisition(dt_s=0.035, exposure_s=0.0)
D0 = 0.1
tr = simulate_tracks(1, 60, 1.0, D0, s, acq0, rng=rng)
cv = compute_msd(tr, acq0, MSDOptions(max_lag=20))
lin = fit_linear_msd(cv, 6).parameters
ll = fit_loglog_msd(cv, max(lin["offset_um2"], 0), 20).parameters
tau = cv.tau_s
ax[0].plot(tau, cv.msd_um2, "o", ms=3.5, color=C["black"], label="time-averaged MSD")
ax[0].plot(tau, 4 * D0 * tau + 4 * s**2, color=C["grey"], ls=":", label=f"truth 4Dτ + 4σ² (D = {D0})")
ax[0].plot(tau, 4 * lin["D_um2_s"] * tau + lin["offset_um2"], color=C["blue"],
           label=f"linear, 6 lags: D = {lin['D_um2_s']:.3f}")
ax[0].plot(tau, 4 * ll["K_um2_s_alpha"] * tau ** ll["alpha"] + max(lin["offset_um2"], 0), color=C["red"], ls="--",
           label=f"log-log, 20 lags: α = {ll['alpha']:.2f}")
ax[0].set_xlabel("lag τ (s)"); ax[0].set_ylabel("MSD (µm²)")
ax[0].set_title("A. One 60-frame Brownian track")
ax[0].legend(frameon=False, fontsize=7)
n_pairs_note = cv.n_pairs
ax[0].text(0.98, 0.04, f"pairs per lag: {n_pairs_note[0]} at lag 1, {n_pairs_note[-1]} at lag 20",
           transform=ax[0].transAxes, ha="right", fontsize=7, color=C["grey"])

# B, C. per-track D from the two methods. The classic estimators exclude tracks with exposure blur
# ("MSD estimators assume exposure 0"), so this comparison uses instantaneous frames.
acq = acq0
lengths = [5, 10, 20, 50]
for a, Dt in zip(ax[1:], [0.1, 0.01]):
    rows = []
    for k, n in enumerate(lengths):
        t = simulate_tracks(400, n, 1.0, Dt, s, acq, rng=rng, n_sub=10)
        m = msd_tracks(t, acq).fits.filter(pl.col("model") == "brownian")["D_um2_s"].to_numpy()
        g = grid_tracks(t, acq).fits
        e, lo, hi = (g[c].to_numpy() for c in ("D_post_mean_um2_s", "D_post_lo_um2_s", "D_post_hi_um2_s"))
        cov = np.mean((lo <= Dt) & (Dt <= hi))
        neg = np.mean(m <= 0)
        rows.append((n, np.median(m) / Dt, np.median(e) / Dt, cov, neg))
        pos = np.array([k - .17, k + .17])
        bp = a.boxplot([np.clip(m / Dt, 1e-2, None), e / Dt], positions=pos, widths=.28, showfliers=False,
                       patch_artist=True, medianprops=dict(color="k"))
        for patch, col in zip(bp["boxes"], [C["orange"], C["blue"]]):
            patch.set_facecolor(col); patch.set_alpha(.6)
        lbl = f"{neg:.0%} ≤ 0\n{cov:.0%} cov." if neg > 0.005 else f"{cov:.0%} cov."
        a.annotate(lbl, (k, 0.99), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=6.5)
    print(Dt, rows)
    a.axhline(1, color=C["grey"], ls=":")
    a.set_xticks(range(len(lengths))); a.set_xticklabels([str(n) for n in lengths])
    a.set_xlabel("track length (frames)"); a.set_ylabel("estimate / true D")
    a.set_yscale("log"); a.set_ylim(8e-3, 40)
    a.set_title(f"{'B' if Dt == 0.1 else 'C'}. 400 tracks per length, true D = {Dt} µm²/s"
                + (f"\n(localization floor σ²/Δt = {s**2 / acq.dt_s:.3f})" if Dt < 0.02 else ""))
handles = [plt.matplotlib.patches.Patch(color=C["orange"], alpha=.6, label="MSD, 3 lags, offset-corrected"),
           plt.matplotlib.patches.Patch(color=C["blue"], alpha=.6, label="gridpost posterior mean E[D]")]
ax[2].legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.18))
ax[1].text(0.0, -0.2, "Labels: share of MSD estimates ≤ 0, and \"cov.\", the share of\ngridpost 90% intervals "
           "holding the true D", transform=ax[1].transAxes, va="top",
           fontsize=6.5, color=C["grey"])
save(fig, "fig_07_msd.png")
