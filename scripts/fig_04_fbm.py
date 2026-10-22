"""Figure 4: anomalous exponent alpha by profile likelihood (diffusionkit.fbmpost).

Uses the real `fbmpost.covariance.fgn_autocovariance`, `likelihood.log_likelihood_surface`,
`analyze_tracks(..., keep_profiles=True)` and `shared_alpha`.
"""
import numpy as np
from scipy.stats import chi2

from diffusionkit import Acquisition
from diffusionkit.fbmpost import FbmPostOptions, analyze_tracks, shared_alpha, simulate_tracks
from diffusionkit.fbmpost.covariance import fgn_autocovariance
from diffusionkit.fbmpost.likelihood import log_likelihood_surface
from common import C, plt, save

acq = Acquisition(dt_s=0.035, exposure_s=0.020)
opts = FbmPostOptions()
rng = np.random.default_rng(4)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)

# A. correlation of successive displacements
k = np.arange(8)
for a, col in zip([0.4, 0.7, 1.0, 1.4], [C["red"], C["orange"], C["black"], C["blue"]]):
    g = fgn_autocovariance(8, acq.dt_s, a, acq.exposure_s)
    ax[0].plot(k, g / g[0], "o-", ms=3, color=col, label=f"α = {a}")
    g0 = fgn_autocovariance(8, acq.dt_s, a, 0.0)
    ax[0].plot(k, g0 / g0[0], ":", color=col, lw=1)
ax[0].axhline(0, color=C["grey"], lw=.6)
ax[0].set_xlabel("lag k (frames)"); ax[0].set_ylabel("γ(k) / γ(0)")
ax[0].set_title("A. Displacement correlation\n(solid: 20 ms exposure, dotted: none)")
ax[0].legend(frameon=False)

# B. one track's likelihood over (alpha, log10 D_ref): a ridge
tr = simulate_tracks(1, 30, 0.6, 0.1, 0.03, acq, rng=rng)
xs = np.linspace(-3, 0.5, 141)
S = log_likelihood_surface(tr, acq, opts, xs)
S -= S.max()
al = opts.alphas()
im = ax[1].pcolormesh(xs, al, np.maximum(S, -15), cmap="magma", shading="auto")
ax[1].contour(xs, al, S, levels=[-chi2.ppf(.95, 1) / 2], colors="white", linewidths=1)
ax[1].plot(np.log10(0.1), 0.6, "c+", ms=10, mew=2)
ax[1].set_xlabel("log10 D_ref (µm²/s at τ_ref = 0.1 s)"); ax[1].set_ylabel("α")
ax[1].set_title("B. One 30-frame track (α = 0.6)\nwhite: 95% LR region; +: truth")
fig.colorbar(im, ax=ax[1], fraction=.05, label="log-lik − max")

# C. per-track profiles and their sum
tracks = simulate_tracks(60, 20, 0.6, 0.1, 0.03, acq, rng=rng)
res = analyze_tracks(tracks, acq, opts, keep_profiles=True)
P = res.profiles
for row in P.loglik[:12]:
    ax[2].plot(P.alphas, row, color=C["sky"], lw=.8, alpha=.8)
sh = shared_alpha(P)
ax[2].axhline(-chi2.ppf(.95, 1) / 2, color=C["grey"], ls="--", lw=.8)
ax[2].axvline(0.6, color=C["black"], ls=":", lw=1)
ax[2].set_ylim(-8, .5); ax[2].set_xlabel("α"); ax[2].set_ylabel("profile log-lik − max")
ax2 = ax[2].twinx()
ax2.plot(sh.alphas, sh.loglik, color=C["red"], lw=2)
ax2.set_ylim(-40, 2); ax2.set_ylabel("sum over 60 tracks", color=C["red"])
ax2.spines["right"].set_visible(True)
ax[2].set_title(f"C. Profiles ℓp(α) of 12 tracks (blue);\nsum of 60 (red): α = {sh.alpha_hat:.2f} "
                f"[{sh.alpha_lo:.2f}, {sh.alpha_hi:.2f}]")

# D. per-track alpha_hat vs length (spread of single-track estimates)
for n, col in [(10, C["orange"]), (30, C["blue"]), (100, C["green"])]:
    f = analyze_tracks(simulate_tracks(200, n, 0.6, 0.1, 0.03, acq, rng=rng), acq, opts).fits
    a = f["alpha_hat"].drop_nulls().drop_nans().to_numpy()
    lo, hi = f["alpha_lo"].to_numpy(), f["alpha_hi"].to_numpy()
    cov = np.mean((lo <= 0.6) & (hi >= 0.6))
    ax[3].hist(a, bins=np.linspace(0, 2, 41), histtype="step", color=col, lw=1.5,
               label=f"{n} frames: median {np.median(a):.2f}, 95% LR coverage {cov:.2f}")
    print(n, np.median(a), cov)
ax[3].axvline(0.6, color=C["black"], ls=":", lw=1)
ax[3].set_xlabel("per-track α̂"); ax[3].set_ylabel("tracks")
ax[3].set_title("D. Single-track α̂ for true α = 0.6")
ax[3].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.2))
print("shared", sh.summary())
save(fig, "fig_04_fbm.png")
