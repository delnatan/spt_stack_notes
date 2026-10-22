"""Figure 5: distributions of D across tracks (diffusionkit.gridpost.lognormal and .deconvolve).

Two populations (60% at D = 0.02, 40% at D = 0.5 um^2/s), track lengths 4-30 frames. Compares what is often
done (a histogram of per-track estimates, the average of posteriors) with the package's three population
models: one shared D, a log-normal, and the evidence-smoothed deconvolution.
"""
import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import (analyze_tracks, deconvolve_tracks, lognormal_tracks, shared_D_tracks)
from common import C, plt, save

acq = Acquisition(dt_s=0.035, exposure_s=0.020)
rng = np.random.default_rng(5)
parts, tid = [], 0
truth = []
for i in range(600):
    D = 0.02 if rng.random() < 0.6 else 0.5
    n = int(np.clip(rng.geometric(1 / 10) + 3, 4, 30))
    parts.append(simulate_tracks(1, n, 1.0, D, 0.03, acq, rng=rng, n_sub=10, first_track_id=i))
    truth.append(D)
table = pl.concat(parts)
res = analyze_tracks(table, acq, keep_likelihoods=True)
print(res.fits.columns)
u = res.options.u_D()
D = np.exp(u)
du = u[1] - u[0]
lls = res.likelihoods.loglik_D
post = np.exp(lls - lls.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)

dec = deconvolve_tracks(res)
ln = lognormal_tracks(res)
sh = shared_D_tracks(res)
print("lam", dec.lam, "lognormal", ln.summary(), "shared", sh.summary())

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
dens = lambda w: w / du  # per unit ln D
# A. the naive pictures
est = res.fits.filter(pl.col("status") == "ok")["D_post_mean_um2_s"].to_numpy()
bins = np.linspace(np.log(1e-4), np.log(5), 40)
ax[0].hist(np.log(est), bins=bins, density=True, color=C["grey"], alpha=.45, label="histogram of per-track E[D]")
ax[0].plot(u, dens(post.mean(0)), color=C["orange"], lw=1.8, label="average of per-track posteriors")
for d0, f in [(0.02, .6), (0.5, .4)]:
    ax[0].axvline(np.log(d0), color=C["black"], ls=":", lw=1)
ax[0].set_title("A. Per-track summaries smear the two populations")
# B. the population models
lo, hi = dec.band(.9)
ax[1].fill_between(u, dens(lo), dens(hi), color=C["blue"], alpha=.25, lw=0, label="deconvolution, 90% band")
ax[1].plot(u, dens(dec.weights), color=C["blue"], lw=1.8, label=f"deconvolution mode (λ = {dec.lam:.2g})")
ax[1].plot(u, dens(ln.weights), color=C["green"], lw=1.5, ls="--", label="log-normal fit")
s = sh.summary()
ax[1].axvspan(np.log(s["lo"]), np.log(s["hi"]), color=C["red"], alpha=.6, lw=0,
              label=f"one shared D: {s['median']:.3g} µm²/s" if "median" in s else "one shared D")
for d0 in (0.02, 0.5):
    ax[1].axvline(np.log(d0), color=C["black"], ls=":", lw=1)
ax[1].set_title("B. Population models fitted to the track likelihoods")
for a in ax[:2]:
    a.set_xlim(np.log(1e-4), np.log(5))
    t = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1])
    a.set_xticks(np.log(t)); a.set_xticklabels([f"{x:g}" for x in t])
    a.set_xlabel("D (µm²/s)"); a.set_ylabel("density per unit ln D")
    a.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(0, -0.18))
# C. evidence over lam
ax[2].semilogx(dec.lam_grid, dec.log_evidence - dec.log_evidence.max(), "o-", ms=3, color=C["blue"])
ax[2].axvline(dec.lam, color=C["red"], lw=1)
ax[2].set_ylim(-30, 3)
ax[2].set_xlabel("smoothness λ (rough ← → smooth)"); ax[2].set_ylabel("Laplace log evidence − max")
ax[2].set_title("C. The data choose the smoothness")
m = dec.mass(0.1)
print("mass above 0.1:", np.quantile(m, [.05, .5, .95]))
ax[2].text(0.03, 0.06, f"mass with D ≥ 0.1: {np.median(m):.2f} [{np.quantile(m,.05):.2f}, {np.quantile(m,.95):.2f}]"
           f"\ntrue share: {np.mean(np.array(truth) == 0.5):.2f}", transform=ax[2].transAxes, fontsize=7.5)
save(fig, "fig_05_populations.png")
