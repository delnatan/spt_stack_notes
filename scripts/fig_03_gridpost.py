"""Figures 3.1-3.2: the Brownian displacement likelihood and per-track grid posteriors (diffusionkit.gridpost).

Uses the real `gridpost.likelihood.motion_covariance`, `localization_covariance` and `posterior.track_loglik`;
tracks come from `fbmpost.simulate.simulate_tracks` at alpha = 1 (exposure-averaged on a fine time grid,
independently of the likelihood's closed form).
"""
import numpy as np

from diffusionkit import Acquisition
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import GridPostOptions
from diffusionkit.gridpost import posterior as P
from diffusionkit.gridpost.likelihood import localization_covariance, motion_covariance
from common import C, plt, save

acq = Acquisition(dt_s=0.035, exposure_s=0.020)
opts = GridPostOptions()
u = opts.u_D()

# ---- Figure 3.1: the covariance of 8 displacements, Sigma = D A + B
m, D, s = 8, 0.05, 0.03
sd = np.full((m + 1, 2), s)
sd[3] = 0.06                                       # one poorly localized frame
A = motion_covariance(m, acq.dt_s, acq.exposure_s)
B = localization_covariance(sd)[0]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1), constrained_layout=True)
vmax = np.abs(D * A + B).max()
for a, M, t in zip(ax, [D * A, B, D * A + B], ["D·A: blurred Brownian motion", "B: localization (frame 3 noisy)",
                                               "Σ(D) = D·A + B"]):
    im = a.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    a.set_title(t); a.set_xticks(range(m)); a.set_yticks(range(m))
    a.set_xlabel("displacement j")
ax[0].set_ylabel("displacement i")
fig.colorbar(im, ax=ax, fraction=0.03, label="covariance (µm²), per axis")
save(fig, "fig_03_covariance.png")

# ---- Figure 3.2: per-track posteriors
rng = np.random.default_rng(2)
cases = [(5, 0.05, C["orange"]), (20, 0.05, C["blue"]), (100, 0.05, C["green"]), (20, 0.0008, C["purple"])]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
for n, Dt, col in cases:
    tr = simulate_tracks(1, n, 1.0, Dt, s, acq, rng=rng)
    ll = P.track_loglik(tr, acq, u)
    lp = P.log_posterior(ll, P.flat(u))
    p = np.exp(lp)
    sm = P.summary(p, u, 0.9)
    floor = P.localization_floor(tr, acq)
    ax[0].plot(np.exp(u), p / p.max(), color=col,
               label=f"{n} frames, true D = {Dt:g}: 90% [{sm['lo']:.2g}, {sm['hi']:.2g}], edge={P.grid_edge(p)}")
    ax[0].axvline(Dt, color=col, ls=":", lw=1)
ax[0].axvspan(1e-5, floor, color=C["grey"], alpha=0.12, lw=0)
ax[0].text(floor * 0.8, 1.02, "below localization floor", ha="right", fontsize=7, color=C["grey"])
ax[0].set_xscale("log"); ax[0].set_xlim(1e-5, 10)
ax[0].set_xlabel("D (µm²/s)"); ax[0].set_ylabel("posterior (flat in ln D), scaled to peak")
ax[0].set_title("A. One track each: the posterior's width is the answer's precision")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.18))

# coverage check: 400 tracks of 10 frames at D = 0.05; how often the 90% interval holds the truth
hits, means, ns = [], [], []
for n in (4, 6, 10, 20, 40):
    tr = simulate_tracks(300, n, 1.0, 0.05, s, acq, rng=rng)
    h, mm = 0, []
    for (tid,), g in tr.group_by("track_id"):
        sm = P.summary(np.exp(P.log_posterior(P.track_loglik(g, acq, u), P.flat(u))), u, 0.9)
        h += sm["lo"] <= 0.05 <= sm["hi"]
        mm.append(sm["mean"])
    hits.append(h / 300); means.append(np.median(mm) / 0.05); ns.append(n)
ax[1].plot(ns, hits, "o-", color=C["blue"], label="coverage of the 90% interval")
ax[1].plot(ns, means, "s-", color=C["orange"], label="median E[D] / true D")
ax[1].axhline(0.9, color=C["blue"], ls=":", lw=0.8); ax[1].axhline(1, color=C["orange"], ls=":", lw=0.8)
ax[1].set_xscale("log"); ax[1].set_xticks(ns, [str(n) for n in ns]); ax[1].xaxis.set_minor_formatter(plt.NullFormatter())
ax[1].set_xlabel("frames per track (D = 0.05 µm²/s, σ = 30 nm, dt 35 ms, exposure 20 ms)")
ax[1].set_title("B. Calibration under the model (300 tracks per point)")
ax[1].legend(frameon=False)
save(fig, "fig_03_posteriors.png")
print("coverage", dict(zip(ns, hits)), "median mean ratio", dict(zip(ns, np.round(means, 3))))
