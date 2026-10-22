"""Slides: what the mean-squared displacement is and how it is computed from one track (diffusionkit.classic).

Two 16:9 figures, written to ../figures:

  msd_1_how.png    one simulated Brownian track: the displacement pairs at three lags, each lag's squared
                   displacements, and their average, which is `classic.compute_msd`'s time-averaged MSD
  msd_2_shape.png  what the curve's shape says: subdiffusive, Brownian and superdiffusive tracks (fBm), their MSD on
                   linear and log-log axes, and the localization offset 4 sigma^2

Tracks come from `fbmpost.simulate.simulate_tracks` with no exposure blur (the classic estimators assume none).
"""
import numpy as np

from diffusionkit import Acquisition
from diffusionkit.classic import MSDOptions, compute_msd
from diffusionkit.fbmpost.simulate import simulate_tracks
from common import C, plt, save

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12})
W, H = 11, 6.2
acq = Acquisition(dt_s=0.035, exposure_s=0.0)
s = 0.03

# ---- 1. how one track's MSD is computed ----
D, n = 0.1, 25
# seed 19: a track whose MSD stays within ~25% of 4Dt + 4s^2; many single tracks stray further (figure 7.1A)
tr = simulate_tracks(1, n, 1.0, D, s, acq, rng=np.random.default_rng(19))
xy = tr.select("x_um", "y_um").to_numpy()
xy = xy - xy[0]
L = 12
cv = compute_msd(tr, acq, MSDOptions(max_lag=L))
sq = {l: np.sum((xy[l:] - xy[:-l]) ** 2, axis=1) for l in range(1, L + 1)}
assert np.allclose([sq[l].mean() for l in range(1, L + 1)], cv.msd_um2)   # the same numbers compute_msd reports

show = {1: C["blue"], 4: C["orange"], 8: C["green"]}
fig = plt.figure(figsize=(W, H))
fig.suptitle("The MSD: average squared displacement over every pair of frames ℓ apart", x=.03, ha="left",
             fontsize=18)
fig.text(.03, .885, f"One simulated track: {n} frames, Δt = 35 ms, D = {D:g} µm²/s, localization σ = 30 nm",
         fontsize=12, color="0.4")
pad = .06
lo, hi = xy.min(0) - pad, xy.max(0) + pad
half = (hi - lo).max() / 2
mid = (lo + hi) / 2
for j, (l, col) in enumerate(show.items()):
    a = fig.add_axes([.02, .6 - j * .265, .2, .24])
    a.plot(*xy.T, "-o", color="0.75", ms=3, lw=1, zorder=1)
    for i in range(n - l):
        a.annotate("", xy[i + l], xy[i], arrowprops=dict(arrowstyle="-|>", color=col, lw=1, alpha=.55,
                                                         shrinkA=0, shrinkB=0, mutation_scale=7), zorder=2)
    a.set_xlim(mid[0] - half, mid[0] + half); a.set_ylim(mid[1] - half, mid[1] + half)
    a.set_aspect("equal"); a.axis("off")
    a.text(1.02, .62, f"lag ℓ = {l}", transform=a.transAxes, fontsize=14, color=col, weight="bold")
    a.text(1.02, .4, f"{n - l} pairs,\nframe i → i + {l}", transform=a.transAxes, fontsize=11.5, color="0.3",
           va="top", linespacing=1.3)
    if j == 2:
        a.plot([mid[0] + half - .2, mid[0] + half], [mid[1] - half] * 2, color="k", lw=2)
        a.text(mid[0] + half - .1, mid[1] - half - .03, "0.2 µm", ha="center", va="top", fontsize=11)

ax = fig.add_axes([.45, .14, .52, .68])
rng = np.random.default_rng(0)
for l in range(1, L + 1):
    col = show.get(l, "0.6")
    tau = l * acq.dt_s
    jit = rng.uniform(-.006, .006, len(sq[l]))
    ax.plot(tau + jit, sq[l], "o", ms=3.5, color=col, alpha=.45 if l in show else .3, lw=0)
ax.plot(cv.tau_s, cv.msd_um2, "o-", color="k", ms=8, lw=1.5, zorder=5, label="MSD(ℓ): mean of each column")
t = np.linspace(0, (L + .5) * acq.dt_s, 50)
ax.plot(t, 4 * D * t + 4 * s ** 2, ls=":", color="0.3", lw=1.5, label="expected: 4Dτ + 4σ²")
ax.set_xlim(0, (L + .5) * acq.dt_s)
ax.set_ylim(0, np.percentile(np.concatenate(list(sq.values())), 98))
ax.set_xlabel("lag τ = ℓΔt (s)")
ax.set_ylabel("squared displacement (µm²)")
ax.legend(frameon=False, loc="upper left")
ax.text(.03, .78, "MSD(ℓ) = mean over i of |x(i + ℓ) − x(i)|²\n(n − ℓ pairs: fewer, and more overlapping,\nat long lags)",
        transform=ax.transAxes, ha="left", va="top", fontsize=12.5, linespacing=1.4,
        bbox=dict(fc="white", ec="0.8", pad=6))
save(fig, "msd_1_how.png")

# ---- 2. what the shape of the curve says ----
cases = [(0.5, "subdiffusion  α = 0.5", C["blue"]), (1.0, "Brownian  α = 1", "0.3"),
         (1.5, "superdiffusion  α = 1.5", C["red"])]
n2, n_tr, L2 = 200, 60, 40
fig = plt.figure(figsize=(W, H))
fig.suptitle("The shape of the MSD curve: MSD = 4Kτ^α + 4σ²", x=.03, ha="left", fontsize=18)
fig.text(.03, .885, f"fBm tracks, D = 0.1 µm²/s at τ = 0.1 s, σ = 30 nm; curves averaged over {n_tr} tracks of "
         f"{n2} frames", fontsize=12, color="0.4")
lin = fig.add_axes([.3, .14, .32, .66])
lg = fig.add_axes([.69, .14, .25, .66])
ones, insets = [], []
for j, (alpha, name, col) in enumerate(cases):
    tt = simulate_tracks(n_tr, n2, alpha, 0.1, s, acq, rng=np.random.default_rng(10 + j))
    msd = np.mean([compute_msd(g, acq, MSDOptions(max_lag=L2)).msd_um2
                   for _, g in tt.group_by("track_id", maintain_order=True)], axis=0)
    tau = np.arange(1, L2 + 1) * acq.dt_s
    one = tt.filter(tt["track_id"] == 0).select("x_um", "y_um").to_numpy()[:80]
    one = one - one[0]
    ones.append(one)
    a = fig.add_axes([.02, .6 - j * .235, .21, .19])
    a.plot(*one.T, color=col, lw=1)
    a.plot(*one[0], "o", color="k", ms=4)
    a.axis("off")
    a.set_title(name, fontsize=12.5, color=col, loc="left")
    insets.append(a)
    lin.plot(tau, msd, color=col, lw=2.5, label=name)
    lg.loglog(tau, msd - 4 * s ** 2, color=col, lw=2.5)
    lg.loglog(tau, msd, color=col, lw=1, ls=":")
    lg.text(tau[-1] * 1.08, msd[-1] - 4 * s ** 2, f"α = {alpha:g}", color=col, fontsize=12, va="center")
# one scale for the three example tracks, so their reach can be compared
half = max(np.abs(o).max() for o in ones) * 1.05
for a, o in zip(insets, ones):
    c = (o.max(0) + o.min(0)) / 2
    a.set_xlim(c[0] - half, c[0] + half); a.set_ylim(c[1] - half / 1.9, c[1] + half / 1.9)
insets[-1].plot([c[0] + half - .5, c[0] + half], [c[1] - half / 2.1] * 2, color="k", lw=2)
insets[-1].text(c[0] + half - .25, c[1] - half / 2.1, "0.5 µm  ", ha="center", va="top", fontsize=11)
lin.set_xlim(0, tau[-1]); lin.set_ylim(0, None)
lin.set_xlabel("lag τ (s)"); lin.set_ylabel("MSD (µm²)")
lin.set_title("Linear axes: bends down, straight, bends up", loc="left", fontsize=13.5)
lg.set_xlabel("lag τ (s)"); lg.set_ylabel("MSD − 4σ² (µm²)")
lg.set_title("Log-log: the slope is α", loc="left", fontsize=13.5)
lg.text(.04, .96, "solid: localization offset 4σ²\nremoved; dotted: raw MSD,\nflattened at short lags by it",
        transform=lg.transAxes, va="top", fontsize=11, color="0.35", linespacing=1.3)
save(fig, "msd_2_shape.png")
