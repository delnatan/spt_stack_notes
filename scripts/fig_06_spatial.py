"""Figure 6: spatial classes of D on the track neighbour graph (diffusionkit.spatial).

800 tracks scattered over a 20 x 20 um field: inside a 6 um disc D = 0.05, outside D = 0.2 um^2/s, track lengths
3-8 frames, so a track's own data often cannot tell the two apart. A control movie reuses the same tracks with their positions shuffled (no pattern). Uses the real
`gridpost.analyze_tracks(..., keep_likelihoods=True)` and `spatial.analyze` with the default 99 shuffles.
"""
import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import analyze_tracks
from diffusionkit.spatial import SpatialOptions, analyze
from common import C, plt, save

acq = Acquisition(dt_s=0.035, exposure_s=0.020)
rng = np.random.default_rng(11)
N = 800
xy = rng.uniform(0, 20, (N, 2))
inside = np.hypot(*(xy - 10).T) < 6
Dtrue = np.where(inside, 0.05, 0.2)
parts = []
for i in range(N):
    n = int(rng.integers(3, 9))
    t = simulate_tracks(1, n, 1.0, Dtrue[i], 0.03, acq, rng=rng, n_sub=10, first_track_id=i)
    parts.append(t.with_columns(pl.col("x_um") + xy[i, 0], pl.col("y_um") + xy[i, 1]))
table = pl.concat(parts)
res = analyze_tracks(table, acq, keep_likelihoods=True)
opts = SpatialOptions(max_classes=3)
pos = pl.DataFrame({"track_id": np.arange(N), "x_um": xy[:, 0], "y_um": xy[:, 1]})
sp = analyze(res, pos, opts)
perm = rng.permutation(N)
ctrl = analyze(res, pos.with_columns(pl.Series("x_um", xy[perm, 0]), pl.Series("y_um", xy[perm, 1])), opts)
print(sp.selection); print("null p", sp.null.p if sp.null else None, "beta", sp.fit.beta)
print(ctrl.selection); print("ctrl n_classes", ctrl.n_classes, ctrl.null.p if ctrl.null else None)

cols = sp.columns()
fits = res.fits.filter(pl.col("status") == "ok")
ED = fits["D_post_mean_um2_s"].to_numpy()
ids = sp.track_ids
own = cols["spatial_class"].to_numpy(); nbr = cols["spatial_class_neighbours"].to_numpy()
# each class stands for the true D nearest its centre, so agreement is read in true-region terms
near = 1 + (np.abs(sp.fit.centre - np.log(0.2)) < np.abs(sp.fit.centre - np.log(0.05)))
truth_cls = np.where(inside[ids], 1, 2)
own0, nbr0 = own, nbr
own, nbr = near[own - 1], near[nbr - 1]
alone = near[sp.fit.evidence.argmax(1)]   # the same classes with beta = 0 (equal weights): own data only
acc_alone, acc_own, acc_nb = (np.mean(c == truth_cls) for c in (alone, own, nbr))
print("accuracy alone", acc_alone)
print("accuracy own+nb", acc_own, "neighbours", acc_nb)

fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
norm = plt.matplotlib.colors.LogNorm(5e-3, 1)
th = np.linspace(0, 2 * np.pi, 200)
ax[0].scatter(*xy[ids].T, c=ED, s=7, cmap="viridis", norm=norm)
ax[0].set_title(f"A. Each track's own E[D] (3-8 frames)\nclass from own data alone: {acc_alone:.0%} right")
sc = None
k_cols = [C["blue"], C["orange"], C["green"]]
cmap = plt.matplotlib.colors.ListedColormap(k_cols[:sp.n_classes])
ax[1].scatter(*sp.xy_um.T, c=own0, s=7, cmap=cmap, vmin=.5, vmax=sp.n_classes + .5)
ax[1].set_title(f"B. spatial_class (own data + neighbours)\nagrees with truth for {acc_own:.0%}")
ax[2].scatter(*sp.xy_um.T, c=nbr0, s=7, cmap=cmap, vmin=.5, vmax=sp.n_classes + .5)
ax[2].set_title(f"C. spatial_class_neighbours (groups())\nagrees with truth for {acc_nb:.0%}")
for a in ax[:3]:
    a.plot(10 + 6 * np.cos(th), 10 + 6 * np.sin(th), color="k", lw=.8, ls="--")
    a.set_aspect("equal"); a.set_xlabel("x (µm)"); a.set_xlim(0, 20); a.set_ylim(0, 20)
ax[0].set_ylabel("y (µm)")
fig.colorbar(plt.cm.ScalarMappable(norm, "viridis"), ax=ax[0], fraction=.046, label="E[D] (µm²/s)")
# D. the shuffle null: neighbour gain for the patterned movie and for the control
nl = sp.null
ax[3].hist(nl.shuffled_gain, bins=15, color=C["grey"], alpha=.6, label="shuffled positions")
ax[3].axvline(nl.gain, color=C["red"], lw=2, label=f"observed: p = {nl.p:.3f}, β = {nl.beta:.2f}")
if ctrl.null is not None:
    ax[3].axvline(ctrl.null.gain, color=C["purple"], lw=2, ls="--",
                  label=f"control movie: p = {ctrl.null.p:.2f}")
ax[3].set_xscale("symlog", linthresh=10); ax[3].set_xlim(left=-1)
ax[3].set_xlabel("neighbour gain (nats, symlog)"); ax[3].set_ylabel("shuffles")
ax[3].set_title(f"D. Is there a pattern? (C = {sp.n_classes})")
ax[3].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.18))
save(fig, "fig_06_spatial.png")
