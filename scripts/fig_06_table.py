"""Figure 6.2: the agreement table behind beta's M-step (diffusionkit.spatial.potts.agreement_table).

The movie of figure 6.1 (same seed, same 800 tracks), its symmetric kNN graph, and the prior's expected fraction of
agreeing edges against beta for C = 2 and 3. The dots are the C = 2 and C = 3 fits' (beta, agreement).
"""
import numpy as np
import polars as pl

from diffusionkit import Acquisition
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import analyze_tracks
from diffusionkit.spatial import SpatialOptions, analyze, potts
from common import C, plt, save

acq = Acquisition(dt_s=0.035, exposure_s=0.020)
rng = np.random.default_rng(11)  # as fig_06_spatial.py
N = 800
xy = rng.uniform(0, 20, (N, 2))
inside = np.hypot(*(xy - 10).T) < 6
Dtrue = np.where(inside, 0.05, 0.2)
parts = []
for i in range(N):
    n = int(rng.integers(3, 9))
    t = simulate_tracks(1, n, 1.0, Dtrue[i], 0.03, acq, rng=rng, n_sub=10, first_track_id=i)
    parts.append(t.with_columns(pl.col("x_um") + xy[i, 0], pl.col("y_um") + xy[i, 1]))
res = analyze_tracks(pl.concat(parts), acq, keep_likelihoods=True)
pos = pl.DataFrame({"track_id": np.arange(N), "x_um": xy[:, 0], "y_um": xy[:, 1]})
sp = analyze(res, pos, SpatialOptions(max_classes=3, n_shuffles=0))

A = sp.adjacency
true_agree = np.mean([inside[i] == inside[j] for i, j in zip(*potts.triu(A, 1).nonzero())])
fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
for c, col in [(2, C["blue"]), (3, C["orange"])]:
    t = potts.agreement_table(A, c)
    ax.plot(potts.BETA_GRID, t, color=col, label=f"C = {c}: prior, no data")
    f = sp.fits[c]
    ax.plot(f.beta, f.agreement, "o", color=col, ms=7, label=f"C = {c} fit: β = {f.beta:.2f}")
ax.axhline(true_agree, color=C["grey"], ls=":", lw=1, label=f"true regions: {true_agree:.2f} of edges agree")
ax.set_xlabel("β (per edge)"); ax.set_ylabel("expected fraction of edges that agree")
ax.set_xlim(0, 1.5); ax.set_ylim(.3, 1.02)
ax.legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "fig_06_table.png")
print("mean degree", A.sum() / N, "true agreement", true_agree,
      {c: (sp.fits[c].beta, sp.fits[c].agreement) for c in (2, 3)})
