"""Figure 2.4: the search radius read from the movie, and why it can be soft.

A: next-frame and own-frame neighbour counts around each detection (numpy, the counting step of
track.rs's `search_radius`), their difference, and what the truth predicts for it. B: spotsolve.link (real) at
given radii from half to four times the rule (3x the fast rms step), with the radius it reads marked.
"""
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

import spotsolve
from common import C, plt, save
from sim_02 import CLASSES, SE, movie, rms_step, score

locs = movie()
table = locs.select("frame", "y", "x", "se_y", "se_x")
R_read = spotsolve.search_radius(table)
R_rule = 3 * rms_step(3.0)

# A. Mean count of detections within r: in the next frame, and in the detection's own frame (itself excluded).
r = np.linspace(0, 30, 61)[1:]
by = {t: g.select("y", "x").to_numpy() for (t,), g in table.group_by("frame")}
n1, n0, n = np.zeros_like(r), np.zeros_like(r), 0
for t, a in by.items():
    if t + 1 not in by:
        continue
    ta, tb = cKDTree(a), cKDTree(by[t + 1])
    for k, rk in enumerate(r):
        n1[k] += sum(len(x) for x in ta.query_ball_tree(tb, rk))
        n0[k] += sum(len(x) - 1 for x in ta.query_ball_tree(ta, rk))
    n += len(a)
n1, n0 = n1 / n, n0 / n
# The truth: a detection continues with its particle's visibility, and its step is 2-D Gaussian.
share = {c: (locs["cls"] == c).mean() for c in CLASSES}
q = locs.sort("particle", "frame").with_columns(
    (pl.col("frame").shift(-1).over("particle") == pl.col("frame") + 1).alias("cont"))
q = q.filter(pl.col("frame") < pl.col("frame").max())
qc = {c: q.filter(pl.col("cls") == c)["cont"].mean() for c in CLASSES}
truth = sum(share[c] * qc[c] * (1 - np.exp(-r ** 2 / rms_step(D) ** 2)) for c, (_, D) in CLASSES.items())

# B. Links at given radii.
mult = np.array([0.5, 0.75, 1, 1.5, 2, 3, 4])
runs = [score(locs, spotsolve.link(table, search_radius=m * R_rule)["track_id"].to_numpy()) for m in mult]
read = score(locs, spotsolve.link(table)["track_id"].to_numpy())

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
ax[0].plot(r, n1, color=C["blue"], label="next frame")
ax[0].plot(r, n0, color=C["grey"], label="own frame (everyone else)")
ax[0].set_xlabel("r (px)"); ax[0].set_ylabel("detections within r, per detection")
ax[0].set_title("A. Neighbours of a detection")
ax[0].legend(frameon=False)
ax[1].plot(r, n1 - n0, color=C["black"], label="next − own: its own continuation")
ax[1].plot(r, truth, "--", color=C["orange"], label="truth: Σ share · q · P(step < r)")
for x, lab, col, ha in [(R_read, "read", C["green"], "left"), (R_rule, "3× fast rms", C["red"], "right")]:
    ax[1].axvline(x, color=col, lw=0.8, ls=":")
    ax[1].text(x, 0.45, f" {lab} \n {x:.1f} px ", color=col, fontsize=7.5, ha=ha)
ax[1].set_ylim(0, 0.9)
ax[1].set_xlabel("r (px)"); ax[1].set_ylabel("excess count")
ax[1].set_title("B. The excess is the step distribution")
ax[1].legend(frameon=False, loc="lower right")
for name, col in [("immobile", C["grey"]), ("slow", C["blue"]), ("fast", C["red"])]:
    ax[2].plot(mult, [x[name][0] for x in runs], "o-", color=col, ms=3, label=name)
    ax[2].plot(R_read / R_rule, read[name][0], "*", color=col, ms=9)
ax[2].set_xscale("log"); ax[2].set_xticks(mult, [f"{m:g}" for m in mult]); ax[2].minorticks_off()
ax[2].set_xlabel("given radius ÷ 3× fast rms step (★: read from the movie)")
ax[2].set_ylabel("recall of true links"); ax[2].set_ylim(0, 1.02)
ax[2].set_title("C. Recall against the radius")
ax[2].legend(frameon=False, loc="lower right")
save(fig, "fig_02_radius.png")
print(f"read {R_read:.2f} px, rule {R_rule:.2f} px, ratio {R_read / R_rule:.2f}")
for m, x in zip(mult, runs):
    print(m, {k: (round(a, 3), round(b, 2)) for k, (a, b) in x.items()})
print("read", {k: (round(a, 3), round(b, 2)) for k, (a, b) in read.items()})
