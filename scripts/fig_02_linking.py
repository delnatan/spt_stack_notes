"""Figure 2.2: spotsolve.link (real; radius read from the movie) against Crocker-Grier's least squared displacement on a simulated
movie that mixes immobile, slow and fast particles, the case the per-track scale is built for.

Crocker-Grier here is one frame-pair assignment minimizing summed squared steps, with ending a track priced at
radius^2 (spotsolve 0.2.0's linker; tests/test_tracking.py keeps it as the reference).
"""
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

import spotsolve
from common import C, plt, save
from sim_02 import CLASSES, SE, movie, rms_step, score

locs = movie()
classes, se = CLASSES, SE
R_rule = 3 * rms_step(3.0)                  # 3x the fast rms step: the rule a given radius follows


def crocker_grier(locs, radius):
    df = locs.with_row_index("i")
    tid = np.arange(df.height)
    by = {t: g for t, g in df.group_by("frame")}
    frames = sorted(k[0] for k in by)
    for t in frames:
        if (t + 1,) not in by:
            continue
        a, b = by[(t,)], by[(t + 1,)]
        pa, pb = a.select("y", "x").to_numpy(), b.select("y", "x").to_numpy()
        d2 = ((pa[:, None] - pb[None]) ** 2).sum(-1)
        big = 1e9
        na, nb = len(pa), len(pb)
        M = np.full((na + nb, nb + na), big)
        M[:na, :nb] = np.where(d2 < radius ** 2, d2, big)
        M[:na, nb:] = np.where(np.eye(na) > 0, radius ** 2, big)
        M[na:, :nb] = np.where(np.eye(nb) > 0, radius ** 2, big)
        M[na:, nb:] = np.where(M[:na, :nb].T < big, 0, big)   # u-track's transposed block
        M[na:, nb:] = 0
        r, c = linear_sum_assignment(M)
        ia, ib = a["i"].to_numpy(), b["i"].to_numpy()
        for i, j in zip(r, c):
            if i < na and j < nb and d2[i, j] < radius ** 2:
                tid[ib[j]] = tid[ia[i]]
    return tid


res = {}
table = locs.select("frame", "y", "x", "se_y", "se_x")
R = spotsolve.search_radius(table)
ss = spotsolve.link(table)                  # the radius read from the movie, the default
res["spotsolve.link, radius read from the movie"] = score(locs, ss["track_id"].to_numpy())
for rad, lab in [(R, "Crocker-Grier, same radius"), (3 * rms_step(0.3), "Crocker-Grier, 3× slow rms")]:
    res[lab] = score(locs, crocker_grier(locs, rad))

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1, 1]), constrained_layout=True)
sub = locs.with_columns(pl.Series("tid", ss["track_id"].to_numpy())).filter(pl.col("frame") < 25)
for (tid,), g in sub.group_by("tid"):
    if g.height < 2:
        continue
    cls = g["cls"][0]
    col = {"immobile": C["grey"], "slow": C["blue"], "fast": C["red"]}[cls]
    ax[0].plot(g["x"], g["y"], "-", color=col, lw=0.8)
ax[0].set_xlim(0, 64); ax[0].set_ylim(0, 64); ax[0].set_aspect("equal")
for n, c in [("immobile", C["grey"]), ("slow", C["blue"]), ("fast", C["red"])]:
    ax[0].plot([], [], color=c, label=n)
ax[0].legend(frameon=False, loc="upper right", fontsize=7)
ax[0].set_title("spotsolve.link tracks (frames 0-24, one quadrant)")
names = list(classes)
x = np.arange(3)
cols = [C["sky"], C["orange"], C["yellow"]]
for k, (lab, r) in enumerate(res.items()):
    ax[1].bar(x + (k - 1) * 0.27, [r[n][0] for n in names], 0.27, color=cols[k], label=lab)
    ax[2].bar(x + (k - 1) * 0.27, [r[n][1] for n in names], 0.27, color=cols[k])
ax[1].set_xticks(x, names); ax[2].set_xticks(x, names)
ax[1].set_ylabel("recall of true links"); ax[2].set_ylabel("identity switches per 100 links")
ax[1].set_title("Recall"); ax[2].set_title("Switches")
fig.legend(*ax[1].get_legend_handles_labels(), frameon=False, fontsize=7.5, loc="outside lower center", ncol=3)
ax[1].set_ylim(0, 1.0)
save(fig, "fig_02_linking.png")
print(f"search radius read from the movie: {R:.1f} px (3x fast rms: {R_rule:.1f} px)")
for k, v in res.items():
    print(k, {n: (round(a, 3), round(b, 2)) for n, (a, b) in v.items()})
