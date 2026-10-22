"""Figure 2.2: spotsolve.link (real, v0.6.0) against Crocker-Grier's least squared displacement on a simulated
movie that mixes immobile, slow and fast particles, the case the per-track scale is built for.

Crocker-Grier here is one frame-pair assignment minimizing summed squared steps, with ending a track priced at
radius^2 (spotsolve 0.2.0's linker; tests/test_tracking.py keeps it as the reference).
"""
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

import spotsolve
from common import C, plt, save

rng = np.random.default_rng(11)
L, T, se = 128.0, 60, 0.3
classes = {"immobile": (40, 0.0), "slow": (60, 0.3), "fast": (30, 3.0)}   # (count, D px²/frame)
rows = []
pid = 0
for name, (n, D) in classes.items():
    for _ in range(n):
        p = rng.uniform(5, L - 5, 2)
        pdet = rng.uniform(0.6, 0.99)
        for t in range(T):
            if t:
                p = p + rng.normal(0, np.sqrt(2 * D), 2)
                p = np.abs(p); p = L - np.abs(L - p)      # reflect at the edges
            if rng.random() < pdet:
                o = p + rng.normal(0, se, 2)
                rows.append((t, o[0], o[1], se, se, pid, name))
        pid += 1
locs = pl.DataFrame(rows, schema=["frame", "y", "x", "se_y", "se_x", "particle", "cls"], orient="row")
fast_rms = np.sqrt(4 * 3.0 + 4 * se ** 2)
R = 3 * fast_rms


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


def score(df, tid):
    df = df.with_columns(pl.Series("tid", tid)).sort("particle", "frame")
    out = {}
    nxt = df.with_columns(pl.col("frame").shift(-1).over("particle").alias("f2"),
                          pl.col("tid").shift(-1).over("particle").alias("t2"))
    true_links = nxt.filter(pl.col("f2") == pl.col("frame") + 1)
    links = df.sort("tid", "frame").with_columns(pl.col("frame").shift(-1).over("tid").alias("f2"),
                                                pl.col("particle").shift(-1).over("tid").alias("p2"))
    links = links.filter(pl.col("f2") == pl.col("frame") + 1)
    for name in classes:
        tl = true_links.filter(pl.col("cls") == name)
        rec = (tl["tid"] == tl["t2"]).mean()
        lk = links.filter(pl.col("cls") == name)
        sw = 100 * (lk["particle"] != lk["p2"]).mean()
        out[name] = (rec, sw)
    return out


res = {}
ss = spotsolve.link(locs.select("frame", "y", "x", "se_y", "se_x"), max_step=R)
res["spotsolve.link, max_step = 3× fast rms"] = score(locs, ss["track_id"].to_numpy())
for rad, lab in [(R, "Crocker-Grier, same radius"), (3 * np.sqrt(4 * 0.3 + 4 * se ** 2), "Crocker-Grier, 3× slow rms")]:
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
ax[1].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(1.1, -0.1), ncol=3)
ax[1].set_ylim(0, 1.0)
save(fig, "fig_02_linking.png")
for k, v in res.items():
    print(k, {n: (round(a, 3), round(b, 2)) for n, (a, b) in v.items()})
