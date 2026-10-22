"""Figure 2.6: what switch_cost trades. spotsolve.link (real, radius read from the movie) on the explainer's
simulated movie at switch costs from 0 to 6: recall and switches per class, and the fast particles' D read from
their linked one-frame steps."""
import polars as pl

import spotsolve
from common import C, plt, save
from sim_02 import CLASSES, SE, links, movie, score

costs = [0, 0.5, 1, 1.5, 2.5, 4, 6]
cols = {"immobile": C["grey"], "slow": C["blue"], "fast": C["red"]}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
for crowd, ls in [(1, "-"), (3, "--")]:
    locs = movie(crowd=crowd)
    table = locs.select("frame", "y", "x", "se_y", "se_x")
    R = spotsolve.search_radius(table)
    res, bias = [], []
    for c in costs:
        tid = spotsolve.link(table, search_radius=R, switch_cost=c)["track_id"].to_numpy()
        res.append(score(locs, tid))
        fast = links(locs, tid).filter(pl.col("cls") == "fast")
        bias.append(100 * ((fast["r2"].mean() / 4 - SE ** 2) / CLASSES["fast"][1] - 1))
    for name, col in cols.items():
        ax[0].plot(costs, [r[name][0] for r in res], ls, marker="o", ms=3, color=col,
                   label=name if crowd == 1 else None)
        ax[1].plot(costs, [r[name][1] for r in res], ls, marker="o", ms=3, color=col)
    ax[2].plot(costs, bias, ls, marker="o", ms=3, color=C["red"], label=f"{locs.height // 60} detections/frame")
    print(crowd, f"R {R:.1f}", [(c, {k: (round(a, 3), round(b, 1)) for k, (a, b) in r.items()}, round(d, 1))
                                for c, r, d in zip(costs, res, bias)])
for a in ax:
    a.axvline(spotsolve.SWITCH_COST, color=C["grey"], lw=0.8, ls=":")
    a.set_xlabel("switch_cost")
ax[0].set_ylabel("recall of true links"); ax[0].set_title("A. Recall (solid: 102 detections/frame, dashed: 3×)")
ax[0].legend(frameon=False, loc="lower left")
ax[1].set_ylabel("identity switches per 100 links"); ax[1].set_title("B. Switches")
ax[2].axhline(0, color=C["black"], lw=0.6)
ax[2].set_ylabel("fast D from linked steps, % bias"); ax[2].set_title("C. Fast particles' D")
ax[2].legend(frameon=False)
save(fig, "fig_02_switch.png")
