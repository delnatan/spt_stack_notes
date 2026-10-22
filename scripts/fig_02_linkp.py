"""Figure 2.5: how well link_p is calibrated. spotsolve.link (real) on the explainer's simulated movie and on the
same box at three times the particles; each link is right when its two detections share a particle."""
import numpy as np
import polars as pl

import spotsolve
from common import C, plt, save
from sim_02 import links, movie

edges = np.array([0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
ax[0].plot([0, 1], [0, 1], color=C["grey"], lw=0.8, ls="--")
for crowd, col in [(1, C["blue"]), (3, C["red"])]:
    locs = movie(crowd=crowd)
    tracks = spotsolve.link(locs.select("frame", "y", "x", "se_y", "se_x"))
    made = links(locs.with_columns(tracks["link_p"]), tracks["track_id"].to_numpy())
    p, right = made["link_p"].to_numpy(), made["right"].to_numpy()
    k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = k == b
        if m.sum() >= 20:
            xs.append(p[m].mean()); ys.append(right[m].mean()); ns.append(m.sum())
    lab = f"{locs.height // 60} detections/frame: mean link_p {p.mean():.3f}, right {right.mean():.3f}"
    ax[0].plot(xs, ys, "o-", color=col, ms=3, label=lab)
    ax[1].hist(p, bins=np.linspace(0, 1, 41), color=col, histtype="step", lw=1.2, label=f"{locs.height // 60}/frame")
    print(crowd, lab, [(round(x, 3), round(y, 3), n) for x, y, n in zip(xs, ys, ns)])
ax[0].set_xlabel("link_p (bin mean)"); ax[0].set_ylabel("share of links right")
ax[0].set_title("A. Calibration"); ax[0].legend(frameon=False, loc="upper left", fontsize=7)
ax[0].set_xlim(0.3, 1.01); ax[0].set_ylim(0.3, 1.01)
ax[1].set_yscale("log"); ax[1].set_xlabel("link_p"); ax[1].set_ylabel("links")
ax[1].set_title("B. Most links are sure"); ax[1].legend(frameon=False, loc="upper left")
save(fig, "fig_02_linkp.png")
