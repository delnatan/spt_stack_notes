"""Figure 1.4: single fits vs mixtures on crowded fields (real `spotsolve.localize_stack`).

Recall and precision match reports to truth within 1 px by an optimal assignment, as the benchmark does.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import spotsolve
from spotsolve.simulate import simulate
from common import C, plt, save

sigma = 1.45


def score(found, truth, r=1.0):
    if not len(found) or not len(truth):
        return 0, len(found), len(truth)
    d = cdist(found, truth)
    i, j = linear_sum_assignment(np.where(d < r, d, 1e6))
    tp = int((d[i, j] < r).sum())
    return tp, len(found), len(truth)


rows = []
crop = None
for dens in (0.005, 0.02, 0.04):
    sims = [simulate((96, 96), density=dens, sigma=sigma, background=20, amplitude_range=(200, 1500), seed=s)
            for s in range(4)]
    stack = np.array([s.image for s in sims])
    for mix in (False, True):
        res = spotsolve.localize_stack(stack, sigma, gain=1.0, fit_mixtures=mix)
        tp = fp = nt = 0
        for r, s in zip(res, sims):
            a, b, c = score(r.positions, s.positions)
            tp, fp, nt = tp + a, fp + b, nt + c
        rows.append((dens, mix, tp / nt, tp / max(fp, 1)))
        if dens == 0.02:
            crop = crop or {}
            crop[mix] = res[0].positions
            crop["sim"] = sims[0]

fig = plt.figure(figsize=(11, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
s = crop["sim"]
sl = (slice(20, 70), slice(20, 70))
for k, (mix, title) in enumerate([(False, "Single fits"), (True, "Mixtures (fit_mixtures=True)")]):
    a = fig.add_subplot(gs[k])
    a.imshow(s.image[sl], cmap="gray")
    t = s.positions - [20, 20]
    f = crop[mix] - [20, 20]
    keep = lambda p: (p[:, 0] > -0.5) & (p[:, 0] < 49.5) & (p[:, 1] > -0.5) & (p[:, 1] < 49.5)
    a.plot(t[keep(t), 1], t[keep(t), 0], "o", mfc="none", mec=C["green"], ms=8, mew=1.1, label="truth")
    a.plot(f[keep(f), 1], f[keep(f), 0], "+", color=C["orange"] if not mix else C["sky"], ms=8, mew=1.4, label="reported")
    a.set_title(f"{title}, density 0.02 /px²")
    a.set_xticks([]); a.set_yticks([])
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
a = fig.add_subplot(gs[2])
dens = sorted({r[0] for r in rows})
x = np.arange(len(dens))
for off, mix, c in [(-0.2, False, C["orange"]), (0.2, True, C["sky"])]:
    rec = [r[2] for r in rows if r[1] == mix]
    pre = [r[3] for r in rows if r[1] == mix]
    a.bar(x + off, rec, 0.38, color=c, label=("mixtures" if mix else "single fits") + ": recall")
    a.plot(x + off, pre, "kD", ms=4)
a.plot([], [], "kD", ms=4, label="precision")
a.set_xticks(x, [f"{d:g}" for d in dens])
a.set_xlabel("emitter density (/px²), σ = 1.45 px")
a.set_ylim(0, 1.05)
a.set_title("Recall and precision (4 frames each)")
a.legend(frameon=False, loc="lower left")
save(fig, "fig_01_mixtures.png")
for r in rows:
    print(r)
