"""Slides: what the hidden Potts model in diffusionkit.spatial does, on figure 6.1's toy cell.

The same 800 simulated tracks as fig_06_spatial.py (3-8 frames; D = 0.05 um^2/s inside a 6 um disc, 0.2 outside),
fitted with two classes. Seven 16:9 figures, in presentation order, written to ../figures:

  potts_1_own_steps.png     what classifying a track by its own steps means: its likelihood curve over D against
                            the two class curves, and Bayes' rule
  potts_2_em_loop.png       the model in one line (clustering x spatial coupling) and the EM loop
  potts_3_mixture_em.png    EM without neighbours (beta = 0): the classes find the two D values, the map stays noisy
  potts_4_vote.png          one track: its own evidence, its neighbours' vote, and the two combined
  potts_5_beta.png          what beta means: samples of the Potts prior (no data) on this graph, and the agreement curve
                            the fit reads beta from
  potts_6_em_neighbours.png EM with neighbours, step by step: assign, refit the classes, update beta
  potts_7_shuffle.png       the control: the same tracks with positions shuffled get beta ~ 0 and no pattern

Uses gridpost.analyze_tracks and the pieces of spatial.potts (graph, agreement_table, beta_for, fit). The EM loop in
`em_snapshots` is potts.fit's loop with snapshots kept.
"""
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components

from diffusionkit import Acquisition
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import analyze_tracks
from diffusionkit.spatial import potts
from common import C, plt, save

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12})
SLOW, FAST = C["blue"], C["orange"]
DARK = "0.25"
PSLOW = LinearSegmentedColormap.from_list("pslow", [FAST, "#e6e6e6", SLOW])
W, H = 11, 6.2
TH = np.linspace(0, 2 * np.pi, 200)

# ---- figure 6.1's toy cell (same seed and order of draws as fig_06_spatial.py) ----
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
table_xy = pl.concat(parts)
res = analyze_tracks(table_xy, acq, keep_likelihoods=True)
L = res.likelihoods
ids = np.asarray(L.track_ids)
ll = np.asarray(L.loglik_D, float)
lik = np.exp(ll - ll.max(axis=1, keepdims=True))
u = res.options.u_D()
xy, inside = xy[ids], inside[ids]
A = potts.graph(xy, 8)
table = potts.agreement_table(A, 2)
E = triu(A, 1).tocoo()
edges = np.stack([xy[E.row], xy[E.col]], axis=1)


def em_snapshots(lik, start=None, learn=True, keep=(0, 1, 2, 3, 5, 10)):
    """potts.fit's loop for two classes, with snapshots of (q, centre, sd, beta, agreement) after each kept step.

    Step 0 is the E-step on the starting classes. By default the start is the plain mixture (beta = 0, converged),
    one of the starts the package uses, and beta is learned from 0."""
    if start is None:
        plain = potts.fit(lik, u, A, 2, beta=0.)
        start = (plain.centre, plain.sd)
    centre, sd = (np.asarray(v, float).copy() for v in start)
    b = 0.
    q = np.full((len(lik), 2), .5)
    snaps = {}
    for it in range(0, 301):
        q_old, b_old = q, b
        if it > 0:                                     # M-step: classes from the current assignments, then beta
            mass = (lik.T @ (q / ev)) * p
            total = mass.sum(axis=0) + potts.TINY
            centre = u @ mass / total
            sd = np.sqrt(np.maximum((u ** 2) @ mass / total - centre ** 2, 0.) + potts.MIN_SD ** 2)
            if learn:
                b = potts.beta_for(potts.edge_agreement(q, A), table)
        p = potts.densities(u, centre, sd)             # E-step: own evidence x neighbours' vote
        ev = lik @ p + potts.TINY
        for _ in range(potts.SWEEPS):
            q = potts._softmax(np.log(ev) + b * (A @ q))
        snap = (q, centre.copy(), sd.copy(), b, potts.edge_agreement(q, A))
        if it in keep:
            snaps[it] = snap
        if it > 10 and np.abs(q - q_old).max() < 1e-3 and abs(b - b_old) < 1e-3:
            break
    snaps[it] = snap
    return snaps, it


snaps, n_iter = em_snapshots(lik)
q, centre, sd, beta, _ = snaps[n_iter]
assert centre[0] < centre[1]                      # class 0 slow, class 1 fast
ev = lik @ potts.densities(u, centre, sd)
own = ev[:, 0] / ev.sum(1)                        # P(slow) from a track's own steps (equal weights, beta = 0)
nbr_counts = A @ q                                # expected neighbours in each class
print("beta", beta, "D", np.exp(centre), "iterations", n_iter)
print("accuracy own", np.mean((own > .5) == inside), "fit", np.mean((q[:, 0] > .5) == inside))


def field(ax, title=None, boundary=True):
    ax.set_aspect("equal")
    ax.set_xlim(0, 20); ax.set_ylim(0, 20)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if boundary:
        ax.plot(10 + 6 * np.cos(TH), 10 + 6 * np.sin(TH), color="k", lw=1, ls="--")
    if title:
        ax.set_title(title, loc="left")


def dots(ax, p_slow, s=14, **kw):
    return ax.scatter(*xy.T, c=p_slow, cmap=PSLOW, vmin=0, vmax=1, s=s, lw=0, **kw)


def colorbar(fig, rect, label="P(slow class)"):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=PSLOW), cax=cax, orientation="horizontal")
    cb.set_ticks([0, .5, 1], labels=["fast", "50/50", "slow"])
    cb.set_label(label, fontsize=12)
    cb.outline.set_visible(False)


x = u / np.log(10)                                # log10 D on the grid
peak = x[lik.argmax(1)]                           # each track's most likely log10 D
BINS = np.arange(-3, .51, .1)
DTRUE = np.log10([.05, .2])


def class_curves(ax, ci, si, weight, lw=2.5):
    """The two classes as curves in log10 D, scaled to the histogram of `weight`-weighted tracks."""
    dens = potts.densities(u, ci, si) / np.gradient(x)[:, None]
    for c, col in enumerate([SLOW, FAST]):
        ax.plot(x, dens[:, c] * weight[c] * .1, color=col, lw=lw)


def class_hist(ax, qi):
    """Tracks at their most likely D, each split between the classes by its P(class)."""
    ax.hist([peak, peak], bins=BINS, weights=[qi[:, 0], qi[:, 1]], stacked=True, color=[SLOW, FAST], alpha=.35)


def d_axis(ax, label=True):
    ax.set_xlim(-3, .5); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if label:
        ax.set_xlabel("log₁₀ D (µm²/s)", fontsize=12)


# ---- 1. what "classifying by its own steps" means ----
ev_c = lik @ potts.densities(u, centre, sd)
own_c = ev_c[:, 0] / ev_c.sum(1)
nf = np.asarray(L.n_frames)
far = np.abs(np.hypot(*(xy - 10).T) - 6) > 1.5
picks = [np.where(inside & far & (nf == 8) & (own_c > .9))[0],
         np.where(~inside & far & (nf == 8) & (own_c < .1))[0],
         np.where(far & (nf <= 4) & (np.abs(own_c - .5) < .1))[0]]
picks = [c[np.argmin(np.abs(peak[c] - t))] for c, t in zip(picks, [DTRUE[0], DTRUE[1], peak[picks[2]].mean()])]
labels = ["Clearly slow", "Clearly fast", "Can't tell"]
fig = plt.figure(figsize=(W, H))
fig.suptitle("Classifying a track by its own steps: Bayes' rule with two classes", x=.03, ha="left", fontsize=18)
fig.text(.03, .885, "A class is a range of D (a bell curve in log D). A track's steps give a likelihood curve: "
         "how well each D explains them.", fontsize=12, color="0.4")
traj = {i: table_xy.filter(pl.col("track_id") == ids[i]).select("x_um", "y_um").to_numpy() for i in picks}
half = max(np.ptp(v - v[0], axis=0).max() for v in traj.values()) * .6
dens = potts.densities(u, centre, sd) / np.gradient(x)[:, None]
for r, (i, lab) in enumerate(zip(picks, labels)):
    y0 = .6 - r * .245
    t = fig.add_axes([.03, y0, .12, .2])
    v = traj[i] - traj[i][0]
    m = (v.max(0) + v.min(0)) / 2
    t.plot(*v.T, color=DARK, lw=1.5, marker="o", ms=4)
    t.plot(*v[0], "o", color="k", ms=6)
    t.set_xlim(m[0] - half, m[0] + half); t.set_ylim(m[1] - half, m[1] + half)
    t.set_aspect("equal"); t.axis("off")
    t.text(-.05, .5, lab, transform=t.transAxes, ha="right", va="center", fontsize=14, rotation=90)
    t.text(.5, -.02, f"{nf[i]} positions", transform=t.transAxes, ha="center", va="top", fontsize=11,
           color="0.35")
    a = fig.add_axes([.2, y0 + .01, .46, .19])
    li = lik[i] / lik[i].max()
    a.fill_between(x, 0, li, color="0.85", lw=0)
    a.plot(x, li, color="k", lw=1.5)
    prod = lik[i][:, None] * dens
    prod = prod / prod.max() * .75
    for c, col in enumerate([SLOW, FAST]):
        a.plot(x, dens[:, c] / dens.max(), color=col, lw=2)
        a.fill_between(x, 0, prod[:, c], color=col, alpha=.45, lw=0)
    a.set_ylim(0, 1.08)
    d_axis(a, label=(r == 2))
    if r < 2:
        a.set_xticklabels([])
    pb = fig.add_axes([.71, y0 + .06, .26, .08])
    pb.barh(0, own_c[i], color=SLOW); pb.barh(0, 1 - own_c[i], left=own_c[i], color=FAST)
    pb.set_xlim(0, 1); pb.axis("off")
    pb.text(own_c[i] / 2, 0, f"{own_c[i]:.0%}", ha="center", va="center", color="w", fontsize=12) \
        if own_c[i] > .12 else None
    pb.text((1 + own_c[i]) / 2, 0, f"{1 - own_c[i]:.0%}", ha="center", va="center", color="w", fontsize=12) \
        if own_c[i] < .88 else None
    if r == 0:
        fig.text(.71, .79, "P(slow) = slow area / (slow + fast area)", fontsize=13)
        fig.text(.71, .755, "Bayes' rule with equal prior odds", fontsize=11, color="0.35")
    if r == 2:
        a.text(-2.95, .9, "A short track's broad\ncurve overlaps both\nclasses: it splits its vote", fontsize=11.5,
               va="top", linespacing=1.3)
fig.text(.2, .855, "grey: this track's likelihood     curves: the two classes     shaded: likelihood × class "
         "(its area is the evidence)", fontsize=11, color="0.35")
save(fig, "potts_1_own_steps.png")

# ---- 2. the whole model: clustering plus a spatial coupling term ----
fig = plt.figure(figsize=(W, H))
fig.suptitle("The model: clustering in D, plus a spatial coupling term", x=.03, ha="left", fontsize=18)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
box = plt.matplotlib.patches.FancyBboxPatch
ax.text(.04, .7, "P(track in class)  ∝", fontsize=22, va="center")
for x0, w, col, term, head, note in [
        (.33, .25, "#e8eef6", "overlap of its likelihood\nwith the class curve", "clustering",
         "own steps vs the class (slide 1)"),
        (.65, .31, "#fbefd9", "exp(β × neighbours\nin that class)", "spatial coupling",
         "neighbours' P(class), strength β")]:
    ax.add_patch(box((x0, .6), w, .2, boxstyle="round,pad=.01", fc=col, ec="0.6"))
    ax.text(x0 + w / 2, .7, term, fontsize=16, ha="center", va="center", linespacing=1.3)
    ax.text(x0 + w / 2, .55, head, fontsize=15, ha="center", va="top", weight="bold")
    ax.text(x0 + w / 2, .505, note, fontsize=12, ha="center", va="top", color="0.35")
ax.text(.615, .7, "×", fontsize=26, ha="center", va="center")
ax.text(.04, .43, "Fitted by EM, repeating three steps until nothing changes:", fontsize=14)
steps = [("1  Assign", "P(class) for every track\nfrom the line above"),
         ("2  Refit the classes", "centre and width in log D,\nfrom tracks weighted by P(class)"),
         ("3  Update β", "so the prior expects as many\nagreeing neighbours as step 1 shows")]
for j, (head, body) in enumerate(steps):
    x0 = .04 + j * .32
    ax.add_patch(box((x0, .16), .28, .2, boxstyle="round,pad=.01", fc="#f3f3f3", ec="0.6"))
    ax.text(x0 + .015, .34, head, fontsize=14, weight="bold", va="top")
    ax.text(x0 + .015, .275, body, fontsize=12, va="top", linespacing=1.35)
    if j < 2:
        ax.annotate("", (x0 + .315, .26), (x0 + .285, .26), arrowprops=dict(arrowstyle="-|>", lw=1.6))
ax.annotate("", (.18, .15), (.82, .15), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="0.4",
                                                         connectionstyle="arc3,rad=-.06"))
ax.text(.5, .035, "β starts at 0: plain clustering (a Gaussian mixture in log D, a soft k-means). β is learned, "
        "so it stays near 0 when there is no pattern.", fontsize=12, ha="center", color="0.3")
save(fig, "potts_2_em_loop.png")

# ---- 3. EM without neighbours: a mixture model ----
init = potts.initial(lik, u, 2)
msnaps, m_iter = em_snapshots(lik, start=init, learn=False, keep=(0, 1, 2, 5))
show = [0, 1, 2, 5, m_iter]
fig = plt.figure(figsize=(W, H))
fig.suptitle("Clustering alone (β = 0): two classes, but a noisy map", x=.03, ha="left",
             fontsize=18)
fig.text(.03, .885, "Each step: assign every track by its own evidence, then refit the classes from the "
         "assignments", fontsize=12, color="0.4")
for i, it in enumerate(show):
    qi, ci, si, _, _ = msnaps[it]
    a = fig.add_axes([.03 + i * .195, .5, .175, .3])
    class_hist(a, qi)
    class_curves(a, ci, si, qi.sum(0))
    for v, col in zip(DTRUE, [SLOW, FAST]):
        a.axvline(v, color=col, ls=":", lw=1)
    d_axis(a, label=False)
    head = "Initial guess" if it == 0 else (f"Converged (step {it})" if it == m_iter else f"Step {it}")
    a.set_title(head, loc="left", fontsize=14)
    m = fig.add_axes([.03 + i * .195, .08, .175, .3])
    field(m)
    dots(m, qi[:, 0], s=5)
    m.text(10, -.5, f"{np.mean((qi[:, 0] > .5) == inside):.0%} right", ha="center", va="top", fontsize=12)
fig.text(.03, .43, "log₁₀ D (µm²/s).  Tracks at their most likely D, each split between the classes by "
         "P(class); curves: the classes; dotted: true D", fontsize=11, color="0.4")
colorbar(fig, [.79, .925, .18, .018], label=None)
save(fig, "potts_3_mixture_em.png")


# ---- 4. one track asks its neighbours ----
# A track in the slow disc whose own steps point the wrong way, and whose neighbours put it right.
cand = np.where(inside & (own < .3) & (q[:, 0] > .9) & (np.hypot(*(xy - 10).T) < 4.5))[0]
k = cand[np.argmin(own[cand])]
nb = A[k].indices
r = 2.3
fig = plt.figure(figsize=(W, H))
fig.suptitle("The coupling for one track: own evidence × neighbours' vote", x=.03, ha="left",
             fontsize=18)
ax = fig.add_axes([.02, .17, .3, .68])
field(ax, "Own steps only")
ax.add_collection(LineCollection(edges, color="0.6", lw=.3, alpha=.5, zorder=0))
dots(ax, own, s=10)
ax.add_patch(Rectangle(xy[k] - r, 2 * r, 2 * r, fill=False, lw=1.5, color="k"))
colorbar(fig, [.05, .12, .24, .025])
ax.text(0, -.9, f"8 nearest neighbours each, linked both ways", fontsize=11, color="0.4", va="top")

az = fig.add_axes([.33, .17, .3, .68])
field(az, "Its neighbours (after the fit)")
az.set_xlim(xy[k, 0] - r, xy[k, 0] + r); az.set_ylim(xy[k, 1] - r, xy[k, 1] + r)
for s in az.spines.values():
    s.set_visible(True)
az.add_collection(LineCollection(edges, color="0.75", lw=.6, zorder=0))
az.add_collection(LineCollection([[xy[k], xy[j]] for j in nb], color="k", lw=1.6, zorder=1))
dots(az, q[:, 0], s=70, zorder=2)
az.scatter(*xy[k], c=[own[k]], cmap=PSLOW, vmin=0, vmax=1, s=260, edgecolors="k", lw=2, zorder=3)
az.annotate("this track", xy[k], xytext=(-95, 28), textcoords="offset points", fontsize=13, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5),
            arrowprops=dict(arrowstyle="-", color="k", lw=1, shrinkB=9))

bx = fig.add_axes([.67, .17, .31, .68])
bx.set_xlim(0, 1); bx.set_ylim(-.3, 3.3); bx.axis("off")
n_s, n_f = nbr_counts[k]
vote = np.exp(beta * nbr_counts[k]); vote /= vote.sum()
rows = [("Its own steps", own[k], f"from its {int(L.n_frames[k])} positions alone"),
        (f"Neighbours' vote", vote[0], f"{len(nb)} neighbours ≈ {n_s:.1f} slow, {n_f:.1f} fast\n"
                                       f"weight ∝ exp(β × count), β = {beta:.2f}"),
        ("Together", q[k, 0], "own steps × neighbours' vote")]
for i, (name, p, note) in enumerate(rows):
    y = 2.6 - 1.2 * i
    bx.barh(y, p, height=.34, color=SLOW, left=0)
    bx.barh(y, 1 - p, height=.34, color=FAST, left=p)
    bx.text(0, y + .27, name, fontsize=15, va="bottom", weight="bold" if i == 2 else None)
    bx.text(0, y - .25, note, fontsize=11, va="top", color="0.35")
    bx.text(p / 2, y, f"slow {p:.0%}", ha="center", va="center", color="w", fontsize=12) if p > .2 else None
    bx.text(p + (1 - p) / 2, y, f"fast {1 - p:.0%}", ha="center", va="center", color="w", fontsize=12) \
        if p < .8 else None
    if i < 2:
        bx.text(1.04, y - .6, "×" if i == 0 else "=", fontsize=22, ha="center", va="center")
save(fig, "potts_4_vote.png")


# ---- 5. beta: how strongly neighbours agree ----
def potts_sample(beta, n_class=2, sweeps=300, seed=1):
    """A draw from the Potts prior on this graph (no data), by Swendsen-Wang as in potts.agreement_table."""
    g = np.random.default_rng(seed)
    z = g.integers(n_class, size=N)
    keep = 1 - np.exp(-beta)
    for _ in range(sweeps):
        bond = (z[E.row] == z[E.col]) & (g.random(len(E.row)) < keep)
        bonds = csr_matrix((np.ones(int(bond.sum())), (E.row[bond], E.col[bond])), shape=(N, N))
        nc, cl = connected_components(bonds, directed=False)
        z = g.integers(n_class, size=nc)[cl]
    return z


betas = [0., .3, round(beta, 2), .8]
names = ["β = 0: neighbours ignored", "β = 0.3: patches", f"β = {betas[2]:.2f}: this cell's fit",
         "β = 0.8: one class"]
fig = plt.figure(figsize=(W, H))
fig.suptitle("β sets how much neighbours agree; the fit picks it from the data", x=.03, ha="left", fontsize=18)
fig.text(.03, .88, "Draws from the Potts prior on this cell's neighbour graph, with no data", fontsize=12,
         color="0.4")
for i, (b, name) in enumerate(zip(betas, names)):
    a = fig.add_axes([.02 + (i % 2) * .26, .48 - (i // 2) * .42, .24, .32])
    z = potts_sample(b)
    field(a, boundary=False)
    a.scatter(*xy.T, c=np.where(z == 0, SLOW, FAST), s=7, lw=0)
    agree = np.mean(z[E.row] == z[E.col])
    a.set_title(name, fontsize=13, loc="left", weight="bold" if i == 2 else None)
    a.text(10, -.6, f"{agree:.0%} of neighbour pairs agree", ha="center", va="top", fontsize=11, color="0.35")
ac = fig.add_axes([.6, .13, .37, .64])
ac.plot(potts.BETA_GRID, table, color="k", lw=2)
fit_agree = potts.edge_agreement(q, A)
ac.axhline(fit_agree, color=SLOW, ls="--", lw=1.2)
ac.plot([beta, beta], [0, fit_agree], color=SLOW, ls="--", lw=1.2)
ac.plot(beta, fit_agree, "o", color=SLOW, ms=10, zorder=5)
for b in betas:
    ac.plot(b, np.interp(b, potts.BETA_GRID, table), "o", mfc="w", mec="k", ms=7, zorder=4)
ac.text(1.48, fit_agree - .015, f"agreement the fit sees: {fit_agree:.0%}", ha="right", va="top", color=SLOW,
        fontsize=12)
ac.text(beta + .03, .52, f"β = {beta:.2f}", color=SLOW, fontsize=13)
ac.text(1.48, .58, "Near the fitted β the prior alone\nwould put most tracks in one class.\nThe tracks' own "
        "steps decide\nwhere each class actually sits.", ha="right", va="bottom", fontsize=12, color="0.25",
        linespacing=1.4)
ac.set_xlim(0, 1.5); ac.set_ylim(.48, 1.01)
ac.set_xlabel("β (per neighbour pair)")
ac.set_ylabel("neighbour pairs that agree (prior)")
ac.set_title("β is where the prior expects as much agreement\nas the fitted classes show", loc="left")
save(fig, "potts_5_beta.png")

# ---- 6. EM with neighbours, step by step ----
show = [0, 1, 3, n_iter]
fig = plt.figure(figsize=(W, H))
fig.suptitle("Add the spatial coupling: β rises as neighbours agree", x=.03, ha="left", fontsize=18)
row_lab = [(.66, "1  Assign"), (.34, "2  Refit\n   classes"), (.13, "3  Update β")]
for y, lab in row_lab:
    fig.text(.015, y, lab, fontsize=14, weight="bold", va="center")
for i, it in enumerate(show):
    qi, ci, si, bi, ag = snaps[it]
    x0 = .13 + i * .218
    a = fig.add_axes([x0, .5, .2, .33])
    field(a)
    dots(a, qi[:, 0], s=5)
    head = "Start (slide 3's fit)" if it == 0 else (f"Converged (step {it})" if it == n_iter else f"Step {it}")
    a.set_title(head, loc="left", fontsize=14)
    a.text(10, -.4, f"{np.mean((qi[:, 0] > .5) == inside):.0%} right", ha="center", va="top", fontsize=12)
    d = fig.add_axes([x0, .29, .2, .13])
    class_hist(d, qi)
    class_curves(d, ci, si, qi.sum(0), lw=2)
    for v, col in zip(DTRUE, [SLOW, FAST]):
        d.axvline(v, color=col, ls=":", lw=1)
    d_axis(d, label=False)
    d.set_xticks([-3, -2, -1, 0]); d.tick_params(labelsize=10)
    g = fig.add_axes([x0, .06, .2, .14])
    g.plot(potts.BETA_GRID, table, color="0.6", lw=1.5)
    b_new = potts.beta_for(ag, table)              # the next M-step's beta
    g.plot(b_new, ag, "o", color=SLOW, ms=7, zorder=3)
    g.plot([0, b_new], [ag, ag], color=SLOW, ls="--", lw=1)
    g.set_xlim(0, 1); g.set_ylim(.48, 1.02)
    g.set_yticks([.5, 1], labels=["50%", "100%"] if i == 0 else []); g.set_xticks([0, .4, .8])
    g.tick_params(labelsize=10)
    g.text(.98, .55, f"{ag:.0%} agree → β = {b_new:.2f}", ha="right", fontsize=11, color=SLOW)
fig.text(.13, .435, "Tracks at their most likely D split by P(class); curves: the refitted classes", fontsize=11,
         color="0.4")
fig.text(.13, .225, "Agreement of these assignments, read off the prior's curve to give β for the next "
         "assignment", fontsize=11, color="0.4")
colorbar(fig, [.79, .925, .18, .018], label=None)
save(fig, "potts_6_em_neighbours.png")

# ---- 7. control: shuffled positions ----
perm = np.random.default_rng(5).permutation(N)
ctrl = potts.fit(lik[perm], u, A, 2, table, init=(snaps[0][1], snaps[0][2]))
fig = plt.figure(figsize=(W, H))
fig.suptitle("Control: shuffle the positions and the neighbours stop agreeing", x=.03, ha="left", fontsize=18)
fig.text(.03, .88, "Same tracks and same steps, placed at random over the same points", fontsize=12, color="0.4")
for i, (name, pq, b, bound) in enumerate([("Real positions", q[:, 0], beta, True),
                                          ("Shuffled positions", ctrl.q[:, 0], ctrl.beta, False)]):
    a = fig.add_axes([.04 + i * .34, .14, .3, .68])
    field(a, boundary=bound)
    dots(a, pq, s=12)
    a.set_title(name, loc="left")
    a.text(10, -.6, f"fitted β = {b:.2f}", ha="center", va="top", fontsize=14,
           weight="bold")
colorbar(fig, [.75, .7, .21, .03])
fig.text(.74, .58, "With no spatial pattern, the fitted\nβ is near 0: each track keeps its\nown "
         "evidence, and no patches\nare invented.\n\nThe formal test repeats this\nshuffle 99 times "
         "(figure 6.1D).", fontsize=13, va="top", linespacing=1.4)
save(fig, "potts_7_shuffle.png")
print("control beta", ctrl.beta)
