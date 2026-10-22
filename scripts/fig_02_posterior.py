"""Figure 2.1: how a track carries its own diffusion scale (rust/spotsolve-core/src/track.rs, re-implemented).

The grid of N_D = 8 diffusion coefficients, the log step density mixed over the track's posterior, the update
with FORGET = 0.05 returned to the base, and the gain log p_a(d) - log lambda - SWITCH_COST of a candidate link.
"""
import numpy as np
from scipy.special import logsumexp

from common import C, plt, save

N_D, FORGET, SWITCH = 8, 0.05, 2.5
R, se = 12.0, 0.3                          # max_step (px) and localization SE per axis
se2 = se ** 2
d_lo, d_hi = se2 / 8, max(R * R / 36 - se2, 2 * se2 / 8)
Dk = d_lo * (d_hi / d_lo) ** (np.arange(N_D) / (N_D - 1))
base = np.full(N_D, -np.log(N_D))
q, rho = 0.8, 0.004                        # continuation and density read from a movie
log_lambda = 2 * np.log(1 - q) + np.log(rho) - np.log(q)


def log_step(d2, v):
    return -np.log(2 * np.pi * v) - d2 / (2 * v)


def log_pred(w, d2):
    return logsumexp(w + log_step(d2, 2 * Dk + 2 * se2))


def update(w, d2):
    t = w + log_step(d2, 2 * Dk + 2 * se2)
    t -= logsumexp(t)
    return np.log((1 - FORGET) * np.exp(t) + FORGET * np.exp(base))


rng = np.random.default_rng(4)
tracks = {"slow, D = 0.05 px²/frame": (0.05, C["blue"]), "fast, D = 2.5 px²/frame": (2.5, C["red"])}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1.25]), constrained_layout=True)
states = {}
for k, (name, (D, c)) in enumerate(tracks.items()):
    w = base.copy()
    hist = [np.exp(w)]
    for _ in range(15):
        step = rng.normal(0, np.sqrt(2 * D), 2) + rng.normal(0, se, 2) - rng.normal(0, se, 2)
        w = update(w, step @ step)
        hist.append(np.exp(w))
    states[name] = (w, c, D)
    im = ax[k].imshow(np.array(hist).T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[k].set_yticks(range(N_D), [f"{d:.2g}" for d in Dk])
    ax[k].set_xlabel("steps linked so far")
    ax[k].set_ylabel("D_k (px²/frame)" if k == 0 else "")
    ax[k].set_title(f"Track posterior over D_k\n{name}")
fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.02, label="posterior weight")

d = np.linspace(0, R, 400)
for name, (w, c, D) in states.items():
    gain = np.array([log_pred(w, x * x) for x in d]) - log_lambda - SWITCH
    ax[2].plot(d, gain, color=c, label=name.split(",")[0] + " track after 15 steps")
    r = d[np.flatnonzero(gain > 0)[-1]] if (gain > 0).any() else 0
    ax[2].axvline(r, color=c, ls=":", lw=1)
gain0 = np.array([log_pred(base, x * x) for x in d]) - log_lambda - SWITCH
ax[2].plot(d, gain0, color=C["grey"], ls="--", label="new track, flat prior")
ax[2].axhline(0, color="k", lw=0.6)
ax[2].set_ylim(-8, 12)
ax[2].set_xlabel("candidate step length d (px)")
ax[2].set_ylabel("link gain, nats")
ax[2].set_title("Gain = log p_a(d) − log λ − 2.5\n(links only where > 0; dotted = implied radius)")
ax[2].legend(frameon=False, fontsize=7)
save(fig, "fig_02_posterior.png")
print("Dk", Dk, "log_lambda", log_lambda)
