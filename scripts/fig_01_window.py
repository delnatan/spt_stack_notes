"""Figure 1.3: what happens inside one seed's window, step by step (spotsolve's `detect::fit_seed` with mixtures).

Re-implemented in numpy for display, because spotsolve does not expose its intermediate steps: weighted least
squares with fixed pixel variances V = read + gain * d stands in for the quasi-likelihood, and the width is held at
sigma. The logic is the code's: the efficient-score map on the residual (`detect::propose`), a component added at its
peak and everything refitted, kept while z = sqrt(2 dQ) >= u, then the backward check of each component's cost.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from common import C, plt, save

SIGMA, U_BAR, READ, GAIN, BG = 1.45, 5.0, 4.0, 1.0, 20.0
R = int(np.ceil(4 * SIGMA))  # window half-side, as in the code
N = 2 * R + 1
yy, xx = np.mgrid[0:N, 0:N].astype(float)


def psf(y, x):
    """Unit-flux Gaussian of width SIGMA integrated over each pixel."""
    s = SIGMA * np.sqrt(2)
    fy = .5 * (erf((yy + .5 - y) / s) - erf((yy - .5 - y) / s))
    fx = .5 * (erf((xx + .5 - x) / s) - erf((xx - .5 - x) / s))
    return fy * fx


def model(theta):
    m = np.full((N, N), theta[0])
    for a, y, x in theta[1:].reshape(-1, 3):
        m += a * psf(y, x)
    return m


def jacobian(theta, h=1e-5):
    m0 = model(theta).ravel()
    J = np.empty((m0.size, theta.size))
    for j in range(theta.size):
        t = theta.copy()
        t[j] += h
        J[:, j] = (model(t).ravel() - m0) / h
    return J


rng = np.random.default_rng(7)
truth = [(900., R, R), (550., R + 1.6, R - 1.7), (420., R - 3.6, R + 3.2)]  # the seed's emitter, a close one, a third
mu = model(np.array([BG] + [v for t in truth for v in t]))
d = mu + np.sqrt(READ + GAIN * mu) * rng.standard_normal(mu.shape)
W = 1 / (READ + GAIN * np.maximum(d, 1.))


def Q(theta):
    """Deviance in nats: half the weighted sum of squares."""
    return .5 * np.sum(W * (d - model(theta)) ** 2)


def refit(theta):
    k = (theta.size - 1) // 3
    lo = [-np.inf] + [0., -.5, -.5] * k
    hi = [np.inf] + [np.inf, N - .5, N - .5] * k
    r = least_squares(lambda t: (np.sqrt(W) * (d - model(t))).ravel(), theta, bounds=(lo, hi), x_scale="jac")
    return r.x


def score_map(theta):
    """The efficient score z(p) for a new component at each pixel, with the current ones free to readjust."""
    res = (W * (d - model(theta))).ravel()
    J = jacobian(theta)
    WJ = J * W.ravel()[:, None]
    F = J.T @ WJ
    g = J.T @ res
    z, a = np.full((N, N), -np.inf), np.zeros((N, N))
    for py in range(N):
        for px in range(N):
            gp = psf(py, px).ravel()
            b = WJ.T @ gp
            Fi_b = np.linalg.solve(F, b)
            u = gp @ res - Fi_b @ g
            i = gp @ (W.ravel() * gp) - b @ Fi_b
            if i > 1e-9:
                z[py, px], a[py, px] = u / np.sqrt(i), u / i
    return z, a


theta = refit(np.array([np.median(d)]))  # the level alone: the null the first component is tested against
steps = []
while True:
    z, a = score_map(theta)
    py, px = np.unravel_index(np.argmax(z), z.shape)
    trial = refit(np.append(theta, [max(a[py, px], 1.), py, px]))
    dq = Q(theta) - Q(trial)
    keep = 2 * dq >= U_BAR ** 2
    steps.append(dict(z=z, peak=(py, px), zpeak=z[py, px], zfit=np.sqrt(max(2 * dq, 0)), keep=keep, theta=theta))
    if not keep:
        break
    theta = trial

costs = []
for j in range((theta.size - 1) // 3):
    rest = np.delete(theta, slice(1 + 3 * j, 4 + 3 * j))
    costs.append(Q(refit(rest)) - Q(theta))

fig, ax = plt.subplots(1, len(steps) + 1, figsize=(3.0 * (len(steps) + 1), 3.6), constrained_layout=True)
ax[0].imshow(d, cmap="gray")
ax[0].plot(R, R, "+", color=C["orange"], ms=12, mew=2)
for _, y, x in truth:
    ax[0].plot(x, y, "o", mfc="none", mec=C["sky"], ms=9, mew=1)
ax[0].set_title(f"The window (±{R} px) around the seed (+)\ntrue emitters ○")
vmax = max(s["z"].max() for s in steps)
for i, s in enumerate(steps):
    a_ = ax[i + 1]
    a_.imshow(s["z"], cmap="magma", vmin=0, vmax=vmax)
    a_.contour(s["z"], [U_BAR], colors=[C["sky"]], linewidths=.8)
    k = (s["theta"].size - 1) // 3
    for _, y, x in s["theta"][1:].reshape(-1, 3):
        a_.plot(x, y, "x", color="white", ms=8, mew=1.5)
    a_.plot(s["peak"][1], s["peak"][0], "o", mfc="none", mec=C["green"] if s["keep"] else C["red"], ms=12, mew=2)
    verdict = "≥ u: keep, refit, repeat" if s["keep"] else "< u: stop adding"
    a_.set_title(f"Step {i + 1}: residual score, {k} fitted (×)\npeak z = {s['zpeak']:.1f}; after refit "
                 f"√(2ΔQ) = {s['zfit']:.1f}\n{verdict}", color=C["black"] if s["keep"] else C["red"], fontsize=8.5)
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([])
save(fig, "fig_01_window.png")
print("bar u =", U_BAR, "| removal costs (nats):", np.round(costs, 1), "vs u^2/2 =", U_BAR ** 2 / 2)
print("forward:", [(round(s["zpeak"], 1), round(s["zfit"], 1), s["keep"]) for s in steps])
