"""Figure 1.2: the camera's noise curve and the quasi-likelihood deviance spotsolve fits and tests with.

Mirrors `prefilter::Curve::at` and `Curve::deviance` (rust/spotsolve-core/src/prefilter.rs): above the floor the
deviance is the Poisson deviance of d + tau at m + tau, divided by the gain, tau = read / gain.
"""
import numpy as np

from common import C, plt, save


def V(m, read, gain, floor=1e-3):
    return np.maximum(read + gain * m, floor)


def dev(d, m, read, gain):
    tau = read / gain
    a, b = d + tau, m + tau
    return (a * np.log(a / b) - (a - b)) / gain


cams = [("Poisson camera (read 0, gain 1)", 0.0, 1.0, C["blue"]),
        ("sCMOS-like (read 9 ADU², gain 2)", 9.0, 2.0, C["orange"]),
        ("EMCCD-like (read 0, gain 1 × excess 2)", 0.0, 2.0, C["green"])]

fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
m = np.linspace(0, 120, 200)
for name, r, g, c in cams:
    ax[0].plot(m, V(m, r, g), color=c, label=name)
ax[0].set_xlabel("light the model puts on a pixel, m (ADU above offset)")
ax[0].set_ylabel("pixel variance V(m)")
ax[0].set_title("A. One noise curve per movie: V = read + gain·m")
ax[0].legend(frameon=False)

# B: deviance per pixel as the data d move around a model value m = 20: skewed, unlike least squares.
mu = 20.0
d = np.linspace(2, 50, 300)
for name, r, g, c in cams[:2]:
    ax[1].plot(d, dev(d, mu, r, g), color=c, label=name.split(" (")[0])
    ax[1].plot(d, (d - mu) ** 2 / (2 * V(mu, r, g)), color=c, ls=":", lw=1)
ax[1].axvline(mu, color=C["grey"], lw=0.6)
ax[1].set_ylim(0, 8)
ax[1].set_xlabel("observed pixel value d (model m = 20)")
ax[1].set_ylabel("deviance q(d, m), nats")
ax[1].set_title("B. Quasi-likelihood deviance (solid)\nvs fixed-weight least squares (dotted)")
ax[1].legend(frameon=False)

# C: what the weights 1/V(m) do across a bright spot: the peak is down-weighted relative to the wings.
x = np.arange(-6, 7)
from scipy.special import erf
s = 1.45
E = 0.5 * (erf((x + .5) / (s * np.sqrt(2))) - erf((x - .5) / (s * np.sqrt(2))))
for flux, c in [(300, C["sky"]), (3000, C["red"])]:
    mm = 20 + flux * E * E[6]
    w = 1 / V(mm, 0, 1)
    ax[2].plot(x, w / w.max(), "o-", color=c, ms=3, label=f"flux {flux}")
ax[2].set_xlabel("pixel offset from spot centre")
ax[2].set_ylabel("relative weight 1/V(m)")
ax[2].set_title("C. Fisher-scoring weights across a spot\n(Poisson camera, background 20)")
ax[2].legend(frameon=False)
save(fig, "fig_01_noise.png")
