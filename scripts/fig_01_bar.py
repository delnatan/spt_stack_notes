"""Figure 1.3: the detection bar u, set per movie by empirical Bayes (rust/spotsolve-core/src/posterior.rs).

Runs the real `spotsolve.localize_stack` (single fits). With p_real ~ 0 the bar falls to its floor Z_LO = 3,
so every candidate height detect_z >= 3 is reported; the null curve is `prefilter::false_rate` re-implemented.
"""
import numpy as np

import spotsolve
from spotsolve import psf
from common import C, plt, save

sigma, H, W, T = 1.45, 128, 128, 24
lo, hi = 1.0 * sigma, min(1.5 * sigma, 0.5 * np.ceil(4 * sigma))   # Settings::widths()


def false_rate(u, lo, hi):
    """Expected noise maxima above u per pixel, widths searched over [lo, hi] (prefilter::false_rate)."""
    a, b = 1 / (lo ** 2 + 1 / 12), 1 / (hi ** 2 + 1 / 12)
    e = np.exp(-0.5 * u * u)
    rho1, rho2, rho3 = e / (2 * np.pi), u * e / (2 * np.pi) ** 1.5, (u * u - 1) * e / (4 * np.pi ** 2)
    return 0.25 * (a - b) * rho3 + 0.25 * a * rho2 + (a / (2 * np.pi) - 3 * (a - b) / (8 * np.pi)) * rho1


def movie(n_per_frame, bg, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] * 1.0
    out = []
    for _ in range(T):
        p = rng.uniform(6, H - 6, (n_per_frame, 2))
        f = np.exp(rng.uniform(np.log(60), np.log(3000), n_per_frame))   # a spread of fluxes
        clean = psf.model(psf.pack(bg, f, p[:, 0], p[:, 1]), yy, xx, sigma) if n_per_frame else np.full((H, W), bg)
        out.append(rng.poisson(clean))
    return np.array(out, float)


noise = movie(0, 50.0, 1)
gauss = 50 + np.sqrt(50) * np.random.default_rng(7).standard_normal((T, H, W))   # read noise only
spots = movie(40, 50.0, 2)
z_noise = np.concatenate([r.info["detect_z"] for r in spotsolve.localize_stack(noise, sigma, gain=1.0, p_real=1e-9)])
# gain ~ 0: the curve is flat at the measured read variance, so the deviance is Gaussian least squares
z_gauss = np.concatenate([r.info["detect_z"] for r in spotsolve.localize_stack(gauss, sigma, gain=1e-6, p_real=1e-9)])
low = spotsolve.localize_stack(spots, sigma, gain=1.0, p_real=1e-9)
z_spots = np.concatenate([r.info["detect_z"] for r in low])
p_spots = np.concatenate([r.info["p_real"] for r in low])
u90 = spotsolve.localize_stack(spots, sigma, gain=1.0)[0].info["u"]
u_noise90 = spotsolve.localize_stack(noise, sigma, gain=1.0)[0].info["u"]

edges = np.geomspace(3, 40, 41)
mid = np.sqrt(edges[1:] * edges[:-1])
area = T * H * W
null = area * (false_rate(edges[:-1], lo, hi) - false_rate(edges[1:], lo, hi))

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
zz = np.linspace(3, 5.5, 60)
ax[0].plot(zz, [(z_gauss >= v).sum() for v in zz], color=C["purple"], label="Gaussian noise (read only)")
ax[0].plot(zz, [(z_noise >= v).sum() for v in zz], color=C["grey"], label="Poisson noise, background 50")
ax[0].plot(zz, area * false_rate(zz, lo, hi), color=C["red"], ls="--", label="Euler-characteristic prediction")
ax[0].set_yscale("log"); ax[0].set_ylim(0.5, None)
ax[0].set_xlabel("candidate height z")
ax[0].set_ylabel("noise candidates with detect_z ≥ z")
ax[0].set_title(f"A. Noise alone, {T} frames of {H}×{W}")
ax[0].legend(frameon=False)

ax[1].hist(z_spots, edges, color=C["blue"], alpha=0.5, label="movie with emitters")
ax[1].step(edges[:-1], null, where="post", color=C["red"], label="noise null")
ax[1].axvline(u90, color=C["black"], ls="--", lw=1)
counts = np.histogram(z_spots, edges)[0]
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(0.3, 3 * counts.max())
ax[1].text(u90 * 1.04, 1.5 * counts.max(), f"u = {u90:.2f} (p_real 0.9)", fontsize=8, va="top")
ax[1].set_xlabel("detect_z")
ax[1].set_title("B. Heights = noise maxima + emitters")
ax[1].legend(frameon=False, loc="lower right")

order = np.argsort(z_spots)
ax[2].plot(z_spots[order], p_spots[order], color=C["blue"])
ax[2].axhline(0.9, color=C["grey"], lw=0.7, ls=":")
ax[2].axvline(u90, color=C["black"], ls="--", lw=1)
ax[2].set_xscale("log")
ax[2].set_xlim(3, 40)
ax[2].set_xlabel("detect_z")
ax[2].set_ylabel("P(real | z) = f1 / (f0 + f1)")
ax[2].set_title("C. Local false-discovery view (Efron)")
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
for a in ax[1:]:
    a.xaxis.set_major_locator(FixedLocator([3, 4, 5, 7, 10, 20, 40]))
    a.xaxis.set_major_formatter(ScalarFormatter()); a.xaxis.set_minor_formatter(NullFormatter())
save(fig, "fig_01_bar.png")
print("noise candidates", len(z_noise), "expected above 3:", area * false_rate(3, lo, hi),
      "u_noise", u_noise90, "u_spots", u90)
for v in (3, 3.5, 4, 4.5, 5):
    print(v, "gauss", (z_gauss >= v).sum(), "poisson", (z_noise >= v).sum(), "EC", area * false_rate(v, lo, hi))
