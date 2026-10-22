"""Figure 1.1: the detection pipeline on one simulated frame.

Panels 2-3 re-implement spotsolve's screen (rust/spotsolve-core/src/prefilter.rs: `score`, `screen`) in numpy
for illustration; panel 4 is the real `spotsolve.localize` output (single fits and mixtures).
"""
import numpy as np
from scipy import ndimage
from scipy.special import erf

import spotsolve
from spotsolve import psf
from common import C, plt, save

rng = np.random.default_rng(3)
H = W = 72
sigma, bg = 1.45, 20.0
# Emitters: a sparse field plus two close pairs (2.0 and 3.0 sigma apart) and a faint one.
pos = [(12, 14), (15, 52), (30, 30), (50, 15), (58, 58), (40, 55), (24, 64), (62, 34)]
amp = [600, 400, 300, 800, 250, 500, 350, 450]
pos += [(44, 30), (44, 30 + 2.0 * sigma), (20, 38), (20 + 3.0 * sigma, 38), (8, 30)]
amp += [500, 500, 450, 450, 110]
pos = np.array(pos, float) + rng.uniform(-0.5, 0.5, (len(pos), 2))
amp = np.array(amp, float)
yy, xx = np.mgrid[0:H, 0:W] * 1.0
clean = psf.model(psf.pack(bg, amp, pos[:, 0], pos[:, 1]), yy, xx, sigma)
frame = rng.poisson(clean).astype(float)

# --- the screen: regress each (2R+1)^2 window on one pixel-integrated PSF g and a constant (prefilter::score)
R = int(np.ceil(4 * sigma))
t = np.arange(-R, R + 1)
k = 0.5 * (erf((t + 0.5) / (sigma * np.sqrt(2))) - erf((t - 0.5) / (sigma * np.sqrt(2))))
def corr(img, k1):
    return ndimage.correlate1d(ndimage.correlate1d(img, k1, axis=0, mode="constant"), k1, axis=1, mode="constant")
ones = np.ones_like(frame)
fg, fu = corr(frame, k), corr(frame, np.ones_like(k))
s1, s2, n = corr(ones, k), corr(ones, k * k), corr(ones, np.ones_like(k))
den = s2 - s1 ** 2 / n
A = (fg - s1 * fu / n) / den
level = (fu - A * s1) / n
V = np.maximum(level, 1e-3)          # Poisson camera: read 0, gain 1 (Curve::POISSON)
z = A * np.sqrt(den) / np.sqrt(V)    # score test of the flux, z = A |g - gbar| / sqrt(V)
mask = z >= 2.0                      # posterior::CANDIDATE
neg_log = -ndimage.gaussian_laplace(frame, sigma)
size = 2 * int(np.ceil(sigma)) + 1
seeds = np.argwhere(mask & (neg_log == ndimage.maximum_filter(neg_log, size)))

single = spotsolve.localize(frame, sigma, gain=1.0)
mixed = spotsolve.localize(frame, sigma, gain=1.0, fit_mixtures=True)

fig, ax = plt.subplots(1, 4, figsize=(11.5, 3.2))
ax[0].imshow(frame, cmap="gray")
ax[0].set_title("1. Frame (Poisson, background 20)")
im = ax[1].imshow(z, cmap="magma", vmin=-2, vmax=12)
ax[1].contour(mask, levels=[0.5], colors=C["sky"], linewidths=0.7)
ax[1].set_title("2. Score z per pixel; mask z ≥ 2")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
ax[2].imshow(neg_log, cmap="gray")
ax[2].plot(seeds[:, 1], seeds[:, 0], "+", color=C["orange"], ms=7, mew=1.3)
ax[2].set_title(f"3. −LoG with seeds in the mask ({len(seeds)})")
ax[3].imshow(frame, cmap="gray")
ax[3].plot(pos[:, 1], pos[:, 0], "o", mfc="none", mec=C["green"], ms=9, mew=1.2, label="truth")
ax[3].plot(single.positions[:, 1], single.positions[:, 0], "x", color=C["red"], ms=6, mew=1.3,
           label=f"single fits ({len(single)})")
ax[3].plot(mixed.positions[:, 1], mixed.positions[:, 0], "+", color=C["sky"], ms=8, mew=1.3,
           label=f"mixtures ({len(mixed)})")
ax[3].set_title(f"4. spotsolve.localize, u = {mixed.info['u']:.2f}")
ax[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=0.6)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
save(fig, "fig_01_pipeline.png")
print("truth", len(pos), "single", len(single), "mixed", len(mixed), "u", single.info["u"], mixed.info["u"])
