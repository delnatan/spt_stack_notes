"""Ridge plot: one track's posterior as its steps come in, each ridge beside the trajectory it was computed from.

Each row uses the track's first n steps. Left: the whole trajectory, with the steps not yet used in faint gray and
the steps used in dark gray. Right: the posterior from those steps, flat prior, as a ridge offset vertically from
the others. The text on each ridge gives the information gained (KL from the flat prior, in bits) and the 90%
interval.

  --what D      Brownian D: `gridpost.posterior.track_loglik` on the grid of `GridPostOptions().u_D()`, flat prior
                in ln D (exactly what gridpost reports per track).
  --what alpha  fBm alpha: `fbmpost` profile likelihood over K (`whiten_track`, `profile_K`), normalized on
                `FbmPostOptions().alphas()` (a flat prior on alpha).

By default the track is simulated (`fbmpost.simulate.simulate_tracks`). `--csv` reads a track table instead, with
columns track_id, frame, x_um, y_um, sigma_x_um, sigma_y_um (pick one with `--track-id`).

Examples:
    python ridge_posterior.py                                    # D = 0.1 um^2/s, 256 steps
    python ridge_posterior.py --D 0.01 --out ridge_D_floor.png   # below the localization floor
    python ridge_posterior.py --what alpha --alpha 0.6
    python ridge_posterior.py --csv tracks.csv --track-id 17 --dt 0.035 --exposure 0.02
    python ridge_posterior.py --scale density                     # common height scale: ridges sharpen upward

Needs diffusionkit, numpy, polars and matplotlib.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import logsumexp

from diffusionkit import Acquisition
from diffusionkit.fbmpost import FbmPostOptions
from diffusionkit.fbmpost.likelihood import whiten_track
from diffusionkit.fbmpost.profile import profile_K
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import GridPostOptions
from diffusionkit.gridpost import posterior as P

FAINT, DARK, INK = "0.82", "0.25", "0.1"


def parse():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--what", choices=["D", "alpha"], default="D")
    a.add_argument("--D", type=float, default=0.1, help="simulated D (or D_ref at 0.1 s for alpha), um^2/s")
    a.add_argument("--alpha", type=float, default=1.0, help="simulated alpha (1 = Brownian)")
    a.add_argument("--sigma", type=float, default=0.03, help="localization SD, um")
    a.add_argument("--dt", type=float, default=0.035, help="frame interval, s")
    a.add_argument("--exposure", type=float, default=0.0, help="exposure time, s (blur is modelled)")
    a.add_argument("--steps", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256],
                   help="numbers of steps, one ridge each")
    a.add_argument("--seed", type=int, default=3)
    a.add_argument("--csv", help="track table to read instead of simulating")
    a.add_argument("--track-id", type=int, help="which track in --csv (default: the longest)")
    a.add_argument("--scale", choices=["peak", "density"], default="peak",
                   help="peak: every ridge the same height; density: one scale for all, so ridges sharpen upward")
    a.add_argument("--out", default=None, help="output image (default ridge_<what>.png)")
    return a.parse_args()


def load_track(args, acq):
    if args.csv:
        table = pl.read_csv(args.csv)
        if args.track_id is None:
            args.track_id = table.group_by("track_id").len().sort("len")["track_id"][-1]
        track = table.filter(pl.col("track_id") == args.track_id).sort("frame")
        truth = None
    else:
        alpha = 1.0 if args.what == "D" else args.alpha
        track = simulate_tracks(1, max(args.steps) + 1, alpha, args.D, args.sigma, acq,
                                rng=np.random.default_rng(args.seed))
        truth = args.D if args.what == "D" else args.alpha
    n_max = track.height - 1
    steps = [n for n in args.steps if n <= n_max]
    if len(steps) < len(args.steps):
        print(f"track has {n_max} steps; ridges for {sorted(set(args.steps) - set(steps))} dropped")
    return track, steps, truth


def posterior_D(prefix, acq):
    """x = log10 D, density per log10 D, bits, 90% interval in D."""
    u = GridPostOptions().u_D()
    lp = P.log_posterior(P.track_loglik(prefix, acq, u), P.flat(u))
    p = np.exp(lp)
    s = P.summary(p, u, 0.9)
    x = u / np.log(10)
    return x, p / np.gradient(x), P.information_bits(lp, P.flat(u)), (s["lo"], s["hi"])


def posterior_alpha(prefix, acq):
    """x = alpha, the profile likelihood normalized on the alpha grid, bits, 90% interval in alpha."""
    alphas = FbmPostOptions().alphas()
    lp = profile_K(whiten_track(prefix, acq, alphas)).loglik[0]
    lp = lp - logsumexp(lp)
    p = np.exp(lp)
    bits = float(np.sum(p * (lp + np.log(len(alphas)))) / np.log(2))
    c = np.cumsum(p)
    lo, hi = np.interp([0.05, 0.95], c, alphas)
    return alphas, p / np.gradient(alphas), bits, (lo, hi)


def flat_prior(what):
    """The starting point: flat prior on the same grid, 0 bits, 90% interval of the prior itself."""
    if what == "D":
        u = GridPostOptions().u_D()
        x, lo_hi = u / np.log(10), np.exp(np.interp([0.05, 0.95], [0, 1], [u[0], u[-1]]))
    else:
        x = FbmPostOptions().alphas()
        lo_hi = np.interp([0.05, 0.95], [0, 1], [x[0], x[-1]])
    return x, np.full_like(x, 1 / (x[-1] - x[0])), 0.0, tuple(lo_hi)


def main():
    args = parse()
    acq = Acquisition(dt_s=args.dt, exposure_s=args.exposure)
    track, steps, truth = load_track(args, acq)
    xy = track.select("x_um", "y_um").to_numpy()
    xy = xy - xy[0]
    post = posterior_D if args.what == "D" else posterior_alpha
    rows = [flat_prior(args.what)] + [post(track.head(n + 1), acq) for n in steps]
    ns = [0] + steps

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13})
    K = len(rows)
    head = 0.8                       # room above the top row for its ridge
    overlap = 1.8                    # a full-height ridge spans this many rows
    prior_h = 0.45                   # height of the flat prior in --scale peak (a full-height block would hide it)
    W, H = 11, 6.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.235, 0.17, 0.735, 0.69])
    ax.set_ylim(0, K + head)

    # Trajectory insets, all on the same scale, centred on their rows and staggered in two columns so each can be
    # two rows tall without touching its neighbours.
    pad = 0.06 * np.ptp(xy, axis=0).max()
    lo, hi = xy.min(0) - pad, xy.max(0) + pad
    half = (hi - lo).max() / 2
    mid = (lo + hi) / 2
    box = ax.get_position()
    row_h = box.height / (K + head)                  # one row, as a fraction of figure height
    side_h = 1.9 * row_h
    side_w = side_h * H / W
    x_cols = [box.x0 - 0.01 - 2 * side_w - 0.005, box.x0 - 0.01 - side_w]
    insets = []
    for i, n in enumerate(ns):
        y_mid = box.y0 + (K - 0.5 - i) * row_h
        t = fig.add_axes([x_cols[i % 2], y_mid - side_h / 2, side_w, side_h])
        t.plot(*xy.T, color=FAINT, lw=0.9, zorder=1)
        if n:
            t.plot(*xy[:n + 1].T, color=DARK, lw=1.3, zorder=2)
        t.plot(*xy[0], "o", color=INK, ms=4, zorder=3)
        t.set_xlim(mid[0] - half, mid[0] + half)
        t.set_ylim(mid[1] - half, mid[1] + half)
        t.axis("off")
        insets.append(t)
    # Scale bar under the lowest inset of the left column, clear of the x axis.
    t = insets[K - 1 if (K - 1) % 2 == 0 else K - 2]
    bar = float(f"{2 * half / 4:.1g}")
    t.plot([mid[0] + half - bar, mid[0] + half], [mid[1] - half] * 2, color=INK, lw=2, clip_on=False)
    t.text(mid[0] + half - bar / 2, mid[1] - half - 0.06 * half, f"{bar:g} µm", ha="center", va="top",
           fontsize=12, color=INK)

    # Ridges, drawn top (prior) to bottom so each lower ridge covers the one behind it.
    top = max(r[1].max() for r in rows[1:])
    for i, (x, dens, bits, (lo_i, hi_i)) in enumerate(rows):
        base = K - 1 - i
        if i == 0 and args.scale == "peak":
            h = np.full_like(x, prior_h)
        else:
            h = dens / (dens.max() if args.scale == "peak" else top) * overlap
        ax.fill_between(x, base, base + h, color="white", lw=0, zorder=2 * i + 1)
        ax.fill_between(x, base, base + h, color=FAINT, alpha=0.55, lw=0, zorder=2 * i + 1)
        ax.plot(x, base + h, color=INK, lw=1.4, zorder=2 * i + 2)
        ax.axhline(base, color=INK, lw=0.5, zorder=2 * i + 2)
        if i == 0:
            label = "0 bits   flat on " + ("10⁻⁵–10" if args.what == "D" else "0–2")
        else:
            fmt = "{:.2g}" if args.what == "D" else "{:.2f}"
            label = f"{bits:.1f} bits   " + fmt.format(lo_i) + "–" + fmt.format(hi_i)
        ax.text(0.005, base + 0.1, "prior" if i == 0 else f"{ns[i]} steps", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=13, color=INK, zorder=100)
        ax.text(0.995, base + 0.1, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=13, color=INK, zorder=100,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    unit = " (µm²/s)" if args.what == "D" else ""
    ax.text(0.995, K + head - 0.05, f"information · 90% interval{unit}", transform=ax.get_yaxis_transform(),
            ha="right", va="top", fontsize=13, color="0.4", zorder=100)
    if truth is not None:
        xt = np.log10(truth) if args.what == "D" else truth
        ax.axvline(xt, color=INK, ls=":", lw=1.5, zorder=99)
    if args.what == "D":
        # Show the part of the grid where any posterior has mass (the prior covers all of it).
        x_all = rows[0][0]
        keep = np.zeros_like(x_all, bool)
        for x, dens, *_ in rows[1:]:
            keep |= dens > 1e-3 * dens.max()
        ax.set_xlim(x_all[keep].min() - 0.2, x_all[keep].max() + 0.2)
        ax.set_xlabel("log₁₀ D (µm²/s)")
        floor = args.sigma ** 2 / (args.dt - args.exposure / 3) if not args.csv else None
        if floor is not None and ax.get_xlim()[0] < np.log10(floor):
            ax.axvspan(ax.get_xlim()[0], np.log10(floor), color="0.93", zorder=0, lw=0)
            ax.text(ax.get_xlim()[0], K + head - 0.05, " below localization floor", ha="left", va="top",
                    fontsize=12, color="0.45")
        what = "D"
    else:
        ax.set_xlim(0, 2)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xlabel("α")
        what = "α"
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)

    if args.csv:
        src = f"track {args.track_id} from {args.csv}"
    else:
        src = (f"simulated {'D' if args.what == 'D' else 'D_ref'} = {args.D:g} µm²/s"
               + (f", α = {args.alpha:g}" if args.what == "alpha" else "")
               + f", σ = {1000 * args.sigma:g} nm, Δt = {1000 * args.dt:g} ms"
               + (f", exposure {1000 * args.exposure:g} ms" if args.exposure else ""))
    fig.text(0.03, 0.965, f"Posterior of {what} as steps come in", fontsize=18, ha="left", va="top")
    fig.text(0.03, 0.905, src, fontsize=12, color="0.4", ha="left", va="top")
    fig.text(0.03, 0.03, "dark: steps used   faint: not yet used" + ("   dotted: true value" if truth else ""),
             fontsize=12, color="0.4")
    out = args.out or f"ridge_{args.what}.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
