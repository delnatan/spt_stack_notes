"""Figures 4b.1-4b.2: what each step of a track teaches about D and alpha, against the MSD.

For D: the real `gridpost.posterior.track_loglik`, `log_posterior`, `flat` and `information_bits` on the first n
steps of long simulated tracks, the MSD's D from `classic.compute_msd` + `fit_brownian_msd` (3 lags, the package
default, and the 30% rule), and the Cramer-Rao bound from `gridpost.likelihood.motion_covariance` and
`localization_covariance`. For alpha: the real `fbmpost.likelihood.whiten_batch` and `profile.profile_K`, the
profile normalized on `FbmPostOptions().alphas()` (a flat prior on alpha), the MSD's alpha from
`fit_anomalous_msd` and `fit_loglog_msd` on the first 30% of lags, and the bound from `fgn_motion_covariance`.
No exposure blur, since the MSD estimators refuse blurred tracks. Tracks from `fbmpost.simulate.simulate_tracks`.
Run from this folder; the simulations are cached in ./_cache_04b.npz (delete it to rerun them).
"""
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from diffusionkit import Acquisition
from diffusionkit.classic import MSDOptions, compute_msd
from diffusionkit.classic.estimators import fit_anomalous_msd, fit_brownian_msd, fit_loglog_msd
from diffusionkit.fbmpost import FbmPostOptions
from diffusionkit.fbmpost.covariance import fgn_motion_covariance
from diffusionkit.fbmpost.likelihood import whiten_batch
from diffusionkit.fbmpost.profile import profile_K
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import GridPostOptions
from diffusionkit.gridpost import posterior as P
from diffusionkit.gridpost.likelihood import brownian_loglik_batch, localization_covariance, motion_covariance
from common import C, plt, save

acq = Acquisition(dt_s=0.035, exposure_s=0.0)
SIG = 0.03                                   # localization SD, um
u = GridPostOptions().u_D()                  # ln D grid, 1e-5 .. 10 um^2/s
W_D = u[-1] - u[0]                           # the flat prior's width in ln D (nats)
alphas = FbmPostOptions().alphas()           # 0.02 .. 1.98
W_A = alphas[-1] - alphas[0] + (alphas[1] - alphas[0])
STEPS_D = np.array([2, 4, 8, 16, 32, 64, 128, 256])
STEPS_A = np.array([4, 8, 16, 32, 64, 128, 256])
D_CASES = [(0.1, C["blue"]), (0.01, C["purple"])]
A_TRUE, DREF = 0.6, 0.1
N_D, N_A = 3000, 300  # tracks per D case, and for alpha (whitening 256 steps is the slow part)
CACHE = Path(__file__).with_name("_cache_04b.npz")


def window(tr, n):
    return tr.filter(tr["frame"] < n + 1)


def sd_frames(n):
    return np.full((n + 1, 2), SIG)


def fisher_lnD(D, m):
    """Fisher information on ln D of m displacements, both axes."""
    S = D * motion_covariance(m, acq.dt_s) + localization_covariance(sd_frames(m))[0]
    M = np.linalg.solve(S, D * motion_covariance(m, acq.dt_s))
    return float(np.trace(M @ M))  # 2 axes x 1/2 tr((S^-1 dS)^2)


def fisher_alpha(m, a=A_TRUE, h=1e-4):
    """Efficient information on alpha with ln K unknown, both axes."""
    K = DREF * 0.1 ** (1 - a)
    B = localization_covariance(sd_frames(m))[0]
    Ca = fgn_motion_covariance(m, acq.dt_s, np.array([a]))[0]
    dC = (fgn_motion_covariance(m, acq.dt_s, np.array([a + h]))[0]
          - fgn_motion_covariance(m, acq.dt_s, np.array([a - h]))[0]) / (2 * h)
    S = K * Ca + B
    G = [np.linalg.solve(S, K * Ca), np.linalg.solve(S, K * dC)]
    I = np.array([[np.trace(p @ q) for q in G] for p in G])
    return I[1, 1] - I[0, 1] ** 2 / I[0, 0]


def gauss_bits(I, W):
    """KL of a normal of precision I from a flat prior of width W, in bits."""
    return np.log2(W * np.sqrt(I / (2 * np.pi * np.e)))


def msd_D(tr, n, lags):
    cv = compute_msd(tr, acq, MSDOptions(max_lag=lags))
    return fit_brownian_msd(cv).parameters["D_um2_s"]


def simulate():
    out = {}
    rng = np.random.default_rng(12)
    for D, _ in D_CASES:
        tracks = simulate_tracks(N_D, STEPS_D[-1] + 1, 1.0, D, SIG, acq, rng=rng)
        xy = tracks.select("x_um", "y_um").to_numpy().reshape(N_D, -1, 2)
        bits, mle, m3, m30 = (np.zeros((N_D, len(STEPS_D))) for _ in range(4))
        for j, n in enumerate(STEPS_D):
            ll = brownian_loglik_batch(xy[:, :n + 1], np.full((N_D, n + 1, 2), SIG), acq.dt_s, 0.0, np.exp(u))
            for t in range(N_D):
                lp = P.log_posterior(ll[t], P.flat(u))
                bits[t, j] = P.information_bits(lp, P.flat(u))
            mle[:, j] = np.exp(u[ll.argmax(1)])
        for (tid,), g in tracks.group_by("track_id"):
            for j, n in enumerate(STEPS_D):
                w = window(g, n)
                m3[tid, j] = msd_D(w, n, min(3, n))
                m30[tid, j] = msd_D(w, n, max(2, round(0.3 * n)))
        out[f"D{D}"] = np.stack([bits, mle, m3, m30])
        print("D", D, "done", flush=True)
    # alpha: prefixes of the same fBm tracks
    rng = np.random.default_rng(13)
    tracks = simulate_tracks(N_A, STEPS_A[-1] + 1, A_TRUE, DREF, SIG, acq, rng=rng)
    xy = tracks.select("x_um", "y_um").to_numpy().reshape(N_A, -1, 2)
    abits, amle, anls, alog, a10 = (np.full((N_A, len(STEPS_A)), np.nan) for _ in range(5))
    for j, n in enumerate(STEPS_A):
        w = whiten_batch(xy[:, :n + 1], np.full((N_A, n + 1, 2), SIG), acq.dt_s, 0.0, alphas)
        lp = profile_K(w).loglik
        lp = lp - logsumexp(lp, axis=1, keepdims=True)
        p = np.exp(lp)
        abits[:, j] = (p * (lp + np.log(len(alphas)))).sum(1) / np.log(2)
        amle[:, j] = alphas[lp.argmax(1)]
        for (t,), g in tracks.group_by("track_id"):
            g = window(g, n)
            cv = compute_msd(g, acq, MSDOptions(max_lag=max(3, round(0.3 * n))))
            anls[t, j] = msd_alpha(fit_anomalous_msd(cv))
            alog[t, j] = msd_alpha(fit_loglog_msd(cv))
            cv = compute_msd(g, acq, MSDOptions(max_lag=min(10, n)))
            a10[t, j] = msd_alpha(fit_anomalous_msd(cv))
        print("alpha", n, "done", flush=True)
    out["A"] = np.stack([abits, amle, anls, alog, a10])
    np.savez(CACHE, **out)
    return out


def msd_alpha(f):
    return f.parameters.get("alpha", np.nan) if f.status != "insufficient_data" else np.nan


res = dict(np.load(CACHE)) if CACHE.exists() else simulate()

# ---------------- Figure 4b.1: D ----------------
fig, ax = plt.subplots(1, 3, figsize=(14, 3.7), constrained_layout=True)
rng = np.random.default_rng(3)
one = simulate_tracks(1, STEPS_D[-1] + 1, 1.0, 0.1, SIG, acq, rng=rng)
cmap = plt.get_cmap("viridis")
for j, n in enumerate(STEPS_D):
    w = window(one, n)
    lp = P.log_posterior(P.track_loglik(w, acq, u), P.flat(u))
    col = cmap(j / (len(STEPS_D) - 1))
    dens = np.exp(lp) / np.gradient(u / np.log(10))
    ax[0].plot(u / np.log(10), dens, color=col, lw=1.4,
               label=f"{n} steps: {P.information_bits(lp, P.flat(u)):.1f} bits")
    d3 = msd_D(w, n, min(3, n))
    if d3 > 0:
        ax[0].plot(np.log10(d3), -0.25 - 0.18 * j, "v", color=col, ms=5)
    else:
        ax[0].plot(-2.95, -0.25 - 0.18 * j, "<", color=col, ms=5)
ax[0].axvline(-1, color=C["black"], ls=":", lw=1)
ax[0].axhline(0, color=C["grey"], lw=.5)
ax[0].set_xlim(-3, 0.5)
ax[0].set_xlabel("log₁₀ D (µm²/s)"); ax[0].set_ylabel("posterior density per log₁₀ D (flat prior)")
ax[0].set_title("A. One track (D = 0.1), its first n steps\n▼ below: the 3-lag MSD estimate from the same steps")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")

nfine = np.unique(np.round(np.geomspace(2, 256, 40)).astype(int))
for D, col in D_CASES:
    bits = res[f"D{D}"][0]
    q = np.percentile(bits, [10, 50, 90], axis=0)
    ax[1].fill_between(STEPS_D, q[0], q[2], color=col, alpha=.15, lw=0)
    ax[1].plot(STEPS_D, q[1], "o-", color=col, ms=4, label=f"D = {D}: median over {N_D} tracks (band 10-90%)")
    ax[1].plot(nfine, [gauss_bits(fisher_lnD(D, m), W_D) for m in nfine], color=col, ls="--", lw=1)
ax[1].plot([], [], color=C["grey"], ls="--", lw=1, label="normal with the Fisher information (theory)")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(STEPS_D, [str(s) for s in STEPS_D])
ax[1].set_xlabel("steps in the track"); ax[1].set_ylabel("information gained about D (bits)")
ax[1].set_title("B. KL(posterior ‖ flat prior on ln D): ½ bit per doubling")
ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")

for D, col in D_CASES:
    _, mle, m3, m30 = res[f"D{D}"]
    crb = np.array([D ** 2 / fisher_lnD(D, n) for n in STEPS_D])
    for est, ls, mk, lab in [(mle, "-", "o", "likelihood (MLE)"), (m3, "--", "s", "MSD, 3 lags"),
                             (m30, ":", "^", "MSD, first 30% of lags")]:
        eff = crb / np.mean((est - D) ** 2, axis=0)
        ax[2].plot(STEPS_D, eff, ls=ls, marker=mk, color=col, ms=4, label=f"D = {D}: {lab}")
ax[2].axhline(1, color=C["grey"], lw=.6)
ax[2].set_xscale("log", base=2); ax[2].set_xticks(STEPS_D, [str(s) for s in STEPS_D])
ax[2].set_ylim(0, 1.4)
ax[2].set_xlabel("steps in the track"); ax[2].set_ylabel("efficiency  =  Cramér–Rao bound / MSE of D̂")
ax[2].set_title("C. Share of the track's information each estimator uses")
ax[2].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0, -0.17), ncol=2)
save(fig, "fig_04b_D.png")

# ---------------- Figure 4b.2: alpha ----------------
fig, ax = plt.subplots(1, 3, figsize=(14, 3.7), constrained_layout=True)
one = simulate_tracks(1, STEPS_A[-1] + 1, A_TRUE, DREF, SIG, acq, rng=np.random.default_rng(5))
xy1 = one.select("x_um", "y_um").to_numpy()[None]
da = alphas[1] - alphas[0]
for j, n in enumerate(STEPS_A):
    w = whiten_batch(xy1[:, :n + 1], np.full((1, n + 1, 2), SIG), acq.dt_s, 0.0, alphas)
    lp = profile_K(w).loglik[0]
    lp = lp - logsumexp(lp)
    bits = np.sum(np.exp(lp) * (lp + np.log(len(alphas)))) / np.log(2)
    ax[0].plot(alphas, np.exp(lp) / da, color=cmap(j / (len(STEPS_A) - 1)), lw=1.4, label=f"{n} steps: {bits:.1f} bits")
ax[0].axvline(A_TRUE, color=C["black"], ls=":", lw=1)
ax[0].set_xlabel("α"); ax[0].set_ylabel("normalized profile likelihood (flat prior on α)")
ax[0].set_title(f"A. One fBm track (α = {A_TRUE}), its first n steps")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")

abits, amle, anls, alog, a10 = res["A"]
q = np.percentile(abits, [10, 50, 90], axis=0)
ax[1].fill_between(STEPS_A, q[0], q[2], color=C["orange"], alpha=.15, lw=0)
ax[1].plot(STEPS_A, q[1], "o-", color=C["orange"], ms=4, label=f"α: median over {N_A} tracks (band 10-90%)")
nfa = nfine[nfine >= 4]
ax[1].plot(nfa, [gauss_bits(fisher_alpha(m), W_A) for m in nfa], color=C["orange"], ls="--", lw=1,
           label="normal with the Fisher information, K unknown")
q = np.percentile(res["D0.1"][0], [50], axis=0)[0]
ax[1].plot(STEPS_D, q, "o-", color=C["blue"], ms=3, alpha=.5, label="D (from figure 4b.1B, D = 0.1)")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(STEPS_D, [str(s) for s in STEPS_D])
ax[1].set_xlabel("steps in the track"); ax[1].set_ylabel("information gained (bits)")
ax[1].set_title("B. KL(normalized profile ‖ flat prior on α)")
ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")

crb = np.array([1 / fisher_alpha(n) for n in STEPS_A])
for est, ls, mk, col, lab in [(amle, "-", "o", C["orange"], "profile likelihood (MLE)"),
                              (anls, "--", "s", C["red"], "MSD, NLS in linear space (package default)"),
                              (alog, ":", "^", C["purple"], "MSD, log-log slope"),
                              (a10, "-.", "D", C["sky"], "MSD, NLS on the first 10 lags")]:
    mse = np.nanmean((est - A_TRUE) ** 2, axis=0)
    ax[2].plot(STEPS_A, crb / mse, ls=ls, marker=mk, color=col, ms=4, label=lab)
ax[2].axhline(1, color=C["grey"], lw=.6)
ax[2].set_xscale("log", base=2); ax[2].set_xticks(STEPS_A, [str(s) for s in STEPS_A])
ax[2].set_ylim(0, 1.6)
ax[2].set_xlabel("steps in the track"); ax[2].set_ylabel("efficiency  =  Cramér–Rao bound / MSE of α̂")
ax[2].set_title("C. Share of the track's information about α each estimator uses\n(MSD fits on the first 30% of lags unless noted)")
ax[2].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.17))
save(fig, "fig_04b_alpha.png")

# ---------------- numbers for the text ----------------
np.set_printoptions(precision=3, suppress=True)
crb_a = crb
for D, _ in D_CASES:
    bits, mle, m3, m30 = res[f"D{D}"]
    crb = np.array([D ** 2 / fisher_lnD(D, n) for n in STEPS_D])
    print(f"D={D} steps", STEPS_D)
    print("  median bits", np.median(bits, 0), " theory", np.array([gauss_bits(fisher_lnD(D, n), W_D) for n in STEPS_D]))
    print("  sd(lnD) CRB", np.sqrt(1 / np.array([fisher_lnD(D, n) for n in STEPS_D])))
    for nm, e in [("mle", mle), ("msd3", m3), ("msd30", m30)]:
        print(f"  eff {nm}", crb / np.mean((e - D) ** 2, 0), " share<=0", np.mean(e <= 0, 0))
print("alpha steps", STEPS_A)
print("  median bits", np.median(abits, 0), " theory", np.array([gauss_bits(fisher_alpha(n), W_A) for n in STEPS_A]))
print("  sd CRB", np.sqrt(crb_a))
for nm, e in [("mle", amle), ("nls", anls), ("loglog", alog), ("nls10", a10)]:
    print(f"  eff {nm}", crb_a / np.nanmean((e - A_TRUE) ** 2, 0), " median", np.nanmedian(e, 0),
          " missing", np.mean(np.isnan(e), 0))
print("  share of MLE at grid edges", np.mean((amle <= alphas[0]) | (amle >= alphas[-1]), 0))
