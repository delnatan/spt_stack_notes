"""Three-region cells: D_A inside one disc, D_B inside another, D_C elsewhere. Methods: exact (exact-step Potts,
mean-field E-step), ms (multi-start, needs PYTHONPATH=dk_ms), map (smooth map). Usage: three.py method seed p|n"""
import sys, os, time, pickle
sys.path.insert(0, '/tmp/claude-0/s')
import numpy as np, polars as pl
from diffusionkit import Acquisition
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import analyze_tracks, GridPostOptions
acq = Acquisition(dt_s=0.035, exposure_s=0.020)
DS = np.array([.03, .1, .35])
u = GridPostOptions().u_D()

def movie(seed):
    cache = f'/tmp/claude-0/s/movie3_{seed}.pkl'
    if os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    rng = np.random.default_rng(seed); N = 800
    xy = rng.uniform(0, 20, (N, 2))
    region = np.full(N, 2)
    region[np.hypot(*(xy - [6, 6]).T) < 4.5] = 0
    region[np.hypot(*(xy - [14, 14]).T) < 4.5] = 1
    t = pl.concat([simulate_tracks(1, int(rng.integers(3, 9)), 1., DS[region[i]], .03, acq, rng=rng, n_sub=10, first_track_id=i) for i in range(N)])
    lik = analyze_tracks(t, acq, keep_likelihoods=True).likelihoods
    pickle.dump((lik, xy, region), open(cache, 'wb'))
    return lik, xy, region

def acc(lnd_of_track, region):
    """Each track's fitted ln D (its class centre or map value) to the nearest true ln D: right if that is its region's."""
    near = np.abs(lnd_of_track[:, None] - np.log(DS)[None, :]).argmin(1)
    return float(np.mean(near == region))

method, seed, patterned = sys.argv[1], int(sys.argv[2]), sys.argv[3] == 'p'
lik, xy, region = movie(seed)
if not patterned:
    xy = xy[np.random.default_rng(1000 + seed).permutation(len(xy))]
ll = lik.loglik_D; L = np.exp(ll - ll.max(1, keepdims=True))
t0 = time.time()
if method == 'exact':
    import exact as X
    ch, fits, scores, p, _ = X.analyze(L, u, xy, estep='mf')
    f = fits[ch]
    out = dict(C=ch, beta=round(f['beta'], 3), p=p, D=np.round(np.exp(f['centre']), 3).tolist(),
               acc=acc(f['centre'][f['q'].argmax(1)], region) if patterned else None,
               gains=[round((scores[c] - scores[c - 1]) / len(L), 4) for c in range(2, 5)])
elif method == 'ms':
    from diffusionkit.spatial import SpatialOptions, analyze_likelihoods
    pos = pl.DataFrame({"track_id": lik.track_ids, "x_um": xy[:, 0], "y_um": xy[:, 1]})
    s = analyze_likelihoods(lik, u, pos, SpatialOptions(max_classes=4, n_shuffles=19))
    f = s.fit
    out = dict(C=s.n_classes, beta=round(f.beta, 2), p=s.null.p if s.null else None, D=np.round(np.exp(f.centre), 3).tolist(),
               acc=acc(f.centre[f.q.argmax(1)], region) if patterned else None,
               gains=[None if g is None else round(g, 4) for g in s.selection["gain_per_track"].to_list()[1:]])
else:
    from diffusionkit.spatial import SmoothMapOptions, smooth_map_likelihoods
    pos = pl.DataFrame({"track_id": lik.track_ids, "x_um": xy[:, 0], "y_um": xy[:, 1]})
    m = smooth_map_likelihoods(lik, u, pos, SmoothMapOptions(n_shuffles=19))
    out = dict(gain=round(m.gain, 1), p=m.null.p, tau_s=round(m.tau_s, 2), tau_v=round(m.tau_v, 1),
               acc=acc(m.ln_D, region) if patterned else None)
    pickle.dump(m, open(f'/tmp/claude-0/s/map3_{seed}_{"p" if patterned else "n"}.pkl', 'wb'))
print(method, seed, 'pattern' if patterned else 'none', out, f'{time.time() - t0:.0f}s', flush=True)
