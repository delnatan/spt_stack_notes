import sys, time, pickle, os
sys.path.insert(0, '/tmp/claude-0/s')
import numpy as np, polars as pl
from diffusionkit import Acquisition
from diffusionkit.fbmpost.simulate import simulate_tracks
from diffusionkit.gridpost import analyze_tracks
from diffusionkit.spatial import SpatialOptions, analyze_likelihoods
import centred as C
acq = Acquisition(dt_s=0.035, exposure_s=0.020)

def movie(seed, patterned=True):
    cache = f'/tmp/claude-0/s/movie_{seed}.pkl'
    if os.path.exists(cache):
        lik, xy, inside = pickle.load(open(cache, 'rb'))
    else:
        rng = np.random.default_rng(seed); N = 800
        xy = rng.uniform(0, 20, (N, 2)); inside = np.hypot(*(xy - 10).T) < 6
        D = np.where(inside, .05, .2)
        t = pl.concat([simulate_tracks(1, int(rng.integers(3, 9)), 1., D[i], .03, acq, rng=rng, n_sub=10, first_track_id=i) for i in range(N)])
        lik = analyze_tracks(t, acq, keep_likelihoods=True).likelihoods
        pickle.dump((lik, xy, inside), open(cache, 'wb'))
    if not patterned:
        xy = xy[np.random.default_rng(1000 + seed).permutation(len(xy))]
    return lik, xy, inside

def acc(q, centre, inside):
    near = 1 + (np.abs(centre - np.log(.2)) < np.abs(centre - np.log(.05)))
    return float(np.mean(near[q.argmax(1)] == np.where(inside, 1, 2)))

method, seeds, patterned, nsh = sys.argv[1], [int(s) for s in sys.argv[2].split(',')], sys.argv[3] == 'p', int(sys.argv[4])
from diffusionkit.gridpost import GridPostOptions
u = GridPostOptions().u_D()
for seed in seeds:
    lik, xy, inside = movie(seed, patterned)
    ll = lik.loglik_D; L = np.exp(ll - ll.max(1, keepdims=True))
    t0 = time.time()
    if method == 'ms':
        pos = pl.DataFrame({"track_id": lik.track_ids, "x_um": xy[:, 0], "y_um": xy[:, 1]})
        s = analyze_likelihoods(lik, u, pos, SpatialOptions(max_classes=3, n_shuffles=nsh))
        f = s.fit
        out = (s.n_classes, round(f.beta, 2), np.round(f.weight, 2), acc(f.q, f.centre, inside) if patterned else None, s.null.p if s.null else None,
               [round(x, 1) for x in s.selection["score"].to_list()])
    else:
        cen, sym = method[0] == 'c', method[1] == 's'
        ch, fits, p = (C.analyze_profile if method.endswith('p') else C.analyze)(L, u, xy, cen, sym, n_shuffles=nsh)
        f = fits[ch]
        out = (ch, round(float(f["beta"]), 2), np.round(f["w"], 2), acc(f["q"], f["centre"], inside) if patterned else None, p,
               [round(fits[c]["score"], 1) for c in (1, 2, 3)], [round(float(fits[c]["beta"]), 1) for c in (1, 2, 3)])
    print(method, seed, 'pattern' if patterned else 'none', out, f"{time.time() - t0:.0f}s", flush=True)
