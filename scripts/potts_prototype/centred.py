"""Prototype: hidden Potts with (w, beta) fitted jointly by the neighbour-predictive score, optional centring
(Caragea & Kaiser 2009) and an optional symmetric graph. Compared against diffusionkit.spatial (multi-start)."""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from diffusionkit.spatial import potts

BETA_MAX, MIN_SD, SWEEPS, TINY = potts.BETA_MAX, potts.MIN_SD, potts.SWEEPS, 1e-300


def graph(xy, k, symmetric):
    nb = potts.knn(xy, k)
    n = len(xy)
    rows, cols = np.repeat(np.arange(n), k), nb.ravel()
    M = csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
    if symmetric:
        M = ((M + M.T) > 0).astype(float)
    deg = np.asarray(M.sum(1)).ravel()
    return csr_matrix(M.multiply(1 / deg[:, None]))


def logits(m, a, beta, centred):
    w = np.exp(a - a.max()); w /= w.sum()
    return np.log(w) + beta * (m - (w if centred else 0.)), w


def softmax(x):
    p = np.exp(x - x.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def score(ev, m, a, beta, centred):
    lg, _ = logits(m, a, beta, centred)
    return float(np.sum(np.log((softmax(lg) * ev).sum(1) + TINY)))


def fit_wb(ev, m, a, beta, centred, learn_beta=True):
    C = len(a)
    if C == 1:
        return a, beta  # one class: neither w nor beta does anything
    def neg(x):
        aa = np.append(x[:C - 1], 0.)
        b = x[C - 1] if learn_beta else beta
        return -score(ev, m, aa, b, centred)
    x0 = np.append(a[:C - 1] - a[-1], beta) if learn_beta else a[:C - 1] - a[-1]
    bounds = [(-30, 30)] * (C - 1) + ([(0., BETA_MAX)] if learn_beta else [])
    r = minimize(neg, x0, method="L-BFGS-B", bounds=bounds)
    aa = np.append(r.x[:C - 1], 0.)
    return aa, (r.x[C - 1] if learn_beta else beta)


def fit(lik, u, A, C, centred, beta=None, init=None, max_iter=300, tol=1e-3):
    if init is None:
        centre, sd, w = potts.initial(lik, u, C)
        a, b = np.log(w), 0.
    else:
        centre, sd, a, b = init["centre"].copy(), init["sd"].copy(), init["a"].copy(), init["beta"]
    learn = beta is None
    if not learn:
        b = beta
    q = np.full((len(lik), C), 1 / C)
    for it in range(1, max_iter + 1):
        q_old, b_old = q, b
        p = potts.densities(u, centre, sd)
        ev = lik @ p + TINY
        for _ in range(SWEEPS):
            lg, _ = logits(A @ q, a, b, centred)
            q = softmax(lg + np.log(ev))
        m = A @ q
        mass = (lik.T @ (q / ev)) * p
        tot = mass.sum(0)
        centre = u @ mass / tot
        sd = np.sqrt(np.maximum((u ** 2) @ mass / tot - centre ** 2, 0) + MIN_SD ** 2)
        a, b = fit_wb(ev, m, a, b, centred, learn)
        if it > 10 and np.abs(q - q_old).max() < tol and abs(b - b_old) < tol:
            break
    p = potts.densities(u, centre, sd)
    ev = lik @ p + TINY
    o = np.argsort(centre)
    res = dict(centre=centre[o], sd=sd[o], a=a[o], beta=b, q=q[:, o], ev=ev[:, o], n_iter=it)
    res["score"] = score(res["ev"], A @ res["q"], res["a"], b, centred)
    res["w"] = np.exp(res["a"] - res["a"].max()); res["w"] /= res["w"].sum()
    return res


def analyze(lik, u, xy, centred, symmetric, max_classes=3, n_shuffles=99, gain_per_track=.02, seed=0, k=8):
    A = graph(xy, k, symmetric)
    n = len(lik)
    plains, fits = {}, {}
    for c in range(1, max_classes + 1):
        plains[c] = fit(lik, u, A, c, centred, beta=0.)
        fits[c] = plains[c] if c == 1 else fit(lik, u, A, c, centred, init=plains[c])
    chosen = 1
    for c in range(2, max_classes + 1):
        if (fits[c]["score"] - fits[c - 1]["score"]) / n < gain_per_track:
            break
        chosen = c
    p = None
    if chosen > 1 and n_shuffles:
        base = plains[chosen]["score"]
        gain = fits[chosen]["score"] - base
        gs = []
        for s in np.random.SeedSequence(seed).spawn(n_shuffles):
            G = graph(xy[np.random.default_rng(s).permutation(n)], k, symmetric)
            f = fit(lik, u, G, chosen, centred, init=plains[chosen])
            gs.append(f["score"] - base)  # beta = 0 scores the same on any graph
        p = (1 + np.sum(np.array(gs) >= gain)) / (1 + n_shuffles)
    return chosen, fits, p


BETA_GRID = np.array([0., .5, 1., 2., 3., 4., 5., 6., 8., 10., 14., 20.])


def fit_profile(lik, u, A, C, centred, plain, refine=True):
    """beta by profile: every beta's fit refitted from the plain mixture (classes and w at that beta), the best score
    kept; a bounded search between the best grid point's neighbours refines it."""
    from scipy.optimize import minimize_scalar
    cache = {}
    def at(b):
        b = float(b)
        if b not in cache:
            cache[b] = fit(lik, u, A, C, centred, beta=b, init=plain)
        return cache[b]
    scores = [at(b)["score"] for b in BETA_GRID]
    i = int(np.argmax(scores))
    best = at(BETA_GRID[i])
    if refine and 0 < i < len(BETA_GRID) - 1:
        r = minimize_scalar(lambda b: -at(b)["score"], bounds=(BETA_GRID[i - 1], BETA_GRID[i + 1]), method="bounded",
                            options={"xatol": .05})
        if at(r.x)["score"] > best["score"]:
            best = at(r.x)
    return best


def analyze_profile(lik, u, xy, centred, symmetric, max_classes=3, n_shuffles=99, gain_per_track=.02, seed=0, k=8,
                    refine_null=True):
    A = graph(xy, k, symmetric)
    n = len(lik)
    plains, fits = {}, {}
    for c in range(1, max_classes + 1):
        plains[c] = fit(lik, u, A, c, centred, beta=0.)
        fits[c] = plains[c] if c == 1 else fit_profile(lik, u, A, c, centred, plains[c])
    chosen = 1
    for c in range(2, max_classes + 1):
        if (fits[c]["score"] - fits[c - 1]["score"]) / n < gain_per_track:
            break
        chosen = c
    p = None
    if chosen > 1 and n_shuffles:
        base = plains[chosen]["score"]
        gain = fits[chosen]["score"] - base
        gs = []
        for s in np.random.SeedSequence(seed).spawn(n_shuffles):
            G = graph(xy[np.random.default_rng(s).permutation(n)], k, symmetric)
            gs.append(fit_profile(lik, u, G, chosen, centred, plains[chosen], refine=refine_null)["score"] - base)
        p = (1 + np.sum(np.array(gs) >= gain)) / (1 + n_shuffles)
    return chosen, fits, p
