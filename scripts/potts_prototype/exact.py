"""Prototype: a hidden Potts model with no class weights, beta fitted by the exact EM M-step.

Prior P(z | beta) = exp(beta S(z)) / Z(beta) on the symmetric kNN graph, S(z) the number of edges whose ends share
a class. The EM M-step for beta solves E_beta[S] = E_q[S]: the prior's expected agreement equals the posterior's.
E_beta[S] depends only on the graph, so it is tabulated once per graph by Swendsen-Wang (`agreement_table`), and
shuffling positions permutes nodes without changing the graph, so the shuffles reuse it.

E-step: Gibbs on the posterior (`estep="gibbs"`, colour-blocked) or mean field (`estep="mf"`). Classes: normals in
ln D refitted from the tracks' grid posteriors, as in diffusionkit.spatial.potts.
"""
import numpy as np
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components
from diffusionkit.spatial import potts

MIN_SD, TINY = potts.MIN_SD, 1e-300
BETA_GRID = np.linspace(0, 1.5, 121)


def graph(xy, k=8):
    nb = potts.knn(xy, k)
    n = len(xy)
    A = csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))
    A = ((A + A.T) > 0).astype(float).tocsr()
    E = triu(A, 1).tocoo()
    return A, np.column_stack([E.row, E.col])


def colouring(A):
    """Greedy colouring: nodes of one colour share no edge, so a Gibbs sweep updates each colour at once."""
    n = A.shape[0]
    col = -np.ones(n, int)
    for i in np.argsort(-np.diff(A.indptr)):
        used = set(col[A.indices[A.indptr[i]:A.indptr[i + 1]]])
        c = 0
        while c in used:
            c += 1
        col[i] = c
    return [np.flatnonzero(col == c) for c in range(col.max() + 1)]


def agreement_table(A, edges, C, grid=BETA_GRID, burn=50, sweeps=100, seed=0):
    """E_beta[S] / |E| on `grid` for C classes with no data, by Swendsen-Wang (it crosses the ordered phase)."""
    rng = np.random.default_rng(seed)
    n, m = A.shape[0], len(edges)
    out = np.empty(len(grid))
    z = rng.integers(C, size=n)
    for g, b in enumerate(grid):
        p_bond = 1 - np.exp(-b)
        acc = 0.
        for s in range(burn + sweeps):
            same = z[edges[:, 0]] == z[edges[:, 1]]
            if s >= burn:
                acc += same.mean()
            bond = same & (rng.random(m) < p_bond)
            e = edges[bond]
            G = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
            nc, lab = connected_components(G, directed=False)
            z = rng.integers(C, size=nc)[lab]
        out[g] = acc / sweeps
    return np.maximum.accumulate(out)  # E_beta[S] rises with beta; MC noise must not make it fall


def beta_for(agree, table, grid=BETA_GRID):
    """The beta whose prior expected agreement is `agree` (the M-step), clipped to the grid."""
    if agree <= table[0]:
        return 0.
    return float(np.interp(agree, table, grid))


def _softmax(x):
    p = np.exp(x - x.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def fit(lik, u, A, edges, colours, C, table, beta=None, estep="gibbs", iters=60, sweeps=10, seed=0, init=None):
    """EM with the exact beta step (beta=None) or at a fixed beta. Returns the classes, q, beta and E_q[S]."""
    rng = np.random.default_rng(seed)
    n = len(lik)
    if init is None:
        centre, sd, _ = potts.initial(lik, u, C)
    else:
        centre, sd = init["centre"].copy(), init["sd"].copy()
    b = 0. if beta is None else beta
    q = np.full((n, C), 1 / C)
    z = None
    betas = []
    for it in range(iters):
        p = potts.densities(u, centre, sd)
        ev = lik @ p + TINY
        lev = np.log(ev)
        if C == 1:
            q = np.ones((n, 1)); agree = 1.
        elif estep == "mf":
            for _ in range(5):
                q = _softmax(lev + b * (A @ q))
            agree = float(np.mean(np.sum(q[edges[:, 0]] * q[edges[:, 1]], axis=1)))
        else:
            if z is None:
                z = q.argmax(1) if init is None else init["q"].argmax(1)
            qs, ag = np.zeros((n, C)), 0.
            for s in range(sweeps):
                for G in colours:
                    cnt = A[G] @ np.eye(C)[z]
                    pr = _softmax(lev[G] + b * cnt)
                    z[G] = (rng.random((len(G), 1)) > np.cumsum(pr, 1)).sum(1).clip(max=C - 1)
                    if s >= sweeps // 2:
                        qs[G] += pr  # Rao-Blackwellized class probabilities
                if s >= sweeps // 2:
                    ag += np.mean(z[edges[:, 0]] == z[edges[:, 1]])
            q = qs / (sweeps - sweeps // 2)
            agree = ag / (sweeps - sweeps // 2)
        mass = (lik.T @ (q / ev)) * p
        tot = mass.sum(0) + TINY
        centre = u @ mass / tot
        sd = np.sqrt(np.maximum((u ** 2) @ mass / tot - centre ** 2, 0) + MIN_SD ** 2)
        if beta is None and C > 1:
            b = beta_for(agree, table)
        betas.append(b)
    if beta is None and estep == "gibbs" and C > 1:
        b = float(np.mean(betas[iters // 2:]))  # Monte Carlo EM: average the second half
    p = potts.densities(u, centre, sd)
    ev = lik @ p + TINY
    o = np.argsort(centre)
    return dict(centre=centre[o], sd=sd[o], q=q[:, o], ev=ev[:, o], beta=b, agree=agree, betas=betas)


def score(f, A):
    """Neighbour-predictive log-likelihood: each track's data under the class its neighbours' q predict."""
    pr = _softmax(f["beta"] * (A @ f["q"]))
    return float(np.sum(np.log((pr * f["ev"]).sum(1) + TINY)))


def analyze(lik, u, xy, max_classes=4, n_shuffles=19, gain_per_track=.02, estep="gibbs", k=8, seed=0):
    A, edges = graph(xy, k)
    colours = colouring(A)
    n = len(lik)
    tables = {c: agreement_table(A, edges, c) for c in range(2, max_classes + 1)}
    plains, fits, scores, starts = {}, {}, {}, {}
    for c in range(1, max_classes + 1):
        plains[c] = fit(lik, u, A, edges, colours, c, tables.get(c), beta=0., estep="mf", iters=100)
        if c == 1:
            fits[c], starts[c] = plains[c], None
        else:
            # Starts: the plain mixture's classes, and each class of the C - 1 fit split in two (a split move);
            # the fit with the best score is kept, and the shuffles repeat it from the same start.
            cands = [dict(centre=plains[c]["centre"], sd=plains[c]["sd"], q=plains[c]["q"])]
            prev = fits[c - 1]
            for j in range(c - 1):
                cen = np.concatenate([np.delete(prev["centre"], j), prev["centre"][j] + np.array([-.5, .5]) * prev["sd"][j]])
                sd = np.concatenate([np.delete(prev["sd"], j), [prev["sd"][j] / np.sqrt(2)] * 2])
                cands.append(dict(centre=cen, sd=sd, q=np.full((n, c), 1 / c)))
            best = None
            for st in cands:
                f = fit(lik, u, A, edges, colours, c, tables[c], estep=estep, init=st)
                if best is None or score(f, A) > score(best, A):
                    best, starts[c] = f, st
            fits[c] = best
        scores[c] = score(fits[c], A)
    chosen = 1
    for c in range(2, max_classes + 1):
        if (scores[c] - scores[c - 1]) / n < gain_per_track:
            break
        chosen = c
    p = None
    if chosen > 1 and n_shuffles:
        base = score(plains[chosen], A)  # beta = 0: no graph in it
        gain = scores[chosen] - base
        gs = []
        for s in np.random.SeedSequence(seed).spawn(n_shuffles):
            perm = np.random.default_rng(s).permutation(n)
            # The shuffled graph is the same graph with its nodes permuted: carry the table over, permute the tracks.
            st = starts[chosen]
            f = fit(lik[perm], u, A, edges, colours, chosen, tables[chosen], estep=estep, init=st | {"q": st["q"][perm]})
            gs.append(score(f, A) - base)
        p = (1 + np.sum(np.array(gs) >= gain)) / (1 + n_shuffles)
    return chosen, fits, scores, p, tables
