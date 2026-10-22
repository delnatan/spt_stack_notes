import sys, time, pickle
sys.path.insert(0, '/tmp/claude-0/s')
import numpy as np, exact as X
from diffusionkit.gridpost import GridPostOptions
u = GridPostOptions().u_D()
seed, patterned, estep = int(sys.argv[1]), sys.argv[2] == 'p', sys.argv[3]
lik, xy, inside = pickle.load(open(f'movie_{seed}.pkl', 'rb'))
if not patterned:
    xy = xy[np.random.default_rng(1000 + seed).permutation(len(xy))]
ll = lik.loglik_D; L = np.exp(ll - ll.max(1, keepdims=True))
t = time.time()
ch, fits, scores, p, _ = X.analyze(L, u, xy, estep=estep)
f = fits[ch]
near = 1 + (np.abs(f['centre'] - np.log(.2)) < np.abs(f['centre'] - np.log(.05)))
acc = float(np.mean(near[f['q'].argmax(1)] == np.where(inside, 1, 2))) if patterned else None
print(estep, seed, 'pattern' if patterned else 'none', dict(C=ch, beta=round(f['beta'], 3), p=p, acc=acc,
      D=np.round(np.exp(f['centre']), 3).tolist(), gains=[round((scores[c] - scores[c-1]) / len(L), 4) for c in range(2, 5)]), f'{time.time()-t:.0f}s', flush=True)
