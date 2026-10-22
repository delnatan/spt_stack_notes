# What the diffusion stack actually does

A set of explainers for the single-particle tracking pipeline behind gemscape2. Each one reads like a blog post: it explains the idea, writes down the equations the code evaluates, shows figures made by running the real code, and names the source file and function to open next to it.

The pipeline has two packages:

```
movie ──► spotsolve.localize ──► detections (x, y, σ_x, σ_y, strength)
             (01 detection)
detections ──► spotsolve.link ──► tracks
                  (02 linking)
tracks ──► diffusionkit.gridpost ──► per-track posterior over D        (03)
       ──► diffusionkit.fbmpost  ──► per-track α and D_ref             (04)
                                     how much each step teaches, vs MSD (04b)
       ──► gridpost population models ──► distribution of D            (05)
       ──► diffusionkit.spatial  ──► where in the cell D differs       (06)
       ──► diffusionkit.classic  ──► MSD curves, for comparison        (07)
```

| # | Explainer | Built on | What is new here |
|---|---|---|---|
| 01 | [Detection and fitting](01_detection.md) | DAOPHOT FIND (Stetson 1987), u-track's `pointSourceDetection` (Aguet 2013), sCMOS shifted-Poisson likelihood (Huang 2013) | Quasi-likelihood from a fitted noise curve, efficient-score proposals for mixtures, a detection bar from Efron's two groups with a random-field null over position *and width* |
| 02 | [Linking](02_linking.md) | u-track's LAP frame-to-frame linker (Jaqaman 2008), Crocker & Grier (1996) | A per-track posterior over D in the link cost, an end cost derived from what the movie shows, an occupancy prior for immobile spots, forward/backward consensus, per-strength classes |
| 03 | [Brownian D per track](03_brownian_D.md) | Michalet & Berglund (2012), Berglund (2010), Relich et al. (2016) | Mostly standard; the novelty is in how it is used (grid posterior, reporting the floor and the grid edge) |
| 04 | [Anomalous α (fBm)](04_alpha_fbm.md) | fractional Brownian motion likelihood (e.g. Lysy et al. 2016) | Exact blurred-fGn covariance, profile likelihood over the amplitude, pooled α by summed profiles |
| 04b | [What each step teaches](04b_information.md) | Fisher information and the Cramér–Rao bound; Michalet & Berglund (2012) | Information gained per track in bits, side by side for D and α, and the MSD's efficiency against the bound |
| 05 | [Distributions of D across tracks](05_populations.md) | Hierarchical / g-modeling (Efron 2016), NPMLE, saspt (Heckert 2022) | A logistic-Gaussian-process density with its smoothness chosen by evidence, sampled by Laplace-preconditioned HMC |
| 06 | [Spatial classes](06_spatial_classes.md) | Hidden Potts / Markov random fields (Zhang 1992; Celeux, Forbes & Peyrard 2003) | β by the exact EM step with a pre-computed Potts agreement table, the number of classes by how well neighbours predict a track's own data, a shuffle null for "is there a pattern", neighbour-only grouping |
| 07 | [The MSD comparison](07_msd_comparison.md) | Qian et al. (1991), Michalet (2010) | Nothing; it is there as the familiar yardstick |

Each explainer ends with a "standard or new?" table. Where I say something "appears new", I mean I know of no published method that does exactly this. Please check those against the literature before we claim them.

## Things worth your attention

These are the places where reading the code against the figures turned up something. The first was a real defect and is fixed in diffusionkit. The others are behaviours to know about, not bugs.

1. **The spatial-class fit could get stuck and pick the wrong number of classes** ([06](06_spatial_classes.md#5-how-the-old-fit-broke-and-what-changed)). The cause is the Potts model's known phase transition: the old code learned β without the normalising constant, so β ran past it. [diffusionkit PR #6](https://github.com/delnatan/diffusionkit/pull/6) (merged, superseding PR #4) drops the class weights and fits β by its exact EM step; it picks the right number of classes on all 14 simulated two- and three-region cells.
2. **The detection bar's noise model under-predicts noise maxima** by about 1.35–1.5× near z = 3–4 on Gaussian noise with the width search, and by more on Poisson noise ([01](01_detection.md#how-good-is-the-null)). The effect is that P(real) reads slightly optimistic, so u sits a little low.
3. **The linker's threshold acts as a hard radius that grows more severe with D**, so fast tracks are cut and D read from linked tracks is biased low ([02](02_linking.md#the-price-of-a-hard-threshold)).
4. **Short tracks near the localization floor under-cover at a fixed true D** ([03](03_brownian_D.md#calibration)). This is the flat prior in ln D at work, not a bug.
5. **Small documentation mismatches.**
   - `docs/gridpost.md` still describes an eigen-whitening that the Brownian likelihood no longer uses; the code uses an LDLᵀ recursion. The eigen-whitening is in `fbmpost`.
   - The spotsolve package docstring still says linking is "by least squared displacement".
   - `prefilter::false_rate`'s comment calls the width coordinate τ = log s; the exact metric coordinate is ½ log v.

## Re-running the figures

Every figure is made by a short script in `scripts/` that calls the real packages (spotsolve 0.6.0, diffusionkit main at e49cb19). The scripts were run from the `scripts/` folder:

```bash
cd scripts
python fig_01_pipeline.py      # writes ../figures/fig_01_pipeline.png
```

Where a script re-implements a step in numpy (the screen in fig 01, the track posterior in fig 02), it says so at the top. That code is there only to draw an internal quantity the package does not expose.
