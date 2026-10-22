"""The simulated movie of explainer 02, shared by its figures: immobile, slow and fast particles in a box,
each seen with its own probability, with known identities to score links against."""
import numpy as np
import polars as pl

L, T, SE = 128.0, 60, 0.3
CLASSES = {"immobile": (40, 0.0), "slow": (60, 0.3), "fast": (30, 3.0)}   # (count, D px²/frame)
COLORS = {"immobile": "grey", "slow": "blue", "fast": "red"}


def movie(seed=11, crowd=1):
    """`crowd` multiplies every class's count in the same box."""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for name, (n, D) in CLASSES.items():
        for _ in range(n * crowd):
            p = rng.uniform(5, L - 5, 2)
            pdet = rng.uniform(0.6, 0.99)
            for t in range(T):
                if t:
                    p = p + rng.normal(0, np.sqrt(2 * D), 2)
                    p = np.abs(p); p = L - np.abs(L - p)      # reflect at the edges
                if rng.random() < pdet:
                    o = p + rng.normal(0, SE, 2)
                    rows.append((t, o[0], o[1], SE, SE, pid, name))
            pid += 1
    return pl.DataFrame(rows, schema=["frame", "y", "x", "se_y", "se_x", "particle", "cls"], orient="row")


def rms_step(D):
    """The rms 2-D step, localization error included."""
    return np.sqrt(4 * D + 4 * SE ** 2)


def links(locs, tid):
    """Every one-frame link the track ids make: the later row's class, link_p if given, the squared step `r2`,
    and whether it joins one particle's detections."""
    df = locs.with_columns(pl.Series("tid", tid)).sort("tid", "frame")
    prev = lambda c: pl.col(c).shift(1).over("tid")  # noqa: E731
    df = df.with_columns(prev("frame").alias("f0"), prev("particle").alias("p0"),
                         ((pl.col("y") - prev("y")) ** 2 + (pl.col("x") - prev("x")) ** 2).alias("r2"))
    df = df.filter(pl.col("f0") == pl.col("frame") - 1)
    return df.with_columns((pl.col("particle") == pl.col("p0")).alias("right"))


def score(locs, tid):
    """Per class: recall of true one-frame links and identity switches per 100 links made."""
    df = locs.with_columns(pl.Series("tid", tid)).sort("particle", "frame")
    true = df.with_columns(pl.col("frame").shift(-1).over("particle").alias("f2"),
                           pl.col("tid").shift(-1).over("particle").alias("t2"))
    true = true.filter(pl.col("f2") == pl.col("frame") + 1)
    made = links(locs, tid)
    out = {}
    for name in CLASSES:
        tl = true.filter(pl.col("cls") == name)
        lk = made.filter(pl.col("cls") == name)
        out[name] = (float((tl["tid"] == tl["t2"]).mean()), 100 * float((~lk["right"]).mean()))
    return out
