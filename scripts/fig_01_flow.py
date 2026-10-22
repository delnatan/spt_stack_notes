"""Figure 1.0: the detection flow (localize_stack, localize_with, fit_seed), drawn as a compact two-panel diagram."""
from matplotlib.patches import FancyBboxPatch

from common import C, plt, save

fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")


def box(x, y, w, h, text, fc="#EEF3FA", ec=C["blue"], size=7.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, linespacing=1.25)


def arrow(x0, y0, x1, y1, color=C["grey"], style="-|>", ls="-"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle=style, color=color, lw=1.1, ls=ls,
                                                        shrinkA=0, shrinkB=0))


# Left: once per movie
ax.add_patch(FancyBboxPatch((0.1, 0.15), 4.3, 4.25, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#FBF7EC",
                            ec=C["orange"], lw=1))
ax.text(0.25, 4.22, "Once per movie  ·  localize_stack", fontsize=8.5, weight="bold", va="center")
left = [
    ("Noise curve  V = read + gain·m\nfrom quiet pixels of every frame", 3.45),
    ("Pass 1: every frame, single fits\nat the low bar z ≥ 2 → candidate heights", 2.6),
    ("Two groups: f0 (random-field null) + f1 (fitted)\nbar u = least z with P(real | z) ≥ 0.9", 1.75),
    ("Pass 2 (mixtures): refit every frame at u\nno mixtures: keep pass-1 fits with z ≥ u", 0.9),
]
for i, (t, y) in enumerate(left):
    box(0.35, y - 0.3, 3.8, 0.62, t)
    if i:
        arrow(2.25, left[i - 1][1] - 0.3, 2.25, y + 0.32)
ax.text(2.25, 0.32, "Pass 1 and pass 2 each run the per-frame flow →", ha="center", fontsize=7, color=C["grey"],
        style="italic")

# Right: per frame
ax.add_patch(FancyBboxPatch((4.75, 0.15), 7.15, 4.25, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#F3F8F4",
                            ec=C["green"], lw=1))
ax.text(4.9, 4.22, "Per frame  ·  localize_with", fontsize=8.5, weight="bold", va="center")
box(5.0, 3.15, 2.6, 0.75, "Screen: matched-filter z\nmask z ≥ 2, 8-connected labels\n−LoG maxima → seeds")
# per-seed loop
ax.add_patch(FancyBboxPatch((8.0, 0.75), 3.7, 3.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white",
                            ec=C["green"], lw=1, ls="--"))
ax.text(9.85, 3.78, "For each seed, independently, once  ·  fit_seed", ha="center", fontsize=7.6, weight="bold",
        color=C["green"])
seed_steps = [
    ("Window ±⌈4σ⌉ around the seed\n(other mask components left out)", 3.25),
    ("Forward: add where the efficient score peaks;\nkeep while 2ΔQ ≥ u²", 2.55),
    ("Backward: drop this seed's weakest\nwhile it costs < u²/2 nats", 1.85),
    ("Widths: out-of-focus vs at bound;\nmotion-blur ellipse (optional)", 1.2),
]
for i, (t, y) in enumerate(seed_steps):
    box(8.2, y - 0.27, 3.3, 0.55, t, size=7.2)
    if i:
        arrow(9.85, seed_steps[i - 1][1] - 0.27, 9.85, y + 0.28)
arrow(7.6, 3.52, 8.2, 3.25)
box(5.0, 1.85, 2.6, 0.75, "Owned: each component reported\nonly by its nearest seed", size=7.4)
box(5.0, 0.55, 2.6, 0.95, "Merge reports from two windows\nwithin 1σ (each the other's nearest)\n→ positions, SEs, detect_z, P(real)",
    size=7.2)
arrow(8.2, 1.2, 7.6, 2.1)
arrow(6.3, 1.85, 6.3, 1.5)
ax.text(6.3, 2.85, "neighbours are extra components\ninside a window, not a second loop", ha="center", fontsize=6.8,
        color=C["grey"], style="italic")
arrow(4.15, 2.6, 5.0, 3.4, color=C["orange"], ls="--")
arrow(4.15, 0.9, 5.0, 3.3, color=C["orange"], ls="--")
save(fig, "fig_01_flow.png")
