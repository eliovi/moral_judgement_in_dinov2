"""Headline figure: depth and size curves after controls, S->G gain vs object content, held-out object groups.
Reads results/round3/*.csv, writes figures/fig_headline.png."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results" / "round3"
OUT = Path(__file__).resolve().parents[1] / "figures" / "fig_headline.png"
SIZES = {"S": ("ViT-S/14 (21M)", "#84b9f3"), "B": ("ViT-B/14 (86M)", "#338ced"),
         "L": ("ViT-L/14 (300M)", "#2370cd"), "G": ("ViT-g/14 (1.1B)", "#0e3d7e")}
INK, INK2, GRID, GRAY = "#191919", "#666666", "#e6e6e6", "#9c9c9c"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
L = pd.read_csv(R / "scale_depth_layers_controls.csv")
gain = pd.read_csv(R / "scale_gain_G_minus_S.csv").set_index("target")
held = pd.read_csv(R / "S3_heldout_object_groups.csv").set_index("target")

fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
fig.subplots_adjust(hspace=0.6, wspace=0.34, top=0.84, bottom=0.08, left=0.09, right=0.98)


def depth_panel(a, col, title, ylabel, ylim):
    for m, (name, c) in SIZES.items():
        d = L[L.model == m]
        a.plot(d.depth, d[col], color=c, lw=2, marker="o", ms=3.5, label=name)
    a.set_xlim(0, 1.02); a.set_ylim(*ylim); a.set_xlabel("relative depth (layer / number of layers)")
    a.set_ylabel(ylabel); a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
    a.set_title(title, loc="left", color=INK, fontsize=11)


depth_panel(ax[0, 0], "objects_pc1", "A. Object content is readable from early layers",
            "Spearman ρ (out-of-fold probe)", (0.4, 0.9))
depth_panel(ax[0, 1], "mbv_ctrl", "B. The moral signal left after controls forms late,\n    and is larger in larger models",
            "Spearman ρ (out-of-fold probe)", (-0.1, 0.22))
ax[0, 1].axhline(0, color=GRAY, lw=1)
ax[0, 0].legend(frameon=False, loc="lower right", fontsize=9, title="DINOv2 size", title_fontsize=9)

# C: gain from S to G
a = ax[1, 0]
items = [("objects_pc1", "object\ncontent"), ("valence_ctrl", "valence"),
         ("morality_ctrl", "morality"), ("mbv_ctrl", "morality\nbeyond\nvalence")]
for i, (k, lab) in enumerate(items):
    g = gain.loc[k]
    col = GRAY if k.startswith("objects") else "#2370cd"
    a.barh(i, g.gain, color=col, height=0.55)
    a.errorbar(g.gain, i, xerr=[[g.gain - g.ci_lo], [g.ci_hi - g.gain]], fmt="none", ecolor=INK, lw=1.2, capsize=3)
    a.text(g.ci_hi + 0.005, i, f"+{g.gain:.2f}", va="center", color=INK, fontsize=9)
a.set_yticks(range(len(items))); a.set_yticklabels([l for _, l in items], fontsize=9); a.invert_yaxis()
a.set_xlim(0, 0.2); a.set_xlabel("gain in ρ, S → g (95% bootstrap CI); grey = object content (top PC)")
a.grid(axis="x", color=GRID, lw=0.8); a.set_axisbelow(True)
a.set_title("C. Gain from smallest to largest model:\n    large after controls, near zero for objects", loc="left", color=INK, fontsize=11)

# D: held-out object groups
a = ax[1, 1]
tg = [("valence", "valence"), ("morality", "morality"), ("mbv", "morality\nbeyond valence")]
series = [("dinov2_random_folds", "DINOv2-g, random folds", "#84b9f3"),
          ("dinov2_heldout_groups", "DINOv2-g, unseen object groups", "#0e3d7e"),
          ("objects_heldout_groups", "object features only, unseen groups", GRAY)]
w = 0.26
for j, (k, lab, c) in enumerate(series):
    xs = [i + (j - 1) * w for i in range(len(tg))]
    vals = [held.loc[t, k] for t, _ in tg]
    a.bar(xs, vals, width=w - 0.03, color=c, label=lab)
    for x, v in zip(xs, vals):
        a.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=INK)
a.set_xticks(range(len(tg))); a.set_xticklabels([l for _, l in tg]); a.set_ylim(0, 1.15); a.set_yticks([0, .2, .4, .6, .8])
a.set_ylabel("Spearman ρ"); a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
a.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)
a.set_title("D. The direction transfers to object categories\n    the probe never saw", loc="left", color=INK, fontsize=11)

fig.suptitle("A self-supervised vision model (DINOv2) tracks human good/bad ratings of SMID images",
             x=0.07, ha="left", fontsize=13, color=INK, y=0.98)
fig.text(0.07, 0.935, "Controls removed: low-level image statistics, ImageNet object logits and supervised-ViT features. "
         "n = 2,940 images.", fontsize=9, color=INK2)
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=170)
print("saved", OUT)
