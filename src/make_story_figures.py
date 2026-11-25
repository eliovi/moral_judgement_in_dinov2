"""Figures for the README story. Each figure makes one point.
Reads results/ CSVs (and results/story/*.json when present), writes figures/story_*.png."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R, FIG = ROOT / "results", ROOT / "figures"
INK, INK2, GRID, GRAY, BLUE, DARK = "#191919", "#666666", "#e9e9e9", "#b5b5b5", "#2370cd", "#0e3d7e"
SIZES = {"S": ("small, 21M", "#84b9f3"), "B": ("base, 86M", "#338ced"), "L": ("large, 300M", "#2370cd"), "G": ("giant, 1.1B", "#0e3d7e")}
plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def title(fig, t, sub=None):
    fig.text(0.03, 0.95, t, fontsize=16, color=INK, weight="bold", va="top")
    if sub:
        fig.text(0.03, 0.865, sub, fontsize=11.5, color=INK2, va="top")


def controls():
    """What DINOv2 still predicts as we take explanations away."""
    steps = ["everything", "minus brightness,\ncolour, texture", "minus objects", "minus a supervised\nimage classifier"]
    mor = [0.7353, 0.699, 0.352, 0.321]          # results/controls_and_language/beyond_controls.csv (+ probe on raw rating)
    mbv = [0.609, 0.579, 0.191, 0.163]          # results/round3: T1 + morality_diagnostics
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.06, right=0.98, wspace=0.12)
    title(fig, "Not just dark images or scary objects",
          "How well a linear probe on DINOv2 predicts people's ratings (Spearman ρ, held-out images), as we remove explanations one by one.\n"
          "The dashed line is the best score shuffled ratings reached in 200 tries.")
    for a, vals, name in [(ax[0], mor, "Morality rating"), (ax[1], mbv, "Morality, with valence removed")]:
        cols = [DARK] + [BLUE] * 3
        a.bar(range(4), vals, color=cols, width=0.62)
        for i, v in enumerate(vals):
            a.text(i, v + 0.015, f"{v:.2f}", ha="center", color=INK, fontsize=12)
        a.axhline(0.048, color=INK2, lw=1, ls="--")
        a.set_xticks(range(4)); a.set_xticklabels(steps, fontsize=10.5)
        a.set_title(name, loc="left", color=INK, fontsize=13)
        a.set_ylim(0, 0.85); a.grid(axis="y", color=GRID); a.set_axisbelow(True)
    ax[0].set_ylabel("ρ with human ratings")
    fig.savefig(FIG / "story_controls.png", dpi=170); plt.close(fig)


def depth():
    """The leftover moral signal appears late and is larger in larger models."""
    L = pd.read_csv(R / "round3" / "scale_depth_layers_controls.csv")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.09, right=0.8)
    title(fig, "It appears only in the last layers, and grows with model size",
          "DINOv2's prediction of the morality rating after removing valence, image statistics and objects, layer by layer.")
    ax.axvspan(0.66, 1.0, color="#f3f7fd", zorder=0)
    for m, (name, c) in SIZES.items():
        d = L[L.model == m]
        ax.plot(d.depth, d.mbv_ctrl, color=c, lw=2.6)
        off = {"B": -0.011, "L": 0.009}.get(m, 0)
        ax.text(1.015, d.mbv_ctrl.iloc[-1] + off, name, color=c, va="center", fontsize=11.5, weight="bold" if m == "G" else None)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(-0.09, 0.21)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["first layer", "25%", "50%", "75%", "last layer"])
    ax.set_ylabel("ρ with human ratings"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.text(0.68, 0.195, "last third of the network", color=INK2, fontsize=10.5)
    fig.savefig(FIG / "story_depth_scale.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    controls(); depth()
    print("saved")
