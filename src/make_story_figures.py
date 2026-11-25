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
          "Dashed line: the best score 200 runs with shuffled ratings reached on the hardest target (right, last bar).")
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


def words():
    """What DINOv2's directions mean, read through a map into CLIP that never saw SMID."""
    W = json.load(open(R / "story" / "words.json"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.04, right=0.98, wspace=0.08)
    title(fig, "What DINOv2's good/bad direction means, in words",
          "DINOv2 features translated into CLIP's image-text space with a map learned on unrated COCO photos.\n"
          "Words that best match images DINOv2 scores high (blue) and low (orange). Vocabulary: 150 virtues, vices, emotions, everyday things.")
    panels = [("morality", "Morally good vs bad",
               f"virtue words beat vice words {W['morality']['auc_virtue_vs_vice']:.0%} of the time"),
              ("morality_beyond_valence", "Morally good, beyond just pleasant",
               f"virtue words beat merely pleasant words {W['morality_beyond_valence']['auc_virtue_vs_positive_emotion']:.0%} of the time")]
    for a, (k, head, foot) in zip(ax, panels):
        a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.text(0.0, 0.98, head, fontsize=14, color=INK, weight="bold", va="top")
        a.text(0.02, 0.88, "DINOv2 scores high", fontsize=11, color=BLUE, va="top")
        a.text(0.52, 0.88, "DINOv2 scores low", fontsize=11, color="#e76624", va="top")
        for i, (hi, lo) in enumerate(zip(W[k]["top"], W[k]["bottom"])):
            y = 0.78 - i * 0.085
            a.text(0.02, y, hi, fontsize=14, color=DARK, va="top")
            a.text(0.52, y, lo, fontsize=14, color="#b4470f", va="top")
        a.text(0.0, 0.06, foot, fontsize=11, color=INK2, va="top", style="italic")
    fig.savefig(FIG / "story_words.png", dpi=170); plt.close(fig)


def shared():
    """CLIP's good/bad direction is found inside DINOv2, more so deeper and in larger models."""
    P = pd.read_csv(R / "story" / "shared_direction.csv", comment="#")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.07, right=0.97, wspace=0.3)
    title(fig, "Two very different models end up with the same good/bad direction",
          "CLIP learned from 400M captioned images; DINOv2 never saw a word. Left: CLIP's morality direction, applied to DINOv2\n"
          "through a translation learned without any ratings. Right: how much of the rating DINOv2's own 10 main axes carry.")
    a = ax[0]
    for m, (name, c) in SIZES.items():
        d = P[P.model == m]
        a.plot(d.depth, d.clip_dir_morality, color=c, lw=2.6, marker="o", ms=5)
        off = {"G": 0.028, "L": 0.0, "B": -0.01}.get(m, 0)
        a.text(1.03, d.clip_dir_morality.iloc[-1] + off, name, color=c, va="center", fontsize=11)
    a.set_xlim(0.2, 1.22); a.set_ylim(0.3, 0.8); a.set_xticks([0.25, 0.5, 0.75, 1]); a.set_xticklabels(["25%", "50%", "75%", "last layer"])
    a.set_ylabel("ρ with human morality ratings"); a.grid(axis="y", color=GRID); a.set_axisbelow(True)
    a.set_title("CLIP's direction, read inside DINOv2", loc="left", color=INK, fontsize=13)
    a = ax[1]
    best = [P[P.model == m].top10pc_valence.max() for m in SIZES]
    a.bar(range(4), best, color=[c for _, c in SIZES.values()], width=0.62)
    for i, v in enumerate(best):
        a.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=12, color=INK)
    a.set_xticks(range(4)); a.set_xticklabels([n.split(",")[0] for n, _ in SIZES.values()])
    a.set_ylim(0, 0.75); a.set_ylabel("ρ with valence ratings"); a.grid(axis="y", color=GRID); a.set_axisbelow(True)
    a.set_title("Valence in DINOv2's top 10 axes", loc="left", color=INK, fontsize=13)
    fig.savefig(FIG / "story_shared.png", dpi=170); plt.close(fig)


ORANGE = "#e76624"


def _crop(path, ratio=4 / 3):
    from PIL import Image
    im = Image.open(path).convert("RGB"); w, h = im.size
    if w / h > ratio:
        nw = int(h * ratio); im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / ratio); im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im


def hero():
    """Five SMID photos in the order DINOv2 places them, from morally bad to good."""
    P = pd.read_csv(R / "story" / "example_picks.csv")
    P = P[P.group == "hero"].sort_values("dinov2_morality_percentile")
    fig = plt.figure(figsize=(13, 4.6))
    title(fig, "A vision model that never saw a label or a word orders images from bad to good",
          "SMID photos in the order a linear probe on DINOv2 places them. Below each: the average human morality rating (1 = very bad, 5 = very good).")
    n = len(P); w = 0.94 / n
    for i, r in enumerate(P.itertuples()):
        a = fig.add_axes([0.03 + i * w, 0.27, w - 0.012, 0.5])
        a.imshow(_crop(R / "story" / "thumbs" / f"{r.image}.jpg")); a.axis("off")
        fig.text(0.03 + i * w + (w - 0.012) / 2, 0.2, f"people: {r.morality_rating:.1f}", ha="center", fontsize=12, color=INK)
    import numpy as np
    a = fig.add_axes([0.03, 0.08, 0.928, 0.04])
    a.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=__import__("matplotlib").colors.LinearSegmentedColormap.from_list("g", [ORANGE, "#d9d9d9", DARK]))
    a.set_xticks([]); a.set_yticks([]); [sp.set_visible(False) for sp in a.spines.values()]
    fig.text(0.03, 0.015, "DINOv2: morally bad", color=ORANGE, fontsize=12, weight="bold")
    fig.text(0.958, 0.015, "DINOv2: morally good", color=DARK, fontsize=12, weight="bold", ha="right")
    fig.savefig(FIG / "story_hero.png", dpi=170); plt.close(fig)


def offdiag():
    """Where morality and pleasantness disagree."""
    P = pd.read_csv(R / "story" / "example_picks.csv")
    fig = plt.figure(figsize=(13, 8.2))
    title(fig, "Morality is not just pleasantness, and DINOv2 knows the difference",
          "Images DINOv2 scores highest and lowest on morality after removing pleasantness (valence). People agree.\n"
          "Numbers are average human ratings from 1 to 5.")
    rows = [("good_but_unpleasant", "Morally good, though unpleasant", DARK, 0.42),
            ("pleasant_not_moral", "Pleasant, but not especially good", ORANGE, 0.03)]
    for g, head, col, y0 in rows:
        d = P[P.group == g]
        fig.text(0.03, y0 + 0.335, head, fontsize=14, color=col, weight="bold")
        for i, r in enumerate(d.itertuples()):
            a = fig.add_axes([0.03 + i * 0.237, y0 + 0.05, 0.225, 0.27])
            a.imshow(_crop(R / "story" / "thumbs" / f"{r.image}.jpg")); a.axis("off")
            fig.text(0.03 + i * 0.237 + 0.1125, y0 + 0.012, f"pleasant {r.valence_rating:.1f}  ·  moral {r.morality_rating:.1f}",
                     ha="center", fontsize=11.5, color=INK)
    fig.savefig(FIG / "story_offdiagonal.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    controls(); depth(); words(); shared(); hero(); offdiag()
    print("saved")
