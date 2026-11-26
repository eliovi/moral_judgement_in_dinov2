"""Draw the six figures in figures/ from the result files in results/. Each figure makes one point.

  python src/figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, THUMBS = ROOT / "results", ROOT / "figures", ROOT / "results" / "examples" / "thumbs"
INK, INK2, GRID, GRAY = "#191919", "#666666", "#e9e9e9", "#b5b5b5"
BLUE, DARK, ORANGE, DARK_ORANGE = "#2370cd", "#0e3d7e", "#e76624", "#b4470f"
SIZES = {"S": ("small, 21M", "#84b9f3"), "B": ("base, 86M", "#338ced"), "L": ("large, 300M", "#2370cd"), "G": ("giant, 1.1B", "#0e3d7e")}
plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
load = lambda name: json.load(open(RES / name))


def title(fig, text, sub=None):
    fig.text(0.03, 0.95, text, fontsize=16, color=INK, weight="bold", va="top")
    if sub:
        fig.text(0.03, 0.865, sub, fontsize=11.5, color=INK2, va="top")


def save(fig, name):
    fig.savefig(FIG / name, dpi=170); plt.close(fig)
    print("saved", FIG / name)


def crop(image_id, ratio=4 / 3):
    im = Image.open(THUMBS / f"{image_id}.jpg").convert("RGB"); w, h = im.size
    if w / h > ratio:
        nw = int(h * ratio); return im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = int(w / ratio); return im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def fig1_strip(picks):
    """Five SMID photos in the order DINOv2 places them, from morally bad to good."""
    P = picks[picks.figure == "strip"].sort_values("dinov2_morality_percentile")
    fig = plt.figure(figsize=(13, 4.6))
    title(fig, "A vision model that never saw a label or a word orders images from bad to good",
          "SMID photos in the order a linear probe on DINOv2 places them. Below each: the average human morality rating (1 = very bad, 5 = very good).")
    w = 0.94 / len(P)
    for i, r in enumerate(P.itertuples()):
        ax = fig.add_axes([0.03 + i * w, 0.27, w - 0.012, 0.5]); ax.imshow(crop(r.image)); ax.axis("off")
        fig.text(0.03 + i * w + (w - 0.012) / 2, 0.2, f"people: {r.morality_rating:.1f}", ha="center", fontsize=12, color=INK)
    ax = fig.add_axes([0.03, 0.08, 0.928, 0.04])
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bad_good", [ORANGE, "#d9d9d9", DARK]))
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    fig.text(0.03, 0.015, "DINOv2: morally bad", color=ORANGE, fontsize=12, weight="bold")
    fig.text(0.958, 0.015, "DINOv2: morally good", color=DARK, fontsize=12, weight="bold", ha="right")
    save(fig, "fig1_bad_to_good.png")


def fig2_controls():
    """What DINOv2 still predicts as explanations are removed one at a time."""
    lin, perm = load("e2_shortcuts.json")["linear_controls"], load("e3_beyond_valence.json")["permutation"]
    steps = ["none", "low-level", "+ ImageNet logits", "+ supervised-ViT features"]
    labels = ["everything", "minus brightness,\ncolour, texture", "minus objects", "minus a supervised\nimage classifier"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.06, right=0.98, wspace=0.12)
    title(fig, "Not just dark images or scary objects",
          "How well a linear probe on DINOv2 predicts people's ratings (Spearman ρ, held-out images), as we remove explanations one by one.\n"
          "Dashed line: the best score 200 runs with shuffled ratings reached on the hardest target (right, last bar).")
    for ax, target, name in [(axes[0], "morality", "Morality rating"),
                             (axes[1], "morality_beyond_valence", "Morality, with valence removed")]:
        vals = [lin[target][s]["dinov2"]["rho"] for s in steps]
        ax.bar(range(4), vals, color=[DARK] + [BLUE] * 3, width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.2f}", ha="center", color=INK, fontsize=12)
        ax.axhline(perm["null_max"], color=INK2, lw=1, ls="--")
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10.5)
        ax.set_title(name, loc="left", color=INK, fontsize=13)
        ax.set_ylim(0, 0.85); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    axes[0].set_ylabel("ρ with human ratings")
    save(fig, "fig2_controls.png")


def fig3_beyond_pleasant(picks):
    """Where morality and pleasantness disagree."""
    fig = plt.figure(figsize=(13, 8.2))
    title(fig, "Morality is not just pleasantness, and DINOv2 knows the difference",
          "Images DINOv2 scores highest and lowest on morality after removing pleasantness (valence). People agree.\n"
          "Numbers are average human ratings from 1 to 5.")
    for group, head, col, y0 in [("good_but_unpleasant", "Morally good, though unpleasant", DARK, 0.42),
                                 ("pleasant_not_moral", "Pleasant, but not especially good", ORANGE, 0.03)]:
        fig.text(0.03, y0 + 0.335, head, fontsize=14, color=col, weight="bold")
        for i, r in enumerate(picks[picks.figure == group].itertuples()):
            ax = fig.add_axes([0.03 + i * 0.237, y0 + 0.05, 0.225, 0.27]); ax.imshow(crop(r.image)); ax.axis("off")
            fig.text(0.03 + i * 0.237 + 0.1125, y0 + 0.012, f"pleasant {r.valence_rating:.1f}  ·  moral {r.morality_rating:.1f}",
                     ha="center", fontsize=11.5, color=INK)
    save(fig, "fig3_beyond_pleasant.png")


def fig4_words():
    """What DINOv2's directions mean, read through a translation into CLIP that never saw SMID."""
    W = load("e4_words.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.04, right=0.98, wspace=0.08)
    title(fig, "What DINOv2's good/bad direction means, in words",
          "DINOv2 features translated into CLIP's image-text space with a map learned on unrated COCO photos.\n"
          "Words that best match images DINOv2 scores high (blue) and low (orange). Vocabulary: 150 virtues, vices, emotions, everyday things.")
    for ax, key, head, foot in [
            (axes[0], "morality", "Morally good vs bad",
             f"virtue words beat vice words {W['morality']['auc']['virtue > vice']:.0%} of the time"),
            (axes[1], "morality_beyond_valence", "Morally good, beyond just pleasant",
             f"virtue words beat merely pleasant words {W['morality_beyond_valence']['auc']['virtue > positive emotion']:.0%} of the time")]:
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.0, 0.98, head, fontsize=14, color=INK, weight="bold", va="top")
        ax.text(0.02, 0.88, "DINOv2 scores high", fontsize=11, color=BLUE, va="top")
        ax.text(0.52, 0.88, "DINOv2 scores low", fontsize=11, color=ORANGE, va="top")
        for i, (hi, lo) in enumerate(zip(W[key]["top"][:8], W[key]["bottom"][:8])):
            ax.text(0.02, 0.78 - i * 0.085, hi[0], fontsize=14, color=DARK, va="top")
            ax.text(0.52, 0.78 - i * 0.085, lo[0], fontsize=14, color=DARK_ORANGE, va="top")
        ax.text(0.0, 0.06, foot, fontsize=11, color=INK2, va="top", style="italic")
    save(fig, "fig4_words.png")


def fig5_depth_scale():
    """The moral signal left after controls appears late and is larger in larger models."""
    L = pd.read_csv(RES / "e5_depth_scale_layers.csv")
    L = L[L.target == "mbv_ctrl"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.09, right=0.8)
    title(fig, "It appears only in the last layers, and grows with model size",
          "DINOv2's prediction of the morality rating after removing valence, image statistics and objects, layer by layer.")
    ax.axvspan(0.66, 1.0, color="#f3f7fd", zorder=0)
    for m, (name, c) in SIZES.items():
        d = L[L.model == m].sort_values("depth")
        ax.plot(d.depth, d.rho, color=c, lw=2.6, marker="o", ms=4)
        ax.text(1.015, d.rho.iloc[-1] + {"B": -0.011, "L": 0.009}.get(m, 0), name, color=c, va="center", fontsize=11.5,
                weight="bold" if m == "G" else None)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(-0.09, 0.21)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["first layer", "25%", "50%", "75%", "last layer"])
    ax.set_ylabel("ρ with human ratings"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.text(0.68, 0.195, "last third of the network", color=INK2, fontsize=10.5)
    save(fig, "fig5_depth_scale.png")


def fig6_shared():
    """CLIP's morality direction is found inside DINOv2, more so deeper and in larger models."""
    P = pd.DataFrame(load("e6_shared.json")["transfer_by_size_depth"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.07, right=0.97, wspace=0.3)
    title(fig, "Two very different models end up with the same good/bad direction",
          "CLIP learned from 400M captioned images; DINOv2 never saw a word. Left: CLIP's morality direction, applied to DINOv2\n"
          "through a translation learned without any ratings. Right: morality from DINOv2's own 10 main axes, found without labels.")
    ax = axes[0]
    for m, (name, c) in SIZES.items():
        d = P[P.model == m]
        ax.plot(d.depth, d.clip_direction_morality, color=c, lw=2.6, marker="o", ms=5)
        ax.text(1.03, d.clip_direction_morality.iloc[-1] + {"G": 0.028, "B": -0.01}.get(m, 0), name, color=c, va="center", fontsize=11)
    ax.set_xlim(0.2, 1.22); ax.set_ylim(0.3, 0.8)
    ax.set_xticks([0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["25%", "50%", "75%", "last layer"])
    ax.set_ylabel("ρ with human morality ratings"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_title("CLIP's direction, read inside DINOv2", loc="left", color=INK, fontsize=13)
    ax = axes[1]
    best = [P[P.model == m].top10pc_morality.max() for m in SIZES]
    ax.bar(range(4), best, color=[c for _, c in SIZES.values()], width=0.62)
    for i, v in enumerate(best):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=12, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([n.split(",")[0] for n, _ in SIZES.values()])
    ax.set_ylim(0, 0.75); ax.set_ylabel("ρ with human morality ratings"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_title("Morality in DINOv2's top 10 axes", loc="left", color=INK, fontsize=13)
    save(fig, "fig6_shared_direction.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    picks = pd.read_csv(RES / "examples" / "picks.csv")
    fig1_strip(picks); fig2_controls(); fig3_beyond_pleasant(picks); fig4_words(); fig5_depth_scale(); fig6_shared()
