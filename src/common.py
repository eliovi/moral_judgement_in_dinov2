"""Shared pieces: where cached features live, how to load them, the probe, and the statistics.

Cache layout (written by extract.py, read by every experiment):

    <cache>/features/smid_labels.csv          image id + mean valence / arousal / morality ratings, in feature row order
    <cache>/features/{S,B,L,G}_smid.npy       DINOv2 features, [n_layers, n_images, 2 x width], float16
    <cache>/features_extra/lowlevel.npy       57 low-level image statistics per image
    <cache>/features_extra/sup_logits.npy     ImageNet logits of a supervised ViT-B/16 (1,000)
    <cache>/features_extra/sup_feats.npy      that ViT's final CLS embedding (768)
    <cache>/features_extra/clip_img.npy       CLIP ViT-L/14 image embeddings, L2-normalised (768)
    <cache>/coco_features/coco_dinov2G_L37.npy  DINOv2-g layer-37 features of the 5,000 COCO val2017 photos
    <cache>/coco_features/coco_clip.npy         CLIP image embeddings of the same photos
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

MODELS = {"S": "facebook/dinov2-small", "B": "facebook/dinov2-base",
          "L": "facebook/dinov2-large", "G": "facebook/dinov2-giant"}
MODEL_NAMES = {"S": "ViT-S/14 (21M)", "B": "ViT-B/14 (86M)", "L": "ViT-L/14 (300M)", "G": "ViT-g/14 (1.1B)"}
BEST_LAYER = 36          # 0-based index into DINOv2-g's 40 layers, i.e. layer 37 (90% depth)
ALPHAS = np.logspace(-1, 5, 13)
BRIGHTNESS, CONTRAST, EDGE_DENSITY, COLORFULNESS = 12, 13, 14, 16   # columns of lowlevel.npy


# ----------------------------------------------------------------------------------------------------- loading
class Cache:
    def __init__(self, root):
        self.root = Path(root)
        self.features = self.root / "features"
        self.extra = self.root / "features_extra"
        self.coco = self.root / "coco_features"

    def labels(self):
        return pd.read_csv(self.features / "smid_labels.csv")

    def dinov2(self, model="G", layer=BEST_LAYER):
        """One layer as float32 [n_images, dim]; layer=None returns the memory-mapped [n_layers, n_images, dim] array."""
        F = np.load(self.features / f"{model}_smid.npy", mmap_mode="r")
        return F if layer is None else np.asarray(F[layer], dtype=np.float32)

    def extra_features(self, name):
        return np.load(self.extra / f"{name}.npy").astype(np.float32)

    def controls(self):
        """Low-level statistics + ImageNet logits + supervised-ViT features, side by side."""
        return np.hstack([self.extra_features(k) for k in ("lowlevel", "sup_logits", "sup_feats")])

    def clip(self):
        return np.load(self.extra / "clip_img.npy").astype(np.float64)

    def coco_pair(self):
        return (np.load(self.coco / "coco_dinov2G_L37.npy").astype(np.float32),
                np.load(self.coco / "coco_clip.npy").astype(np.float64))


def load_metadata(path, labels):
    """SMID's metadata.csv (per-image SEs, rater counts, foundation ratings), aligned to the feature row order."""
    meta = pd.read_csv(path).assign(image=lambda d: d.file_name.str.replace(r"\.[^.]+$", "", regex=True))
    meta = labels[["image"]].merge(meta, on="image", how="left")
    assert np.allclose(meta.moral_mean, labels.morality), "metadata does not match the cached labels"
    return meta


# ----------------------------------------------------------------------------------------------------- targets
def spline_fit(y, x):
    """Cubic-spline regression of y on x. Returns (fitted values, local slope dy/dx)."""
    m = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(x[:, None], y)
    eps = 1e-3
    slope = (m.predict(x[:, None] + eps) - m.predict(x[:, None] - eps)) / (2 * eps)
    return m.predict(x[:, None]), slope


def targets(labels):
    """The three ratings we study, plus morality beyond valence (morality minus a spline fit on valence)."""
    val, mor = labels.valence.to_numpy(float), labels.morality.to_numpy(float)
    return {"valence": val, "morality": mor, "morality_beyond_valence": mor - spline_fit(mor, val)[0]}


# ----------------------------------------------------------------------------------------------------- probe
def probe(k=256):
    """Standardise -> PCA(k) -> ridge with the penalty chosen by internal cross-validation."""
    return make_pipeline(StandardScaler(), PCA(k, random_state=0), RidgeCV(alphas=ALPHAS))


def oof(X, y, k=256, folds=5, seed=0, groups=None):
    """Out-of-fold predictions: every image is scored by a probe that never saw it.
    y may be 1-D or 2-D. With groups, whole groups are held out together (GroupKFold)."""
    split = (GroupKFold(folds).split(X, y, groups) if groups is not None
             else KFold(folds, shuffle=True, random_state=seed).split(X))
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in split:
        pred[te] = probe(min(k, X.shape[1], len(tr) - 1)).fit(X[tr], y[tr]).predict(X[te])
    return pred


def residualize(y, Z):
    """The part of y that Z cannot predict (cross-fitted, so it never removes signal by overfitting)."""
    return y - oof(Z, y)


def fit_map(X, Y, k=256):
    """Linear translation X -> Y (standardise, PCA when X is wide, multi-output ridge)."""
    steps = [StandardScaler()] + ([PCA(k, random_state=0)] if X.shape[1] > k else []) + [RidgeCV(alphas=ALPHAS)]
    return make_pipeline(*steps).fit(X, Y)


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------------------------------- statistics
_RNG = np.random.default_rng(0)


def rho(y, p):
    return round(float(spearmanr(y, p)[0]), 4)


def rho_ci(y, p, n=300):
    """Spearman rho with a 95% bootstrap interval over images."""
    boots = [spearmanr(y[i], p[i])[0] for i in (_RNG.integers(0, len(y), len(y)) for _ in range(n))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": rho(y, p), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
    print("saved", path)


def cli(description):
    """Argument parser shared by the experiment scripts."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--cache", required=True, help="folder holding features/, features_extra/, coco_features/")
    ap.add_argument("--out", default="results", help="folder for result files")
    return ap


def parse(ap):
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    return a
