"""Test 1: how well does each layer of each DINOv2 size predict the human ratings?

For every second layer (plus the last) of DINOv2 S/B/L/G:
  spearman  out-of-fold probe prediction vs the mean rating, all 2,940 images
  auroc     top third vs bottom third of the rating, 300 images each, out-of-fold

  python src/e1_probe_layers.py --cache CACHE --out results
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import MODELS, Cache, cli, oof, parse, probe, rho

RATINGS = ["valence", "morality", "arousal"]


def balanced_extremes(y, n_per_class, rng):
    """Indices of n images from the bottom third and n from the top third, and their 0/1 labels."""
    lo, hi = np.where(y <= np.quantile(y, 1 / 3))[0], np.where(y >= np.quantile(y, 2 / 3))[0]
    n = min(n_per_class, len(lo), len(hi))
    return np.r_[rng.choice(lo, n, replace=False), rng.choice(hi, n, replace=False)], np.r_[np.zeros(n), np.ones(n)]


def auroc(X, yb, k=256, folds=5):
    s = np.zeros(len(yb))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, yb):
        s[te] = probe(min(k, X.shape[1], len(tr) - 1)).fit(X[tr], yb[tr]).predict(X[te])
    return roc_auc_score(yb, s)


def main():
    ap = cli(__doc__)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-per-class", type=int, default=300)
    a = parse(ap)
    cache = Cache(a.cache)
    lab = cache.labels()
    rng = np.random.default_rng(0)
    extremes = {r: balanced_extremes(lab[r].to_numpy(float), a.n_per_class, rng) for r in RATINGS}

    rows = []
    for m in MODELS:
        F = cache.dinov2(m, layer=None)
        n_layers = F.shape[0]
        for l in sorted(set(range(0, n_layers, a.stride)) | {n_layers - 1}):
            X = np.asarray(F[l], dtype=np.float32)
            for r in RATINGS:
                y = lab[r].to_numpy(float)
                idx, yb = extremes[r]
                rows.append({"model": m, "layer": l + 1, "depth": round((l + 1) / n_layers, 4), "rating": r,
                             "spearman": rho(y, oof(X, y)), "auroc": round(auroc(X[idx], yb), 4)})
            print(m, f"layer {l + 1}/{n_layers}", {r["rating"]: r["spearman"] for r in rows[-3:]}, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/e1_probe_layers.csv", index=False)
    best = df.loc[df.groupby(["model", "rating"]).spearman.idxmax()]
    print(best.pivot(index="rating", columns="model", values="spearman")[list(MODELS)])


if __name__ == "__main__":
    main()
