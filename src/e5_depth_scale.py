"""Test 5: where in the network does the moral signal appear, and does it grow with model size?

For 8 evenly spaced layers of each DINOv2 size, probe:
  valence, morality, arousal, morality beyond valence (mbv)
  the same after removing low-level stats + ImageNet logits + supervised-ViT features (*_ctrl)
  object content (top 5 principal components of the ImageNet logits) and two low-level statistics, for comparison
Then compare the smallest and largest model at each target's best layer with a paired bootstrap over images,
and subtract the average gain on object content.

The per-layer table is saved after each model size, and finished sizes are skipped on a rerun.

  python src/e5_depth_scale.py --cache CACHE --out results
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ALPHAS, BRIGHTNESS, EDGE_DENSITY, MODELS, Cache, cli, oof, parse, save_json, targets


def oof_many(X, Y, k=256, folds=5):
    """Out-of-fold predictions for many targets at once (one PCA per fold, one ridge penalty per target)."""
    P = np.zeros_like(Y)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        Z = make_pipeline(StandardScaler(), PCA(min(k, X.shape[1], len(tr) - 1), random_state=0)).fit(X[tr])
        P[te] = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Z.transform(X[tr]), Y[tr]).predict(Z.transform(X[te]))
    return P


def spearman_columns(Y, P):
    ry, rp = (np.apply_along_axis(rankdata, 0, M) for M in (Y, P))
    ry -= ry.mean(0); rp -= rp.mean(0)
    return (ry * rp).sum(0) / np.sqrt((ry ** 2).sum(0) * (rp ** 2).sum(0))


def build_targets(cache):
    lab = cache.labels()
    T = targets(lab)
    ll, logits = cache.extra_features("lowlevel"), cache.extra_features("sup_logits")
    R = oof(cache.controls(), np.c_[T["valence"], T["morality"], T["morality_beyond_valence"]])
    objects = PCA(5, random_state=0).fit_transform(StandardScaler().fit_transform(logits))
    out = {"valence": T["valence"], "morality": T["morality"], "arousal": lab.arousal.to_numpy(float),
           "mbv": T["morality_beyond_valence"],
           "valence_ctrl": T["valence"] - R[:, 0], "morality_ctrl": T["morality"] - R[:, 1],
           "mbv_ctrl": T["morality_beyond_valence"] - R[:, 2],
           **{f"objects_pc{i + 1}": objects[:, i] for i in range(5)},
           "brightness": ll[:, BRIGHTNESS], "edge_density": ll[:, EDGE_DENSITY]}
    return list(out), np.c_[tuple(out.values())]


def main():
    ap = cli(__doc__)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--boot", type=int, default=1000)
    a = parse(ap)
    cache = Cache(a.cache)
    names, Y = build_targets(cache)
    csv = Path(a.out) / "e5_depth_scale_layers.csv"
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["model"])

    rows, best_pred = [], {}
    for m in MODELS:
        F = cache.dinov2(m, layer=None)
        n_layers = F.shape[0]
        layers = sorted({int(round(x)) - 1 for x in np.linspace(n_layers / a.points, n_layers, a.points)})
        prev = done[done.model == m]
        if len(prev) and set(prev.layer) >= {l + 1 for l in layers}:
            print(m, "already done", flush=True)
            rows += prev.to_dict(orient="records")
            needed = {int(prev[prev.target == t].sort_values("rho").layer.iloc[-1]) - 1 for t in names}
            preds = {l: oof_many(np.asarray(F[l], dtype=np.float32), Y) for l in needed}
        else:
            preds = {}
            for l in layers:
                preds[l] = oof_many(np.asarray(F[l], dtype=np.float32), Y)
                rows += [{"model": m, "layer": l + 1, "depth": round((l + 1) / n_layers, 4), "target": t, "rho": round(float(r), 4)}
                         for t, r in zip(names, spearman_columns(Y, preds[l]))]
                print(m, f"layer {l + 1}/{n_layers}", flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)
        df = pd.DataFrame(rows)
        for j, t in enumerate(names):
            d = df[(df.model == m) & (df.target == t)]
            best_pred[(m, t)] = preds[int(d.loc[d.rho.idxmax(), "layer"]) - 1][:, j]

    df = pd.DataFrame(rows)
    best = df.loc[df.groupby(["model", "target"]).rho.idxmax()].pivot(index="target", columns="model", values="rho")[list(MODELS)]
    print(best.round(3))

    rng = np.random.default_rng(0)
    resamples = [rng.integers(0, len(Y), len(Y)) for _ in range(a.boot)]
    gain = {t: np.array([spearmanr(Y[i, j], best_pred[("G", t)][i])[0] - spearmanr(Y[i, j], best_pred[("S", t)][i])[0]
                         for i in resamples]) for j, t in enumerate(names)}
    object_gain = np.mean([gain[f"objects_pc{i + 1}"] for i in range(5)], axis=0)
    ci = lambda x: [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]
    save_json({"best_rho": best.round(4).to_dict(orient="index"),
               "gain_smallest_to_largest": {t: {"gain": round(float(g.mean()), 4), "ci": ci(g),
                                                "minus_object_gain": round(float((g - object_gain).mean()), 4),
                                                "minus_object_gain_ci": ci(g - object_gain)} for t, g in gain.items()}},
              Path(a.out) / "e5_depth_scale.json")


if __name__ == "__main__":
    main()
