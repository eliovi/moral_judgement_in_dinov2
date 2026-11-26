"""Test 2: is DINOv2 using a shortcut (image statistics or objects) instead of the moral content?

All on DINOv2-g layer 37, for valence, morality and morality beyond valence:
  linear_controls   remove what the controls predict (cross-fitted), probe DINOv2 (and CLIP) on what is left;
                    controls added in turn: low-level stats, + ImageNet logits, + supervised-ViT features
  nonlinear         the same with k-NN, gradient boosting, and linear + both stacked
  within_clusters   rank images only within clusters of similar object content (K = 50, 150)
  heldout_groups    train on some object groups, test on unseen ones (grouped 5-fold)

  python src/e2_shortcuts.py --cache CACHE --out results
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from common import Cache, cli, oof, parse, rho, rho_ci, save_json, targets


def oof_model(make, X, y, folds=5):
    p = np.zeros_like(y)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        p[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return p


def within(y, p, clusters):
    """Spearman of y and p after removing each cluster's mean from both (on ranks)."""
    ry, rp = rankdata(y), rankdata(p)
    for c in np.unique(clusters):
        m = clusters == c
        ry[m] -= ry[m].mean(); rp[m] -= rp[m].mean()
    return float(np.corrcoef(ry, rp)[0, 1])


def within_ci(y, p, clusters, n=300, seed=0):
    """Within-cluster correlation with a bootstrap that resamples whole clusters."""
    rng, ids, boots = np.random.default_rng(seed), np.unique(clusters), []
    for _ in range(n):
        pick = rng.choice(ids, len(ids))
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in pick])
        relabel = np.concatenate([np.full((clusters == c).sum(), i) for i, c in enumerate(pick)])
        boots.append(within(y[idx], p[idx], relabel))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": round(within(y, p, clusters), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def between_share(y, clusters):
    """Share of the variance of y explained by cluster membership."""
    return round(float(sum(((y[clusters == c].mean() - y.mean()) ** 2) * (clusters == c).sum()
                           for c in np.unique(clusters)) / ((y - y.mean()) ** 2).sum()), 4)


def main():
    a = parse(cli(__doc__))
    cache = Cache(a.cache)
    T = targets(cache.labels())
    dino, clip = cache.dinov2(), cache.clip()
    ll, logits, sup = (cache.extra_features(k) for k in ("lowlevel", "sup_logits", "sup_feats"))
    controls = np.hstack([ll, logits, sup])
    Zc = PCA(64, random_state=0).fit_transform(StandardScaler().fit_transform(controls))
    res = {}

    print("linear controls", flush=True)
    steps = {"low-level": ll, "+ ImageNet logits": np.hstack([ll, logits]), "+ supervised-ViT features": controls}
    res["linear_controls"] = {}
    for t, y in T.items():
        res["linear_controls"][t] = {"none": {"dinov2": rho_ci(y, oof(dino, y)), "clip": rho_ci(y, oof(clip, y))}}
        for name, Z in steps.items():
            pz = oof(Z, y); r = y - pz
            res["linear_controls"][t][name] = {"controls_alone": rho(y, pz), "dinov2": rho_ci(r, oof(dino, r)),
                                               "clip": rho_ci(r, oof(clip, r))}
        print(" ", t, {k: v["dinov2"]["rho"] for k, v in res["linear_controls"][t].items()}, flush=True)

    print("non-linear controls", flush=True)
    knn = lambda: KNeighborsRegressor(25, weights="distance")
    boost = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    res["nonlinear_controls"] = {}
    for t, y in T.items():
        lin = oof(controls, y)
        preds = {"linear": lin, "knn": oof_model(knn, Zc, y), "boosting": oof_model(boost, Zc, y),
                 "linear+nonlinear": lin + 0.5 * (oof_model(boost, Zc, y - lin) + oof_model(knn, Zc, y - lin))}
        res["nonlinear_controls"][t] = {}
        for name, pz in preds.items():
            r = y - pz
            res["nonlinear_controls"][t][name] = {"controls_alone": rho(y, pz), "dinov2": rho_ci(r, oof(dino, r)),
                                                  "clip": rho_ci(r, oof(clip, r))}
        print(" ", t, {k: v["dinov2"]["rho"] for k, v in res["nonlinear_controls"][t].items()}, flush=True)

    print("within object clusters", flush=True)
    full = {t: {"dinov2": oof(dino, y), "clip": oof(clip, y), "controls": oof(controls, y)} for t, y in T.items()}
    res["within_clusters"] = {}
    for K in [50, 150]:
        clusters = KMeans(K, n_init=4, random_state=0).fit_predict(Zc)
        res["within_clusters"][f"K={K}"] = {t: {m: within_ci(y, p, clusters) for m, p in full[t].items()}
                                            | {"between_cluster_variance_share": between_share(y, clusters)}
                                            for t, y in T.items()}
        print(" ", K, {t: v["dinov2"]["rho"] for t, v in res["within_clusters"][f"K={K}"].items()}, flush=True)

    print("held-out object groups", flush=True)
    groups = KMeans(50, n_init=4, random_state=0).fit_predict(
        PCA(32, random_state=0).fit_transform(StandardScaler().fit_transform(logits)))
    res["heldout_groups"] = {t: {"dinov2_random_folds": rho_ci(y, full[t]["dinov2"]),
                                 "dinov2_unseen_groups": rho_ci(y, oof(dino, y, groups=groups)),
                                 "imagenet_logits_unseen_groups": rho_ci(y, oof(logits, y, groups=groups)),
                                 "clip_unseen_groups": rho_ci(y, oof(clip, y, groups=groups))} for t, y in T.items()}
    print(" ", {t: v["dinov2_unseen_groups"]["rho"] for t, v in res["heldout_groups"].items()}, flush=True)
    save_json(res, f"{a.out}/e2_shortcuts.json")


if __name__ == "__main__":
    main()
