"""
CPU-only tests that make the "beyond objects" and "shared across models" claims harder to dismiss.
Uses only cached features (DINOv2 S/B/L/G per layer, CLIP, low-level stats, ImageNet logits, supervised ViT).

  S1  Non-linear controls. The main control is a linear probe on low-level + objects + supervised ViT, so object
      information could leak through non-linearly. Remove the controls with k-NN and gradient boosting as well,
      then ask how much DINOv2 still predicts.
  S2  Same objects, different rating. Cluster images by object content (k-means on the control features) and
      correlate DINOv2's prediction with the rating *within* clusters only.
  S3  Unseen object groups. Train the probe on some object clusters and test on held-out clusters
      (GroupKFold). If the direction only memorised object categories it should not transfer.
  S4  Convergence with a language-trained model (Platonic angle). Mutual k-NN alignment (Huh et al. 2024)
      between each DINOv2 size/depth and CLIP, with the supervised ViT as a reference.

  python strengthen.py --features features --extra features_extra --out results_strengthen.json
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from analyze_controls_clip import ALPHAS, oof, rho_ci


def oof_model(make, X, y, folds=5, groups=None, seed=0):
    p = np.zeros_like(y, dtype=np.float64)
    split = GroupKFold(folds).split(X, y, groups) if groups is not None else KFold(folds, shuffle=True, random_state=seed).split(X)
    for tr, te in split:
        p[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return p


def probe():
    return make_pipeline(StandardScaler(), PCA(256, random_state=0), RidgeCV(alphas=ALPHAS))


def within(y, p, lab):
    """Spearman of y and p after removing each cluster's mean from both (ranks first)."""
    ry, rp = rankdata(y), rankdata(p)
    for c in np.unique(lab):
        m = lab == c
        ry[m] -= ry[m].mean(); rp[m] -= rp[m].mean()
    return float(np.corrcoef(ry, rp)[0, 1])


def boot_within(y, p, lab, n=300, seed=0):
    rng = np.random.default_rng(seed); cl = np.unique(lab); vals = []
    for _ in range(n):  # resample whole clusters
        pick = rng.choice(cl, len(cl))
        idx = np.concatenate([np.flatnonzero(lab == c) for c in pick])
        lab2 = np.concatenate([np.full((lab == c).sum(), i) for i, c in enumerate(pick)])
        vals.append(within(y[idx], p[idx], lab2))
    return {"rho": round(within(y, p, lab), 4), "lo": round(float(np.percentile(vals, 2.5)), 4), "hi": round(float(np.percentile(vals, 97.5)), 4)}


def mutual_knn(A, B, k=10):
    def nn(X):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        S = X @ X.T; np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]
    na, nb = nn(A), nn(B)
    return float(np.mean([len(set(na[i]) & set(nb[i])) / k for i in range(len(A))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--out", default="results_strengthen.json")
    a = ap.parse_args()
    res = {}
    save = lambda: json.dump(res, open(a.out, "w"), indent=1)

    lab = pd.read_csv(f"{a.features}/smid_labels.csv")
    val, mor = (lab[c].to_numpy(float) for c in ["valence", "morality"])
    sp = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(val[:, None], mor)
    mbv = mor - sp.predict(val[:, None])
    targets = {"valence": val, "morality": mor, "mbv": mbv}
    dino = np.asarray(np.load(f"{a.features}/G_smid.npy", mmap_mode="r")[a.layer], dtype=np.float32)
    ll = np.load(f"{a.extra}/lowlevel.npy"); obj = np.load(f"{a.extra}/sup_logits.npy"); sup = np.load(f"{a.extra}/sup_feats.npy")
    C = np.load(f"{a.extra}/clip_img.npy").astype(np.float32)
    ctrl = np.hstack([ll, obj, sup])
    Zc = PCA(64, random_state=0).fit_transform(StandardScaler().fit_transform(ctrl))

    # S1 non-linear controls
    print("S1 non-linear controls", flush=True)
    makers = {"linear": None,
              "knn": lambda: KNeighborsRegressor(25, weights="distance"),
              "boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)}
    res["S1"] = {}
    for t, y in targets.items():
        res["S1"][t] = {}
        lin = oof(ctrl, y)
        preds = {"linear": lin, "knn": oof_model(makers["knn"], Zc, y), "boosting": oof_model(makers["boosting"], Zc, y)}
        # strongest: linear, then boosting and k-NN on what is left, averaged
        r1 = y - lin
        preds["linear+nonlinear"] = lin + 0.5 * (oof_model(makers["boosting"], Zc, r1) + oof_model(makers["knn"], Zc, r1))
        for name, pc in preds.items():
            r = y - pc
            res["S1"][t][name] = {"controls_alone": round(float(spearmanr(y, pc)[0]), 4),
                                  "dinov2_on_residual": rho_ci(r, oof(dino, r), n=300),
                                  "clip_on_residual": rho_ci(r, oof(C, r), n=300)}
        print(" ", t, {k: v["dinov2_on_residual"]["rho"] for k, v in res["S1"][t].items()}, flush=True)
    save()

    # S2 within object clusters
    print("S2 within object clusters", flush=True)
    res["S2"] = {}
    full = {t: {"dinov2": oof(dino, y), "clip": oof(C, y), "controls": oof(ctrl, y)} for t, y in targets.items()}
    for K in [50, 150]:
        cl = KMeans(K, n_init=4, random_state=0).fit_predict(Zc)
        res["S2"][f"K{K}"] = {t: {m: boot_within(y, p, cl) for m, p in full[t].items()} for t, y in targets.items()}
        res["S2"][f"K{K}"]["share_of_variance_between_clusters"] = {
            t: round(float(sum(((y[cl == c].mean() - y.mean()) ** 2) * (cl == c).sum() for c in np.unique(cl)) / ((y - y.mean()) ** 2).sum()), 4)
            for t, y in targets.items()}
        print(" ", K, {t: {m: v["rho"] for m, v in d.items()} for t, d in res["S2"][f"K{K}"].items() if t in targets}, flush=True)
    save()

    # S3 held-out object clusters
    print("S3 held-out object groups", flush=True)
    groups = KMeans(50, n_init=4, random_state=0).fit_predict(PCA(32, random_state=0).fit_transform(StandardScaler().fit_transform(obj)))
    res["S3"] = {}
    for t, y in targets.items():
        res["S3"][t] = {"dinov2_random_folds": rho_ci(y, full[t]["dinov2"], n=300),
                        "dinov2_heldout_object_groups": rho_ci(y, oof_model(probe, dino, y, groups=groups), n=300),
                        "objects_heldout_object_groups": rho_ci(y, oof_model(probe, obj, y, groups=groups), n=300),
                        "clip_heldout_object_groups": rho_ci(y, oof_model(probe, C, y, groups=groups), n=300)}
        print(" ", t, {k: v["rho"] for k, v in res["S3"][t].items()}, flush=True)
    save()

    # S4 convergence with CLIP across size and depth
    print("S4 mutual k-NN alignment", flush=True)
    res["S4"] = {"reference_supervised_vit_vs_clip": round(mutual_knn(sup, C), 4), "dinov2_vs_clip": [], "dinov2_vs_supervised_vit": []}
    for m in ["S", "B", "L", "G"]:
        F = np.load(f"{a.features}/{m}_smid.npy", mmap_mode="r"); L = F.shape[0]
        for l in sorted(set(list(range(0, L, max(1, L // 8))) + [L - 1])):
            X = np.asarray(F[l], dtype=np.float32)
            res["S4"]["dinov2_vs_clip"].append({"model": m, "layer": l + 1, "depth": round((l + 1) / L, 3), "mknn": round(mutual_knn(X, C), 4)})
            res["S4"]["dinov2_vs_supervised_vit"].append({"model": m, "layer": l + 1, "depth": round((l + 1) / L, 3), "mknn": round(mutual_knn(X, sup), 4)})
        best = max(r["mknn"] for r in res["S4"]["dinov2_vs_clip"] if r["model"] == m)
        print(" ", m, "best mutual-kNN with CLIP", best, flush=True)
    save()
    print("saved", a.out)


if __name__ == "__main__":
    main()
