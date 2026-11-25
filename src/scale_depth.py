"""
Does the part of valence / morality that is NOT explained by image statistics and objects grow with model size
and depth? And does it grow faster than object content, which is the obvious alternative explanation?

For every DINOv2 size (S/B/L/G) and every `--stride`-th layer, an out-of-fold probe (scale -> PCA256 -> ridge)
predicts each target. Targets:
  valence, morality, arousal                     raw ratings
  valence_ctrl, morality_ctrl, mbv, mbv_ctrl     after removing valence (mbv) and/or low-level + objects + supervised ViT (ctrl)
  objects_pc1..5                                 top principal components of the ImageNet logits (non-saturating object content)
  brightness, edge_density                       low-level reference

Then, per target: best rho per size, the depth where each size first reaches 90% of its best, and a paired
bootstrap over images of the G-minus-S gain at the best layer (and of that gain minus the gain on object content).

  python scale_depth.py --features features --extra features_extra --out results_scale_depth
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from analyze_controls_clip import ALPHAS, oof

PARAMS = {"S": 21e6, "B": 86e6, "L": 300e6, "G": 1100e6}


def oof_multi(X, Y, k=256, folds=5, seed=0):
    P = np.zeros_like(Y)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        kk = min(k, X.shape[1], len(tr) - 1)
        m = make_pipeline(StandardScaler(), PCA(kk, random_state=0)).fit(X[tr])
        P[te] = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(m.transform(X[tr]), Y[tr]).predict(m.transform(X[te]))
    return P


def fast_spearman_cols(Y, P):
    ry = np.apply_along_axis(rankdata, 0, Y); rp = np.apply_along_axis(rankdata, 0, P)
    ry -= ry.mean(0); rp -= rp.mean(0)
    return (ry * rp).sum(0) / np.sqrt((ry ** 2).sum(0) * (rp ** 2).sum(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--points", type=int, default=8, help="layers per model, evenly spaced in depth (last layer always included)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results_scale_depth")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(f"{a.features}/smid_labels.csv")
    val, mor, aro = (lab[c].to_numpy(float) for c in ["valence", "morality", "arousal"])
    ll = np.load(f"{a.extra}/lowlevel.npy"); obj = np.load(f"{a.extra}/sup_logits.npy"); sup = np.load(f"{a.extra}/sup_feats.npy")
    sp = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(val[:, None], mor)
    mbv = mor - sp.predict(val[:, None])
    ctrl = np.hstack([ll, obj, sup])
    R = oof(ctrl, np.c_[val, mor, mbv])
    objpc = PCA(5, random_state=0).fit_transform(StandardScaler().fit_transform(obj))
    targets = {"valence": val, "morality": mor, "arousal": aro,
               "valence_ctrl": val - R[:, 0], "morality_ctrl": mor - R[:, 1], "mbv": mbv, "mbv_ctrl": mbv - R[:, 2],
               **{f"objects_pc{i + 1}": objpc[:, i] for i in range(5)},
               "brightness": ll[:, 12], "edge_density": ll[:, 14]}
    names = list(targets); Y = np.c_[tuple(targets.values())]

    rows, best_preds = [], {}
    csv = out / "scale_depth_layers.csv"
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["model"])
    for m in ["S", "B", "L", "G"]:
        F = np.load(f"{a.features}/{m}_smid.npy", mmap_mode="r")
        L = F.shape[0]
        layers = sorted(set(int(round(x)) - 1 for x in np.linspace(L / a.points, L, a.points)))
        preds = {}
        prev = done[done.model == m]
        if len(prev) and set(prev.layer) >= {l + 1 for l in layers}:  # resume: this size finished in an earlier run
            print(m, "loaded from", csv, flush=True)
            rows += prev.to_dict(orient="records")
            for t in names:
                d = prev[prev.target == t]; l = int(d.loc[d.rho.idxmax(), "layer"]) - 1
                if l not in preds:
                    preds[l] = oof_multi(np.asarray(F[l], dtype=np.float32), Y)
                best_preds[(m, t)] = preds[l][:, names.index(t)]
            continue
        for l in layers:
            P = oof_multi(np.asarray(F[l], dtype=np.float32), Y)
            preds[l] = P
            for t, r in zip(names, fast_spearman_cols(Y, P)):
                rows.append({"model": m, "params": PARAMS[m], "layer": l + 1, "depth": (l + 1) / L, "target": t, "rho": round(float(r), 4)})
            print(m, l + 1, "/", L, {t: r["rho"] for r in rows[-len(names):] for t in [r["target"]] if t in ("valence_ctrl", "morality_ctrl", "mbv_ctrl", "objects_pc1")}, flush=True)
        df = pd.DataFrame(rows); df.to_csv(csv, index=False)
        for t in names:
            d = df[(df.model == m) & (df.target == t)]
            best_preds[(m, t)] = preds[int(d.loc[d.rho.idxmax(), "layer"]) - 1][:, names.index(t)]

    df = pd.DataFrame(rows)
    summ = []
    for t in names:
        for m in ["S", "B", "L", "G"]:
            d = df[(df.model == m) & (df.target == t)].sort_values("depth")
            b = d.rho.max()
            summ.append({"target": t, "model": m, "best_rho": b, "best_depth": float(d.loc[d.rho.idxmax(), "depth"]),
                         "depth_to_90pct": float(d[d.rho >= 0.9 * b].depth.min()) if b > 0 else None,
                         "rho_at_half_depth": float(d[d.depth <= 0.5].rho.max())})
    summ = pd.DataFrame(summ); summ.to_csv(out / "scale_depth_summary.csv", index=False)

    # paired bootstrap of the S -> G gain at each model's best layer
    rng = np.random.default_rng(0); n = len(Y)
    idx = [rng.integers(0, n, n) for _ in range(a.boot)]
    gain = {}
    for t in names:
        y = targets[t]; pS, pG = best_preds[("S", t)], best_preds[("G", t)]
        gain[t] = np.array([spearmanr(y[i], pG[i])[0] - spearmanr(y[i], pS[i])[0] for i in idx])
    obj_gain = np.mean([gain[f"objects_pc{i + 1}"] for i in range(5)], axis=0)
    res = {}
    for t in names:
        g = gain[t]
        res[t] = {"gain_G_minus_S": round(float(g.mean()), 4), "ci": [round(float(x), 4) for x in np.percentile(g, [2.5, 97.5])],
                  "gain_minus_object_gain": round(float((g - obj_gain).mean()), 4),
                  "ci_vs_objects": [round(float(x), 4) for x in np.percentile(g - obj_gain, [2.5, 97.5])]}
    json.dump({"bootstrap_gain": res, "summary": summ.to_dict(orient="records")}, open(out / "scale_depth.json", "w"), indent=1)
    print(summ.pivot(index="target", columns="model", values="best_rho").round(3))
    print(json.dumps(res, indent=0))


if __name__ == "__main__":
    main()
