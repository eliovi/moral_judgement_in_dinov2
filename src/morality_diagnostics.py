"""
Why is "morality beyond valence" weak once image statistics and objects are removed?

Target: morality minus a spline fit on valence (mbv), then minus an out-of-fold prediction from
low-level statistics + ImageNet logits + supervised-ViT features (mbv_ctrl). Questions:

  D1  Is DINOv2's prediction of mbv_ctrl statistically significant? Bootstrap CI, analytic p, and a
      permutation null that reruns the whole probe on shuffled targets.
  D2  How much of each target is rating noise? Reliability from SMID's per-image standard errors, and the
      highest correlation any model could reach (noise ceiling).
  D3  Does the signal grow on images whose morality rating is more precise (low SE, more raters)?
  D4  What is mbv made of? Correlation with the moral-foundation ratings and arousal, CLIP word rankings,
      and the images at each end.
  D5  Is the gap DINOv2-specific? CLIP and the supervised ViT on the same residual.

  python morality_diagnostics.py --features features --extra features_extra --metadata metadata.csv --out results_morality_diagnostics.json
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from analyze_controls_clip import ALPHAS, oof, rho_ci

RNG = np.random.default_rng(1)
FOUNDATIONS = ["harm", "fairness", "ingroup", "authority", "purity"]


class FoldCache:
    """Scaler + PCA per fold fitted once (they do not see y), so permutations only refit the ridge."""

    def __init__(self, X, k=256, folds=5, seed=0):
        self.parts = []
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
            kk = min(k, X.shape[1], len(tr) - 1)
            m = make_pipeline(StandardScaler(), PCA(kk, random_state=0)).fit(X[tr])
            self.parts.append((tr, te, m.transform(X[tr]), m.transform(X[te])))

    def predict(self, y):
        p = np.zeros_like(y, dtype=np.float64)
        for tr, te, Ztr, Zte in self.parts:
            p[te] = RidgeCV(alphas=ALPHAS).fit(Ztr, y[tr]).predict(Zte)
        return p


def spline(y, z):
    m = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(z[:, None], y)
    eps = 1e-3
    slope = (m.predict(z[:, None] + eps) - m.predict(z[:, None] - eps)) / (2 * eps)
    return y - m.predict(z[:, None]), slope


def reliability(y, noise_var):
    rel = 1 - float(np.mean(noise_var)) / float(np.var(y))
    return {"reliability": round(rel, 4), "noise_ceiling_rho": round(float(np.sqrt(max(rel, 0))), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--metadata", required=True, help="SMID metadata.csv (needs *_mean and *_se columns)")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--out", default="results_morality_diagnostics.json")
    a = ap.parse_args()

    lab = pd.read_csv(f"{a.features}/smid_labels.csv")
    meta = pd.read_csv(a.metadata).assign(image=lambda d: d.file_name.str.replace(r"\.[^.]+$", "", regex=True))
    meta = lab[["image"]].merge(meta, on="image", how="left")
    assert np.allclose(meta.moral_mean, lab.morality)
    val, mor, aro = (lab[c].to_numpy(float) for c in ["valence", "morality", "arousal"])
    col = lambda c: meta[c].to_numpy(float) if c in meta else None

    dino = np.asarray(np.load(f"{a.features}/G_smid.npy", mmap_mode="r")[a.layer], dtype=np.float32)
    ll = np.load(f"{a.extra}/lowlevel.npy"); obj = np.load(f"{a.extra}/sup_logits.npy"); sup = np.load(f"{a.extra}/sup_feats.npy")
    C = np.load(f"{a.extra}/clip_img.npy").astype(np.float64)
    T = np.load(f"{a.extra}/clip_text.npy").astype(np.float64); tix = json.load(open(f"{a.extra}/text_index.json"))
    temb = {t: T[i] for i, t in enumerate(tix["texts"])}

    mbv, slope = spline(mor, val)
    ctrl = np.hstack([ll, obj, sup])
    mbv_ctrl = mbv - oof(ctrl, mbv)
    targets = {"valence": val, "morality": mor, "mbv": mbv, "mbv_ctrl": mbv_ctrl}
    res = {"n_images": len(lab), "definitions": {
        "mbv": "morality minus a cubic-spline fit on valence",
        "mbv_ctrl": "mbv minus its out-of-fold prediction from low-level stats + ImageNet logits + supervised-ViT features"}}
    save = lambda: json.dump(res, open(a.out, "w"), indent=1)

    # D1 significance
    print("D1 significance", flush=True)
    fc = FoldCache(dino)
    p = fc.predict(mbv_ctrl)
    obs = spearmanr(mbv_ctrl, p)
    null = np.array([spearmanr(y, fc.predict(y))[0] for y in (RNG.permutation(mbv_ctrl) for _ in range(a.perms))])
    res["D1"] = {"dinov2_on_mbv_ctrl": rho_ci(mbv_ctrl, p, n=1000), "analytic_p": float(obs.pvalue),
                 "perm_null": {"n": a.perms, "mean": round(float(null.mean()), 4), "sd": round(float(null.std()), 4),
                               "max": round(float(null.max()), 4), "z": round(float((obs.statistic - null.mean()) / null.std()), 2),
                               "p": float((np.sum(null >= obs.statistic) + 1) / (a.perms + 1))}}
    print(" ", res["D1"], flush=True); save()

    # D2 noise ceilings
    print("D2 reliability", flush=True)
    se_m, se_v = col("moral_se"), col("valence_se")
    nm = se_m ** 2
    nv = se_v ** 2 if se_v is not None else np.zeros_like(nm)
    res["D2"] = {"assumption": "rating errors of morality and valence independent; 'upper' adds a fully correlated error term",
                 "mean_se": {"morality": round(float(se_m.mean()), 4), "valence": round(float(se_v.mean()), 4) if se_v is not None else None}}
    noise_mbv = nm + slope ** 2 * nv
    noise_mbv_hi = noise_mbv + 2 * np.abs(slope) * np.sqrt(nm * nv)
    for t, y, nz in [("valence", val, nv), ("morality", mor, nm), ("mbv", mbv, noise_mbv), ("mbv_ctrl", mbv_ctrl, noise_mbv)]:
        res["D2"][t] = reliability(y, nz) | {"noise_share": round(float(np.mean(nz) / np.var(y)), 4)}
    res["D2"]["mbv_ctrl_upper_noise"] = reliability(mbv_ctrl, noise_mbv_hi)
    obs_r = {t: rho_ci(y, oof(dino, y) if t != "mbv_ctrl" else p, n=300) for t, y in targets.items()}
    for t in targets:
        res["D2"][t]["dinov2_rho"] = obs_r[t]["rho"]
        c = res["D2"][t]["noise_ceiling_rho"]
        res["D2"][t]["share_of_ceiling"] = round(obs_r[t]["rho"] / c, 4) if c > 0 else None
    print(" ", res["D2"], flush=True); save()

    # D3 precision split
    print("D3 precision split", flush=True)
    res["D3"] = {}
    for name, w in [("moral_se", se_m), ("moral_n", col("moral_n"))]:
        if w is None:
            continue
        hi_prec = w <= np.median(w) if name.endswith("se") else w >= np.median(w)
        res["D3"][name] = {"more_precise_half": rho_ci(mbv_ctrl[hi_prec], p[hi_prec], n=300),
                           "less_precise_half": rho_ci(mbv_ctrl[~hi_prec], p[~hi_prec], n=300),
                           "median": float(np.median(w))}
    # weighted by inverse noise
    wts = 1 / noise_mbv
    cv = np.cov(np.c_[mbv_ctrl, p].T, aweights=wts)
    res["D3"]["inverse_noise_weighted_pearson"] = round(float(cv[0, 1] / np.sqrt(cv[0, 0] * cv[1, 1])), 4)
    res["D3"]["unweighted_pearson"] = round(float(np.corrcoef(mbv_ctrl, p)[0, 1]), 4)
    print(" ", res["D3"], flush=True); save()

    # D4 what mbv is made of
    print("D4 content", flush=True)
    res["D4"] = {"rating_correlations": {}}
    for f in FOUNDATIONS + ["arousal", "valence"]:
        y = col(f"{f}_mean") if f in FOUNDATIONS else (aro if f == "arousal" else val)
        if y is None:
            continue
        res["D4"]["rating_correlations"][f] = {t: round(float(spearmanr(targets[t], y)[0]), 4) for t in ["morality", "mbv", "mbv_ctrl"]}
        res["D4"]["rating_correlations"][f]["controls_predict"] = round(float(spearmanr(y, oof(ctrl, y))[0]), 4)
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    words = tix["words"]
    for t in ["mbv", "mbv_ctrl"]:
        wr = []
        for w in words:
            e = temb[tix["template"].format(w)]
            wr.append((w, round(float(spearmanr(Cn @ (e / np.linalg.norm(e)), targets[t])[0]), 4)))
        wr.sort(key=lambda x: -x[1])
        res["D4"][f"words_{t}"] = {"high_end": wr[:15], "low_end": wr[::-1][:15]}
    names = meta["img_name"].astype(str).to_numpy() if "img_name" in meta else lab.image.astype(str).to_numpy()
    o = np.argsort(mbv_ctrl)
    res["D4"]["mbv_ctrl_extremes"] = {"highest": [(names[i], round(float(mor[i]), 2), round(float(val[i]), 2)) for i in o[::-1][:15]],
                                      "lowest": [(names[i], round(float(mor[i]), 2), round(float(val[i]), 2)) for i in o[:15]]}
    print(" ", json.dumps(res["D4"])[:3000], flush=True); save()

    # D5 other models on the same residual
    print("D5 other models", flush=True)
    res["D5"] = {"dinov2": res["D1"]["dinov2_on_mbv_ctrl"], "clip": rho_ci(mbv_ctrl, oof(C, mbv_ctrl), n=300),
                 "clip_plus_dinov2": rho_ci(mbv_ctrl, oof(np.hstack([C.astype(np.float32), dino]), mbv_ctrl), n=300)}
    print(" ", res["D5"], flush=True); save()
    print("saved", a.out)


if __name__ == "__main__":
    main()
