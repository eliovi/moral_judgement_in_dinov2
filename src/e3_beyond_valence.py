"""Test 3: is there moral signal beyond pleasantness, and is it real?

Morality beyond valence (MBV) = morality rating minus a cubic-spline fit on valence. "After controls" = MBV minus what
low-level stats + ImageNet logits + supervised-ViT features predict (cross-fitted).
  definition     share of morality variance, correlation with valence, and the DINOv2 probe for three definitions
  reliability    share of each target that is rating noise (from SMID's per-image standard errors), and the noise ceiling
  permutation    DINOv2 on MBV after controls vs 200 runs on shuffled targets
  precision      the same correlation on the half of images with the most / least precise morality ratings
  foundations    correlation of morality and MBV with SMID's moral-foundation relevance ratings

  python src/e3_beyond_valence.py --cache CACHE --metadata metadata.csv --out results
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from common import Cache, cli, load_metadata, oof, parse, probe, rho, rho_ci, save_json, spline_fit, targets

FOUNDATIONS = ["harm", "fairness", "ingroup", "authority", "purity"]


def permutation_test(X, y, n, k=256, folds=5, seed=1):
    """Refit only the ridge on shuffled targets; scaling and PCA per fold do not see the target, so they are reused."""
    parts = []
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        pipe = probe(min(k, X.shape[1], len(tr) - 1))
        Z = pipe[:-1].fit(X[tr])
        parts.append((tr, te, Z.transform(X[tr]), Z.transform(X[te]), pipe[-1]))

    def predict(target):
        p = np.zeros_like(target)
        for tr, te, Ztr, Zte, ridge in parts:
            p[te] = ridge.fit(Ztr, target[tr]).predict(Zte)
        return p

    observed = spearmanr(y, predict(y))[0]
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(s, predict(s))[0] for s in (rng.permutation(y) for _ in range(n))])
    return {"observed": round(float(observed), 4), "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
            "null_max": round(float(null.max()), 4), "z": round(float((observed - null.mean()) / null.std()), 2),
            "p": float((np.sum(null >= observed) + 1) / (n + 1)), "n_shuffles": n}


def spline_2d(y, Z):
    """Additive cubic-spline regression of y on several columns."""
    m = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return m.fit(Z, y).predict(Z)


def reliability(y, noise_var):
    rel = 1 - float(np.mean(noise_var)) / float(np.var(y))
    return {"reliability": round(rel, 4), "noise_share": round(1 - rel, 4), "noise_ceiling_rho": round(float(np.sqrt(max(rel, 0))), 4)}


def main():
    ap = cli(__doc__)
    ap.add_argument("--metadata", required=True, help="SMID metadata.csv (Hugging Face AIML-TUDA/smid)")
    ap.add_argument("--shuffles", type=int, default=200)
    a = parse(ap)
    cache = Cache(a.cache)
    lab = cache.labels(); meta = load_metadata(a.metadata, lab)
    T = targets(lab)
    val, mor, aro, mbv = T["valence"], T["morality"], lab.arousal.to_numpy(float), T["morality_beyond_valence"]
    dino, clip = cache.dinov2(), cache.clip()
    mbv_ctrl = mbv - oof(cache.controls(), mbv)
    res = {"n_images": len(lab), "valence_morality_rho": rho(val, mor)}

    print("definition", flush=True)
    variants = {"linear on valence": mor - LinearRegression().fit(val[:, None], mor).predict(val[:, None]),
                "spline on valence": mbv,
                "spline on valence + arousal": mor - spline_2d(mor, np.c_[val, aro])}
    res["definition"] = {name: {"share_of_morality_variance": round(float(r.var() / mor.var()), 4),
                                "rho_with_valence": rho(r, val), "dinov2": rho_ci(r, oof(dino, r))}
                         for name, r in variants.items()}

    print("reliability", flush=True)
    se_m, se_v = meta.moral_se.to_numpy(float), meta.valence_se.to_numpy(float)
    slope = spline_fit(mor, val)[1]
    noise_mbv = se_m ** 2 + slope ** 2 * se_v ** 2               # assumes independent rating errors
    p_ctrl = oof(dino, mbv_ctrl)
    res["reliability"] = {name: reliability(y, nz) | {"dinov2": rho(y, oof(dino, y) if name != "MBV after controls" else p_ctrl)}
                          for name, y, nz in [("valence", val, se_v ** 2), ("morality", mor, se_m ** 2),
                                              ("MBV", mbv, noise_mbv), ("MBV after controls", mbv_ctrl, noise_mbv)]}
    res["reliability"]["MBV after controls, errors fully correlated"] = reliability(
        mbv_ctrl, noise_mbv + 2 * np.abs(slope) * se_m * se_v)

    print("permutation", flush=True)
    res["permutation"] = permutation_test(dino, mbv_ctrl, a.shuffles)
    res["mbv_after_controls"] = {"dinov2": rho_ci(mbv_ctrl, p_ctrl), "clip": rho_ci(mbv_ctrl, oof(clip, mbv_ctrl))}
    print(" ", res["permutation"], flush=True)

    print("precision split", flush=True)
    precise = se_m <= np.median(se_m)
    res["precision_split"] = {"more_precise_half": rho_ci(mbv_ctrl[precise], p_ctrl[precise]),
                              "less_precise_half": rho_ci(mbv_ctrl[~precise], p_ctrl[~precise])}

    res["foundations"] = {f: {"morality": rho(mor, meta[f"{f}_mean"]), "MBV": rho(mbv, meta[f"{f}_mean"]),
                              "MBV after controls": rho(mbv_ctrl, meta[f"{f}_mean"])} for f in FOUNDATIONS}
    res["foundations"]["arousal"] = {"morality": rho(mor, aro), "MBV": rho(mbv, aro), "MBV after controls": rho(mbv_ctrl, aro)}
    save_json(res, f"{a.out}/e3_beyond_valence.json")


if __name__ == "__main__":
    main()
