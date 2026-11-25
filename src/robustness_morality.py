"""
Robustness of "morality beyond valence".

The main analysis removes valence from the morality rating with a straight line. If the two are related
non-linearly, part of valence survives in that residual. Here valence (and optionally arousal) are removed
with a flexible spline fit, and we ask how much of the remaining variance is rating noise, using SMID's
per-image standard errors.

  python robustness_morality.py --features features --extra features_extra --metadata metadata.csv --out results_morality_robustness.json
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from analyze_controls_clip import oof, rho_ci


def residual(y, Z, kind):
    if kind == "linear":
        m = LinearRegression()
    else:
        m = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return y - m.fit(Z, y).predict(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--metadata", required=True, help="SMID metadata.csv with *_se columns")
    ap.add_argument("--model", default="G")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--out", default="results_morality_robustness.json")
    a = ap.parse_args()

    lab = pd.read_csv(f"{a.features}/smid_labels.csv")
    meta = pd.read_csv(a.metadata).assign(image=lambda d: d.file_name.str.replace(r"\.[^.]+$", "", regex=True))
    meta = lab[["image"]].merge(meta, on="image", how="left")
    assert meta.moral_mean.notna().all() and np.allclose(meta.moral_mean, lab.morality)

    dino = np.asarray(np.load(f"{a.features}/{a.model}_smid.npy", mmap_mode="r")[a.layer], dtype=np.float32)
    sets = {"dinov2": dino, "clip": np.load(f"{a.extra}/clip_img.npy"),
            "objects": np.load(f"{a.extra}/sup_logits.npy"), "supervised_vit": np.load(f"{a.extra}/sup_feats.npy"),
            "lowlevel": np.load(f"{a.extra}/lowlevel.npy")}
    val, mor, aro = (lab[c].to_numpy(float) for c in ["valence", "morality", "arousal"])
    se2 = meta.moral_se.to_numpy(float) ** 2

    variants = {
        "linear(valence)": residual(mor, val[:, None], "linear"),
        "spline(valence)": residual(mor, val[:, None], "spline"),
        "spline(valence)+spline(arousal)": residual(mor, np.c_[val, aro], "spline"),
    }
    res = {"note": "rho: out-of-fold probe prediction vs the morality residual; noise_share: mean squared SE of the morality "
                   "rating divided by the residual's variance (share of the residual that is rating noise)"}
    for name, r in variants.items():
        out = {"residual_var_share_of_morality": round(float(r.var() / mor.var()), 4),
               "noise_share": round(float(se2.mean() / r.var()), 4),
               "corr_with_valence": round(float(spearmanr(r, val)[0]), 4)}
        for s, X in sets.items():
            out[s] = rho_ci(r, oof(X, r), n=300)
        res[name] = out
        print(name, {k: (v["rho"] if isinstance(v, dict) else v) for k, v in out.items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
