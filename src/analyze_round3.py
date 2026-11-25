"""
Round 3: tests that decide the narrative.

  T1  Morality beyond valence, after controls: does DINOv2 predict the part of morality that valence does not explain,
      once low-level statistics and object identity are removed too?
  T2a Unsupervised structure: is the good-bad axis among DINOv2's main directions of variation, and do label-free
      clusters of DINOv2 features separate it?
  T2b Shared across models: a direction learned in CLIP, read out from DINOv2 features mapped into CLIP space (and back),
      with maps fit on unrelated, unrated photos.
  T3  Random-init DINOv2-G (architecture only).
  T5  DINOv2 -> CLIP map fit on COCO + Places365 (larger, more varied), then the language tests.

  python analyze_round3.py --features features --extra features_extra --coco coco_features --round3 round3 --out results_round3.json
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from analyze_controls_clip import ALPHAS, oof, rho_ci
from coco_translation import evaluate

RNG = np.random.default_rng(0)


def spline_residual(y, z):
    m = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return y - m.fit(z[:, None], y).predict(z[:, None])


def fit_map(Xs, Ys, k=512):
    """Linear map X -> Y (PCA on X when wide)."""
    steps = [StandardScaler()] + ([PCA(k, random_state=0)] if Xs.shape[1] > k else []) + [RidgeCV(alphas=ALPHAS)]
    return make_pipeline(*steps).fit(Xs, Ys)


def l2n(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def eta2(y, lab):
    """Share of the variance of y explained by cluster membership."""
    m = y.mean(); tot = ((y - m) ** 2).sum()
    return float(sum(((y[lab == c].mean() - m) ** 2) * (lab == c).sum() for c in np.unique(lab)) / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--coco", required=True, help="folder with coco_dinov2G_L37.npy and coco_clip.npy")
    ap.add_argument("--round3", required=True)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--null", type=int, default=100)
    ap.add_argument("--out", default="results_round3.json")
    a = ap.parse_args()

    lab = pd.read_csv(f"{a.features}/smid_labels.csv")
    val, mor, aro = (lab[c].to_numpy(float) for c in ["valence", "morality", "arousal"])
    mbv = spline_residual(mor, val)
    dino = np.asarray(np.load(f"{a.features}/G_smid.npy", mmap_mode="r")[a.layer], dtype=np.float32)
    ll = np.load(f"{a.extra}/lowlevel.npy"); obj = np.load(f"{a.extra}/sup_logits.npy"); sup = np.load(f"{a.extra}/sup_feats.npy")
    C = np.load(f"{a.extra}/clip_img.npy").astype(np.float64)
    T = np.load(f"{a.extra}/clip_text.npy").astype(np.float64); tix = json.load(open(f"{a.extra}/text_index.json"))
    temb = {t: T[i] for i, t in enumerate(tix["texts"])}
    targets = {"valence": val, "morality": mor, "morality_beyond_valence": mbv}
    res = {"n_images": len(lab), "layer": a.layer + 1, "mbv_definition": "morality minus a cubic-spline fit on valence"}

    import os
    has_rand = os.path.exists(f"{a.round3}/random_init_G_smid.npy")
    has_places = os.path.exists(f"{a.round3}/places_clip.npy")
    res["available"] = {"random_init": has_rand, "places": has_places}
    res["T3_trained_reference"] = {t: rho_ci(y, oof(dino, y), n=300) for t, y in targets.items()}
    rand = None
    if has_rand:
        rand = np.load(f"{a.round3}/random_init_G_smid.npy").astype(np.float32)
        rlayers = [int(l) for l in np.load(f"{a.round3}/random_init_layers.npy")]
        print("T3 random init", flush=True)
        res["T3_random_init"] = {f"layer_{l + 1}": {t: rho_ci(y, oof(rand[i], y), n=300) for t, y in targets.items()} for i, l in enumerate(rlayers)}
        print(" ", {k: {t: v["rho"] for t, v in d.items()} for k, d in res["T3_random_init"].items()}, flush=True)
        best_rand = max(range(len(rlayers)), key=lambda i: res["T3_random_init"][f"layer_{rlayers[i] + 1}"]["valence"]["rho"])
        rand = rand[best_rand]

    # ---- T1 morality beyond valence after controls
    print("T1 morality beyond valence after controls", flush=True)
    controls = {"lowlevel": ll, "lowlevel+objects": np.hstack([ll, obj]), "lowlevel+objects+supervised": np.hstack([ll, obj, sup])}
    res["T1"] = {}
    for cname, X in controls.items():
        res["T1"][cname] = {}
        for t, y in targets.items():
            pc = oof(X, y); r = y - pc
            res["T1"][cname][t] = {"controls_alone": round(float(spearmanr(y, pc)[0]), 4),
                                   "dinov2_on_residual": rho_ci(r, oof(dino, r), n=300),
                                   "clip_on_residual": rho_ci(r, oof(C, r), n=300)}
            if rand is not None:
                res["T1"][cname][t]["random_init_on_residual"] = rho_ci(r, oof(rand, r), n=300)
        print(" ", cname, {t: v["dinov2_on_residual"]["rho"] for t, v in res["T1"][cname].items()}, flush=True)

    # ---- T2a unsupervised structure of DINOv2
    print("T2a unsupervised structure", flush=True)
    Z = StandardScaler().fit_transform(dino)
    pca = PCA(256, random_state=0).fit(Z); S = pca.transform(Z); ev = pca.explained_variance_ratio_
    comparators = {"brightness": ll[:, 12], "contrast": ll[:, 13], "edge_density": ll[:, 14], "colorfulness": ll[:, 16], "arousal": aro}
    allt = {**targets, **comparators}
    ks = [1, 2, 3, 5, 10, 20, 50, 100, 256]
    res["T2a"] = {"explained_variance_top10": [round(float(v), 4) for v in ev[:10]], "per_target": {}}
    for t, y in allt.items():
        pcr = np.array([spearmanr(S[:, j], y)[0] for j in range(S.shape[1])])
        order = np.argsort(-np.abs(pcr))
        curve = {str(k): round(float(spearmanr(y, oof(S[:, :k], y, k=k))[0]), 4) for k in ks}
        res["T2a"]["per_target"][t] = {
            "top_pcs": [{"pc": int(j + 1), "rho": round(float(pcr[j]), 4), "var_explained": round(float(ev[j]), 4)} for j in order[:5]],
            "first_pc_with_abs_rho_over_0.3": next((int(j + 1) for j in range(len(pcr)) if abs(pcr[j]) > 0.3), None),
            "rho_from_top_k_pcs": curve}
        print(" ", t, "top PCs", [(d["pc"], d["rho"]) for d in res["T2a"]["per_target"][t]["top_pcs"][:3]], "curve", curve, flush=True)
    res["T2a"]["kmeans_eta2"] = {}
    reps = {"dinov2": S[:, :50], "lowlevel": PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(ll)),
            "supervised_vit": PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(sup)),
            "clip": PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(C))}
    if rand is not None:
        reps["random_init"] = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(rand))
    for rep, F in reps.items():
        res["T2a"]["kmeans_eta2"][rep] = {}
        for k in [2, 5, 10, 20, 50]:
            labk = KMeans(k, n_init=4, random_state=0).fit_predict(F)
            null = np.mean([eta2(val, RNG.permutation(labk)) for _ in range(30)])
            res["T2a"]["kmeans_eta2"][rep][str(k)] = {t: round(eta2(y, labk), 4) for t, y in targets.items()} | {"null_valence": round(float(null), 4)}
        print(" ", rep, {k: v["valence"] for k, v in res["T2a"]["kmeans_eta2"][rep].items()}, flush=True)

    # ---- maps fit on unrated photos
    Dc = np.load(f"{a.coco}/coco_dinov2G_L37.npy").astype(np.float32); Cc = np.load(f"{a.coco}/coco_clip.npy").astype(np.float64)
    maps = {"coco": (Dc, Cc)}
    if has_places:
        Dp = np.load(f"{a.round3}/places_dinov2G_L37.npy").astype(np.float32); Cp = np.load(f"{a.round3}/places_clip.npy").astype(np.float64)
        maps["coco+places"] = (np.vstack([Dc, Dp]), np.vstack([Cc, Cp]))

    # ---- T5 language tests with each map
    print("T5 language tests", flush=True)
    res["T5"] = {}
    fitted = {}
    for name, (Dm, Cm) in maps.items():
        d2c = fit_map(Dm, Cm); fitted[name] = d2c
        Chn = l2n(d2c.predict(dino))
        res["T5"][name] = {"n_map_images": int(len(Dm)), **evaluate(Chn, C, lab, temb, tix, null=a.null)}
        ax = res["T5"][name]["text_axes"]
        print(" ", name, res["T5"][name]["translation"], {k: (v["valence|morality"], v["morality|valence"]) for k, v in ax.items() if "mean axis" in k}, flush=True)

    # ---- T2b cross-model direction transfer (map fit on unrated photos, never on SMID or ratings)
    mname = "coco+places" if has_places else "coco"
    print("T2b cross-model transfer, map:", mname, flush=True)
    res["T2b_map"] = mname
    Dm, Cm = maps[mname]
    d2c = fitted[mname]; c2d = fit_map(Cm, Dm.astype(np.float64))
    Dto_c = l2n(d2c.predict(dino)); Cto_d = c2d.predict(C)
    kf = KFold(5, shuffle=True, random_state=0)
    res["T2b"] = {}
    for t, y in targets.items():
        p_cd, p_dc = np.zeros_like(y), np.zeros_like(y)
        for tr, te in kf.split(C):
            pc = make_pipeline(StandardScaler(), PCA(256, random_state=0), RidgeCV(alphas=ALPHAS)).fit(C[tr], y[tr])
            p_cd[te] = pc.predict(Dto_c[te])                       # CLIP's direction, read from mapped DINOv2
            pd_ = make_pipeline(StandardScaler(), PCA(256, random_state=0), RidgeCV(alphas=ALPHAS)).fit(dino[tr], y[tr])
            p_dc[te] = pd_.predict(Cto_d[te].astype(np.float32))   # DINOv2's direction, read from mapped CLIP
        res["T2b"][t] = {"clip_direction_on_mapped_dinov2": rho_ci(y, p_cd, n=300),
                         "dinov2_direction_on_mapped_clip": rho_ci(y, p_dc, n=300),
                         "clip_within": rho_ci(y, oof(C, y), n=300),
                         "dinov2_within": res["T3_trained_reference"][t]}
        print(" ", t, {k: v["rho"] for k, v in res["T2b"][t].items()}, flush=True)

    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
