"""
(a2) Is DINOv2's valence/morality signal more than low-level statistics and object identity?
(b3) Does DINOv2's good-bad axis, translated into CLIP space, line up with language about good and bad?

Inputs: DINOv2 features (one layer of {model}_smid.npy) + smid_labels.csv, and the outputs of extract_controls_clip.py.
Output: one JSON with every number, printed and saved.

  python analyze_controls_clip.py --features features --extra features_extra --model G --layer 36 --out results_controls_clip.json
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-1, 5, 13)
RNG = np.random.default_rng(0)


def oof(X, y, k=256, folds=5, seed=0):
    """Out-of-fold predictions of y (1-D or 2-D) from X: scale -> PCA(k) -> ridge."""
    kf = KFold(folds, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        kk = min(k, X.shape[1], len(tr) - 1)
        m = make_pipeline(StandardScaler(), PCA(kk, random_state=0), RidgeCV(alphas=ALPHAS))
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return pred


def rho_ci(y, p, n=500):
    r = spearmanr(y, p)[0]
    bs = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        bs.append(spearmanr(y[i], p[i])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"rho": round(float(r), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (all 1-D)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z).reshape(-1, 1)
    ex = rx - LinearRegression().fit(rz, rx).predict(rz)
    ey = ry - LinearRegression().fit(rz, ry).predict(rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def unit(v):
    return v / np.linalg.norm(v)


def direction(X, y):
    """Ridge direction of y in the (centred) space of X, in X's own coordinates."""
    Xc = X - X.mean(0)
    return unit(RidgeCV(alphas=ALPHAS).fit(Xc, y - y.mean()).coef_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--model", default="G")
    ap.add_argument("--layer", type=int, default=36, help="0-based layer index (36 = DINOv2-G's best valence layer)")
    ap.add_argument("--out", default="results_controls_clip.json")
    ap.add_argument("--null", type=int, default=100, help="label shuffles for the direction-alignment null")
    a = ap.parse_args()

    lab = pd.read_csv(f"{a.features}/smid_labels.csv")
    dino = np.asarray(np.load(f"{a.features}/{a.model}_smid.npy", mmap_mode="r")[a.layer], dtype=np.float32)
    E = {k: np.load(f"{a.extra}/{k}.npy").astype(np.float32) for k in ["lowlevel", "sup_logits", "sup_feats", "clip_img"]}
    T = np.load(f"{a.extra}/clip_text.npy").astype(np.float64)
    tix = json.load(open(f"{a.extra}/text_index.json"))
    temb = {t: T[i] for i, t in enumerate(tix["texts"])}
    tmpl = tix["template"]
    n = len(lab)
    assert dino.shape[0] == n and all(v.shape[0] == n for v in E.values())

    val, mor, aro = (lab[c].to_numpy(np.float64) for c in ["valence", "morality", "arousal"])
    lr = LinearRegression()
    mor_bv = mor - lr.fit(val[:, None], mor).predict(val[:, None])   # morality beyond valence
    val_bm = val - lr.fit(mor[:, None], val).predict(mor[:, None])   # valence beyond morality
    targets = {"valence": val, "morality": mor, "arousal": aro, "morality_beyond_valence": mor_bv, "valence_beyond_morality": val_bm}

    sets = {
        "lowlevel": E["lowlevel"],
        "objects": E["sup_logits"],
        "supervised_vit": E["sup_feats"],
        "lowlevel+objects": np.hstack([E["lowlevel"], E["sup_logits"]]),
        "lowlevel+objects+supervised": np.hstack([E["lowlevel"], E["sup_logits"], E["sup_feats"]]),
        "dinov2": dino,
        "clip": E["clip_img"],
    }
    res = {"n_images": n, "model": a.model, "layer": a.layer + 1,
           "rating_corr": {"valence~morality": round(float(spearmanr(val, mor)[0]), 4),
                           "valence~arousal": round(float(spearmanr(val, aro)[0]), 4)}}

    # ---- A. what each feature set predicts on its own
    print("A. probes per feature set", flush=True)
    preds = {}
    res["probe_rho"] = {}
    for s, X in sets.items():
        res["probe_rho"][s] = {}
        for t, y in targets.items():
            p = oof(X, y); preds[(s, t)] = p
            res["probe_rho"][s][t] = rho_ci(y, p)
        print(" ", s, {t: v["rho"] for t, v in res["probe_rho"][s].items()}, flush=True)

    # ---- B. beyond controls: residualise the rating on control features (cross-fitted), probe the residual
    print("B. beyond controls", flush=True)
    res["beyond_controls"] = {}
    for c in ["lowlevel", "objects", "lowlevel+objects", "lowlevel+objects+supervised"]:
        res["beyond_controls"][c] = {}
        for t in ["valence", "morality", "arousal"]:
            y = targets[t]
            yres = y - preds[(c, t)]
            out = {"controls_alone": res["probe_rho"][c][t]["rho"]}
            for s in ["dinov2", "clip"]:
                out[f"{s}_on_residual"] = rho_ci(yres, oof(sets[s], yres))
                out[f"controls+{s}"] = round(float(spearmanr(y, oof(np.hstack([sets[c], sets[s]]), y))[0]), 4)
            res["beyond_controls"][c][t] = out
        print(" ", c, {t: v["dinov2_on_residual"]["rho"] for t, v in res["beyond_controls"][c].items()}, flush=True)

    # ---- C. translate DINOv2 -> CLIP image space (linear, cross-fitted)
    print("C. DINOv2 -> CLIP translation", flush=True)
    C = E["clip_img"].astype(np.float64)
    Chat = oof(dino, C, k=512)
    Chn = Chat / np.linalg.norm(Chat, axis=1, keepdims=True)
    cos = (Chn * C).sum(1)
    S = Chn @ C.T
    rank = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    res["translation"] = {"mean_cosine": round(float(cos.mean()), 4), "top1": round(float((rank == 0).mean()), 4),
                          "top5": round(float((rank < 5).mean()), 4), "chance_top1": round(1 / n, 5),
                          "mean_cosine_random_pairs": round(float((Chn * C[RNG.permutation(n)]).sum(1).mean()), 4)}
    print(" ", res["translation"], flush=True)

    # ---- D. language axes: score images along CLIP text directions, real CLIP vs translated DINOv2
    print("D. text axes", flush=True)
    def tdir(pos, neg):
        return unit(temb[tmpl.format(pos)] - temb[tmpl.format(neg)])
    groups = {g: unit(sum(tdir(p, q) for p, q in prs)) for g, prs in tix["pairs"].items()}
    res["text_axes"] = {}
    for space, Z in [("clip", C), ("dinov2_translated", Chn)]:
        res["text_axes"][space] = {}
        items = [(f"{g}: {p} / {q}", tdir(p, q)) for g, prs in tix["pairs"].items() for p, q in prs] + \
                [(f"{g} (mean axis)", d) for g, d in groups.items()]
        for name, d in items:
            sc = Z @ d
            res["text_axes"][space][name] = {
                "valence": round(float(spearmanr(sc, val)[0]), 4), "morality": round(float(spearmanr(sc, mor)[0]), 4),
                "arousal": round(float(spearmanr(sc, aro)[0]), 4),
                "morality|valence": round(partial_spearman(sc, mor, val), 4),
                "valence|morality": round(partial_spearman(sc, val, mor), 4)}
        print(" ", space, {k: v for k, v in res["text_axes"][space].items() if "mean axis" in k}, flush=True)

    # ---- E. alignment of rating directions (learned on images) with the language axes
    print("E. direction alignment", flush=True)
    res["alignment"] = {}
    for space, Z in [("clip", C), ("dinov2_translated", Chn)]:
        res["alignment"][space] = {}
        for t in ["valence", "morality", "morality_beyond_valence", "arousal"]:
            w = direction(Z, targets[t])
            null = [direction(Z, RNG.permutation(targets[t])) for _ in range(a.null)]
            out = {}
            for g, d in groups.items():
                obs = float(w @ d); nd = np.array([v @ d for v in null])
                out[g] = {"cos": round(obs, 4), "null_mean": round(float(nd.mean()), 4), "null_sd": round(float(nd.std()), 4),
                          "z": round(float((obs - nd.mean()) / (nd.std() + 1e-12)), 2)}
            res["alignment"][space][t] = out
        print(" ", space, {t: {g: v["z"] for g, v in o.items()} for t, o in res["alignment"][space].items()}, flush=True)

    # ---- F. name the axes: which words lie furthest along each direction
    words = tix["words"]
    W = np.stack([temb[tmpl.format(w)] for w in words]); W = W - W.mean(0)
    res["words"] = {}
    for space, Z in [("clip", C), ("dinov2_translated", Chn)]:
        for t in ["valence", "morality_beyond_valence", "arousal"]:
            s = W @ direction(Z, targets[t]); o = np.argsort(s)
            res["words"][f"{space}:{t}"] = {"top": [words[i] for i in o[::-1][:12]], "bottom": [words[i] for i in o[:12]]}
    print("F. words", json.dumps(res["words"]["dinov2_translated:valence"]), flush=True)

    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
