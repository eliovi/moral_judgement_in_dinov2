"""
Numbers and example images for the write-up. CPU only (downloads SMID images for thumbnails, then deletes them).

  W  What DINOv2's good/bad directions mean in words. DINOv2-g features are mapped into CLIP space with a linear map
     fit ONLY on unrated COCO photos (no SMID images, no ratings). For each word, we correlate (over SMID images) the
     similarity between the mapped image and the word with DINOv2's own held-out probe score. Words come from a fixed
     vocabulary of virtues, vices, positive/negative emotions and everyday things.
  P  Is the direction shared with CLIP, and does sharing grow with size and depth? A label-free DINOv2->CLIP map is
     cross-fitted on SMID images (ratings never used for the map); CLIP's rating probe is then applied to mapped,
     held-out DINOv2 features. Also: how much of each rating sits in DINOv2's own top-10 principal components.
  E  Example images: a strip ordered by DINOv2's predicted morality, and two off-diagonal groups
     (morally good but unpleasant; pleasant but not morally good), chosen where DINOv2 and people agree.

  python story_assets.py --features features --extra features_extra --coco coco_features --smid-dir data/smid \
      --metadata metadata.csv --out story
"""
import argparse, base64, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from analyze_controls_clip import ALPHAS, oof

VOCAB = {
    "virtue": ["kindness", "compassion", "care", "generosity", "charity", "honesty", "fairness", "justice", "respect",
               "loyalty", "courage", "mercy", "forgiveness", "helping others", "a rescue", "volunteering", "sharing",
               "protection", "gratitude", "tenderness", "comfort", "self-sacrifice", "heroism", "nurturing",
               "cooperation", "solidarity", "dignity", "integrity", "decency", "humility"],
    "vice": ["cruelty", "violence", "abuse", "murder", "theft", "cheating", "lying", "betrayal", "corruption", "greed",
             "hatred", "torture", "bullying", "harassment", "exploitation", "vandalism", "terrorism", "assault",
             "robbery", "fraud", "neglect", "humiliation", "discrimination", "racism", "slavery", "oppression",
             "revenge", "injustice", "selfishness", "arrogance"],
    "positive emotion": ["joy", "happiness", "fun", "excitement", "calm", "pleasure", "delight", "relaxation",
                         "cheerfulness", "amusement", "laughter", "a party", "a vacation", "playfulness", "beauty",
                         "a celebration", "contentment", "a good meal", "a sunny day", "a game"],
    "negative emotion": ["sadness", "fear", "pain", "disgust", "boredom", "anxiety", "grief", "loneliness", "anger",
                         "misery", "a disaster", "illness", "dirt", "decay", "a storm", "an accident", "garbage",
                         "a funeral", "poverty", "exhaustion"],
    "thing": ["a dog", "a cat", "a horse", "a bird", "a car", "a bicycle", "a phone", "a computer", "a chair", "a table",
              "a cake", "a pizza", "a flower", "a tree", "a mountain", "a beach", "a city street", "a kitchen",
              "an office", "a train", "an airplane", "a boat", "a bridge", "a building", "a book", "a clock", "a lamp",
              "a shoe", "a bottle", "a market", "a road", "a river", "a forest", "a field", "a stadium", "a classroom",
              "a hospital", "a church", "a uniform", "a flag", "a crowd", "a child", "an old man", "a woman", "a soldier",
              "a doctor", "a nurse", "a police officer", "a weapon", "money"]}


def thumb(path, size=256):
    from PIL import Image
    im = Image.open(path).convert("RGB"); im.thumbnail((size, size))
    b = io.BytesIO(); im.save(b, "JPEG", quality=82)
    return base64.b64encode(b.getvalue()).decode()


def map_fit(X, Y, k=256):
    return make_pipeline(StandardScaler(), PCA(min(k, X.shape[1], len(X) - 1), random_state=0), RidgeCV(alphas=ALPHAS)).fit(X, Y)


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def words_section(a, dino, val, targets, pred, res):
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    from extract_controls_clip import as_emb
    words = [(w, cat) for cat, ws in VOCAB.items() for w in ws]
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        T = unit(as_emb(model.get_text_features(**tok([f"a photo of {w}" for w, _ in words], padding=True, return_tensors="pt"))).numpy().astype(np.float64))
    del model
    Dc = np.load(f"{a.coco}/coco_dinov2G_L37.npy").astype(np.float32); Cc = np.load(f"{a.coco}/coco_clip.npy").astype(np.float64)
    M = unit(map_fit(Dc, Cc).predict(dino))
    sims = M @ T.T  # images x words
    res["W"] = {"map": "linear DINOv2-g L37 -> CLIP ViT-L/14, fit on 5,000 unrated COCO val2017 photos",
                "map_check_rho_valence_from_clip_probe": round(float(spearmanr(val, oof(M, val))[0]), 4)}
    cats = np.array([c for _, c in words])
    for t in targets:
        s = np.array([spearmanr(sims[:, j], pred[t])[0] for j in range(len(words))])
        order = np.argsort(-s)
        auc = lambda c1, c2: round(float(roc_auc_score(np.r_[np.ones((cats == c1).sum()), np.zeros((cats == c2).sum())],
                                                        np.r_[s[cats == c1], s[cats == c2]])), 4)
        res["W"][t] = {"top": [(words[j][0], words[j][1], round(float(s[j]), 4)) for j in order[:15]],
                       "bottom": [(words[j][0], words[j][1], round(float(s[j]), 4)) for j in order[::-1][:15]],
                       "auc_virtue_vs_vice": auc("virtue", "vice"),
                       "auc_virtue_vs_positive_emotion": auc("virtue", "positive emotion"),
                       "auc_negative_emotion_vs_vice": auc("negative emotion", "vice"),
                       "mean_by_category": {c: round(float(s[cats == c].mean()), 4) for c in VOCAB},
                       "all": {words[j][0]: round(float(s[j]), 4) for j in range(len(words))}}
        print(" ", t, [w for w, _, _ in res["W"][t]["top"][:8]], "|", [w for w, _, _ in res["W"][t]["bottom"][:8]],
              {k: v for k, v in res["W"][t].items() if k.startswith("auc")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ["features", "extra", "coco", "smid_dir", "metadata"]:
        ap.add_argument("--" + k.replace("_", "-"), required=k != "smid_dir")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--out", default="story")
    ap.add_argument("--skip-words", action="store_true", help="testing only")
    ap.add_argument("--only-examples", action="store_true", help="skip W and P (already computed), make the example images")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    save = lambda: json.dump(res, open(out / ("story_examples_meta.json" if a.only_examples else "story_numbers.json"), "w"), indent=1)

    lab = pd.read_csv(f"{a.features}/smid_labels.csv")
    val, mor = lab.valence.to_numpy(float), lab.morality.to_numpy(float)
    sp = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(val[:, None], mor)
    mbv = mor - sp.predict(val[:, None])
    targets = {"valence": val, "morality": mor, "morality_beyond_valence": mbv}
    G = np.load(f"{a.features}/G_smid.npy", mmap_mode="r")
    dino = np.asarray(G[a.layer], dtype=np.float32)
    C = np.load(f"{a.extra}/clip_img.npy").astype(np.float64)
    pred = {t: oof(dino, y) for t, y in targets.items()}  # DINOv2's own held-out scores
    res["dinov2_probe_rho"] = {t: round(float(spearmanr(y, pred[t])[0]), 4) for t, y in targets.items()}

    # ---- W: words, through a map fit on COCO only
    print("W words", flush=True)
    if not (a.skip_words or a.only_examples):
        words_section(a, dino, val, targets, pred, res)
        save()

    # ---- P: shared with CLIP, by size and depth (label-free map, cross-fitted)
    print("P shared direction", flush=True)
    res["P"] = []
    kf = list(KFold(5, shuffle=True, random_state=0).split(C))
    for m in ([] if a.only_examples else ["S", "B", "L", "G"]):
        F = np.load(f"{a.features}/{m}_smid.npy", mmap_mode="r"); L = F.shape[0]
        for d in [0.25, 0.5, 0.75, 0.9, 1.0]:
            l = max(0, int(round(d * L)) - 1)
            X = np.asarray(F[l], dtype=np.float32)
            tr_pred = {t: np.zeros(len(X)) for t in targets}
            for tr, te in kf:
                Mte = unit(map_fit(X[tr], C[tr]).predict(X[te]))      # map fit on training images only, no ratings
                for t, y in targets.items():
                    tr_pred[t][te] = map_fit(C[tr], y[tr]).predict(Mte)  # CLIP's rating direction, read from mapped DINOv2
            Z = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(X))
            row = {"model": m, "layer": l + 1, "depth": round((l + 1) / L, 3)}
            for t, y in targets.items():
                row[f"clip_direction_in_dinov2_{t}"] = round(float(spearmanr(y, tr_pred[t])[0]), 4)
                row[f"top10pc_{t}"] = round(float(spearmanr(y, oof(Z, y, k=10))[0]), 4)
            res["P"].append(row); print(" ", row, flush=True)
            save()
    if not a.only_examples:
        res["P_clip_within"] = {t: round(float(spearmanr(y, oof(C, y))[0]), 4) for t, y in targets.items()}
        save()

    # ---- E: example images
    if a.smid_dir:
        print("E examples", flush=True)
        from extract_features import load_smid
        paths, lab2 = load_smid(a.smid_dir, {})
        assert list(lab2.image) == list(lab.image)
        meta = pd.read_csv(a.metadata).assign(image=lambda d: d.file_name.str.replace(r"\.[^.]+$", "", regex=True))
        meta = lab[["image"]].merge(meta, on="image", how="left")
        pct = lambda x: pd.Series(x).rank(pct=True).to_numpy()
        pm, hm = pct(pred["morality"]), pct(mor)
        pb, hb = pct(pred["morality_beyond_valence"]), pct(mbv)
        purity = pct(meta.purity_mean.to_numpy(float)) if "purity_mean" in meta else np.zeros(len(lab))
        ok = purity < 0.9                                  # leave out the images most rated as purity-relevant (sexual / disgust)
        cand = {}
        for lo, hi in [(0, .06), (.2, .3), (.45, .55), (.7, .8), (.94, 1)]:
            idx = np.flatnonzero(ok & (pm >= lo) & (pm <= hi) & (np.abs(pm - hm) < .1))
            cand[f"strip_{lo:.2f}-{hi:.2f}"] = list(idx[np.argsort(pm[idx])][:8])
        idx = np.flatnonzero(ok & (pb > .85) & (hb > .75) & (val < 2.6))
        cand["good_but_unpleasant"] = list(idx[np.argsort(-pb[idx])][:12])
        idx = np.flatnonzero(ok & (pb < .15) & (hb < .25) & (val > 3.4))
        cand["pleasant_not_moral"] = list(idx[np.argsort(pb[idx])][:12])
        ex = {}
        for k, ids in cand.items():
            ex[k] = [{"image": lab.image[i], "valence": round(float(val[i]), 2), "morality": round(float(mor[i]), 2),
                      "dinov2_morality_pct": round(float(pm[i]), 3), "dinov2_mbv_pct": round(float(pb[i]), 3),
                      "jpg": thumb(paths[i])} for i in ids]
        json.dump(ex, open(out / "story_examples.json", "w"))
        res["E_counts"] = {k: len(v) for k, v in ex.items()}
        save()
    print("saved", out)


if __name__ == "__main__":
    main()
