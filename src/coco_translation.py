"""
Out-of-dataset DINOv2 -> CLIP translation.

The SMID analysis learned its DINOv2 -> CLIP linear map on SMID images (never on ratings). Here the map is learned
on 5,000 unrelated, unrated COCO val2017 photos and then applied, frozen, to SMID. If translated DINOv2 still
scores SMID images along CLIP's affect / moral language axes, the result cannot come from fitting on SMID.

  python coco_translation.py --features features --extra features_extra --out results_coco_translation.json
"""
import argparse, json, os, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_controls_clip import ALPHAS, direction, partial_spearman, unit

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"


def coco_features(img_dir, layer, batch=32):
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    from extract_controls_clip import as_emb
    dev, dt = "cuda", torch.float16
    paths = sorted(Path(img_dir).glob("*.jpg"))
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    dino = AutoModel.from_pretrained("facebook/dinov2-giant").to(dev, dt).eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev, dt).eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    D, C = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch)):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            h = dino(pixel_values=dproc(images=ims, return_tensors="pt")["pixel_values"].to(dev, dt),
                     output_hidden_states=True).hidden_states[layer + 1]          # same readout as extract_features.py
            D.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float().cpu().numpy())
            e = as_emb(clip.get_image_features(pixel_values=cproc(images=ims, return_tensors="pt")["pixel_values"].to(dev, dt))).float()
            C.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(D), np.concatenate(C)


def evaluate(Chn, C, lab, temb, tix, null=100, seed=0):
    """Same tests as sections C-F of analyze_controls_clip.py, for already-translated SMID embeddings Chn."""
    rng = np.random.default_rng(seed)
    n = len(lab)
    val, mor, aro = (lab[c].to_numpy(float) for c in ["valence", "morality", "arousal"])
    mor_bv = mor - np.polyval(np.polyfit(val, mor, 1), val)
    S = Chn @ C.T
    rank = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    out = {"translation": {"mean_cosine": round(float((Chn * C).sum(1).mean()), 4), "top1": round(float((rank == 0).mean()), 4),
                           "top5": round(float((rank < 5).mean()), 4),
                           "mean_cosine_random_pairs": round(float((Chn * C[rng.permutation(n)]).sum(1).mean()), 4)}}
    tmpl = tix["template"]
    tdir = lambda p, q: unit(temb[tmpl.format(p)] - temb[tmpl.format(q)])
    groups = {g: unit(sum(tdir(p, q) for p, q in prs)) for g, prs in tix["pairs"].items()}
    out["text_axes"] = {}
    items = [(f"{g}: {p} / {q}", tdir(p, q)) for g, prs in tix["pairs"].items() for p, q in prs] + [(f"{g} (mean axis)", d) for g, d in groups.items()]
    for name, d in items:
        sc = Chn @ d
        out["text_axes"][name] = {"valence": round(float(spearmanr(sc, val)[0]), 4), "morality": round(float(spearmanr(sc, mor)[0]), 4),
                                  "arousal": round(float(spearmanr(sc, aro)[0]), 4),
                                  "morality|valence": round(partial_spearman(sc, mor, val), 4),
                                  "valence|morality": round(partial_spearman(sc, val, mor), 4)}
    targets = {"valence": val, "morality": mor, "morality_beyond_valence": mor_bv, "arousal": aro}
    if null == 0:
        return out
    out["alignment"] = {}
    for t, y in targets.items():
        w = direction(Chn, y); nulls = [direction(Chn, rng.permutation(y)) for _ in range(null)]
        out["alignment"][t] = {g: {"cos": round(float(w @ d), 4), "z": round(float((w @ d - np.mean([v @ d for v in nulls])) / (np.std([v @ d for v in nulls]) + 1e-12)), 2)}
                               for g, d in groups.items()}
    words = tix["words"]; W = np.stack([temb[tmpl.format(w)] for w in words]); W = W - W.mean(0)
    out["words"] = {}
    for t in ["valence", "morality_beyond_valence", "arousal"]:
        s = W @ direction(Chn, targets[t]); o = np.argsort(s)
        out["words"][t] = {"top": [words[i] for i in o[::-1][:12]], "bottom": [words[i] for i in o[:12]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--model", default="G")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--coco-dir", default="coco")
    ap.add_argument("--cache", default=None, help="folder to cache COCO features (e.g. on Drive)")
    ap.add_argument("--out", default="results_coco_translation.json")
    ap.add_argument("--null", type=int, default=100)
    a = ap.parse_args()

    cache = Path(a.cache or a.coco_dir); cache.mkdir(parents=True, exist_ok=True)
    fd, fc = cache / f"coco_dinov2{a.model}_L{a.layer + 1}.npy", cache / "coco_clip.npy"
    if not (fd.exists() and fc.exists()):
        img_dir = Path(a.coco_dir) / "val2017"
        if not img_dir.exists():
            os.makedirs(a.coco_dir, exist_ok=True)
            z = Path(a.coco_dir) / "val2017.zip"
            assert os.system(f"wget -q -O {z} {COCO_URL}") == 0, "COCO download failed"
            zipfile.ZipFile(z).extractall(a.coco_dir); z.unlink()
        Dc, Cc = coco_features(img_dir, a.layer)
        np.save(fd, Dc.astype(np.float16)); np.save(fc, Cc.astype(np.float32))
    Dc, Cc = np.load(fd).astype(np.float32), np.load(fc).astype(np.float64)
    print("COCO features", Dc.shape, Cc.shape, flush=True)

    lab = pd.read_csv(f"{a.features}/smid_labels.csv")
    dino = np.asarray(np.load(f"{a.features}/{a.model}_smid.npy", mmap_mode="r")[a.layer], dtype=np.float32)
    C = np.load(f"{a.extra}/clip_img.npy").astype(np.float64)
    T = np.load(f"{a.extra}/clip_text.npy").astype(np.float64)
    tix = json.load(open(f"{a.extra}/text_index.json"))
    temb = {t: T[i] for i, t in enumerate(tix["texts"])}

    m = make_pipeline(StandardScaler(), PCA(512, random_state=0), RidgeCV(alphas=ALPHAS)).fit(Dc, Cc)
    Chat = m.predict(dino); Chn = Chat / np.linalg.norm(Chat, axis=1, keepdims=True)
    res = {"map_fit_on": f"COCO val2017 ({len(Dc)} images), applied unchanged to SMID ({len(lab)} images)",
           "smid": evaluate(Chn, C, lab, temb, tix, null=a.null),
           "clip_reference": {k: v for k, v in evaluate(C, C, lab, temb, tix, null=0).items() if k == "text_axes"}}
    t = res["smid"]
    print("translation", t["translation"], flush=True)
    print("axes", {k: v for k, v in t["text_axes"].items() if "mean axis" in k}, flush=True)
    print("alignment z", {k: {g: v["z"] for g, v in o.items()} for k, o in t["alignment"].items()}, flush=True)
    print("words valence", t["words"]["valence"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
