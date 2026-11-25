"""
Extract per-layer DINOv2 features for (a) SMID images with human ratings and
(b) a reference semantic dataset (any ImageFolder: root/<class>/<img>).

Features per layer = concat(CLS token, mean of patch tokens), as in DINOv2's linear eval.
Saved as float16 arrays of shape [n_layers, n_images, 2*hidden_dim].

Usage:
  python extract_features.py --smid-dir data/smid --semantic-dir data/imagenette2-320/train \
      --models S B L G --out features
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

MODELS = {
    "S": "facebook/dinov2-small",
    "B": "facebook/dinov2-base",
    "L": "facebook/dinov2-large",
    "G": "facebook/dinov2-giant",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RATINGS = ["valence", "arousal", "morality"]


# ----------------------------------------------------------------------------- SMID
def _find_rating_col(cols, key):
    """Pick the column for a rating: contains key, prefer 'mean', avoid sd/se/n columns."""
    keys = {"morality": ["moral"], "valence": ["valence"], "arousal": ["arousal"]}[key]
    cands = [c for c in cols if any(k in c.lower() for k in keys)]
    cands = [c for c in cands if not re.search(r"(^|_)(sd|se|std|n|var)($|_)", c.lower())]
    if not cands:
        return None
    mean = [c for c in cands if "mean" in c.lower()]
    return (mean or cands)[0]


def load_smid(root, overrides):
    root = Path(root)
    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT}
    if not images:
        raise SystemExit(f"No images found under {root}")

    csvs = sorted(root.rglob("*.csv"))
    meta = None
    for c in csvs:  # prefer a norms/metadata file, else concat train+test
        if any(k in c.name.lower() for k in ["norm", "metadata"]):
            meta = pd.read_csv(c)
            print(f"[smid] metadata: {c}")
            break
    if meta is None:
        parts = [pd.read_csv(c) for c in csvs if c.stem.lower() in ("train", "test")]
        if not parts:
            raise SystemExit(f"No metadata csv found under {root}. Found: {[c.name for c in csvs]}")
        meta = pd.concat(parts, ignore_index=True)
        print(f"[smid] metadata: concatenated {len(parts)} split files")

    # image-id column = the column whose values best match image file stems
    def match_rate(col):
        vals = meta[col].astype(str).map(lambda s: Path(s).stem)
        return vals.isin(images.keys()).mean()

    id_col = overrides.get("id") or max(meta.columns, key=match_rate)
    if match_rate(id_col) < 0.5:
        raise SystemExit(f"Could not match metadata rows to image files. Columns: {list(meta.columns)}; "
                         f"sample image stems: {list(images)[:5]}. Use --col-id.")

    cols = {}
    for r in RATINGS:
        cols[r] = overrides.get(r) or _find_rating_col(meta.columns, r)
        if cols[r] is None:
            raise SystemExit(f"No column for '{r}'. Columns: {list(meta.columns)}. Use --col-{r}.")
    print(f"[smid] id column: {id_col}; rating columns: {cols}")

    meta = meta.assign(_stem=meta[id_col].astype(str).map(lambda s: Path(s).stem))
    meta = meta[meta._stem.isin(images.keys())].drop_duplicates("_stem").reset_index(drop=True)
    labels = pd.DataFrame({"image": meta._stem, **{r: meta[cols[r]].astype(float) for r in RATINGS}})
    labels = labels.dropna().reset_index(drop=True)
    paths = [images[s] for s in labels.image]
    print(f"[smid] {len(paths)} images with all ratings")
    return paths, labels


# ------------------------------------------------------------------------- semantic
def load_imagefolder(root, max_per_class, seed=0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    paths, labels, classes = [], [], sorted(d.name for d in root.iterdir() if d.is_dir())
    for ci, c in enumerate(classes):
        files = sorted(p for p in (root / c).rglob("*") if p.suffix.lower() in IMG_EXT)
        if len(files) > max_per_class:
            files = [files[i] for i in sorted(rng.choice(len(files), max_per_class, replace=False))]
        paths += files
        labels += [ci] * len(files)
    print(f"[semantic] {len(paths)} images, {len(classes)} classes")
    return paths, np.array(labels), classes


# ------------------------------------------------------------------------ features
@torch.no_grad()
def extract(model_name, paths, batch_size, device):
    proc = AutoImageProcessor.from_pretrained(model_name)  # resize 256 + center-crop 224
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name).to(device=device, dtype=dtype).eval()
    n_reg = getattr(model.config, "num_register_tokens", 0) or 0
    feats = None
    for i in tqdm(range(0, len(paths), batch_size), desc=model_name):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(device, dtype)
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states[1:]  # skip embeddings
        layer_feats = []
        for h in hs:
            cls, patches = h[:, 0], h[:, 1 + n_reg:]
            layer_feats.append(torch.cat([cls, patches.mean(1)], dim=-1).float().cpu().numpy())
        batch = np.stack(layer_feats).astype(np.float16)  # [L, B, 2D]
        if feats is None:
            feats = np.zeros((batch.shape[0], len(paths), batch.shape[2]), np.float16)
        feats[:, i:i + len(ims)] = batch
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smid-dir", required=True)
    ap.add_argument("--semantic-dir", required=True, help="ImageFolder root: <root>/<class>/<image>")
    ap.add_argument("--semantic-max-per-class", type=int, default=300)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--out", default="features")
    ap.add_argument("--batch-size", type=int, default=32)
    for r in ["id"] + RATINGS:
        ap.add_argument(f"--col-{r}", default=None, help=f"override SMID column for {r}")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = {r: getattr(a, f"col_{r}") for r in ["id"] + RATINGS if getattr(a, f"col_{r}")}

    smid_paths, smid_labels = load_smid(a.smid_dir, overrides)
    smid_labels.to_csv(out / "smid_labels.csv", index=False)
    sem_paths, sem_labels, classes = load_imagefolder(a.semantic_dir, a.semantic_max_per_class)
    np.save(out / "semantic_labels.npy", sem_labels)
    json.dump(classes, open(out / "semantic_classes.json", "w"))

    for m in a.models:
        for tag, paths in [("smid", smid_paths), ("semantic", sem_paths)]:
            f = out / f"{m}_{tag}.npy"
            if f.exists():
                print(f"skip {f} (exists)"); continue
            np.save(f, extract(MODELS[m], paths, a.batch_size, device))
            print(f"saved {f}")


if __name__ == "__main__":
    main()
