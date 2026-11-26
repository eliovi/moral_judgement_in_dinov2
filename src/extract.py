"""Feature extraction (needs a GPU). Writes the cache that every experiment reads (layout in common.py).

  python src/extract.py smid     --smid-dir data/smid --cache CACHE   # DINOv2 S/B/L/G, every layer, + smid_labels.csv
  python src/extract.py controls --smid-dir data/smid --cache CACHE   # low-level stats, supervised ViT, CLIP images
  python src/extract.py coco     --coco-dir data/coco --cache CACHE   # DINOv2-g layer 37 + CLIP for COCO val2017

SMID comes from Hugging Face (AIML-TUDA/smid, gated: accept the terms first). COCO val2017 is downloaded if missing.
"""
import argparse, os, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import BEST_LAYER, MODELS, Cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CLIP_NAME, SUPERVISED_NAME = "openai/clip-vit-large-patch14", "google/vit-base-patch16-224"
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"


# ------------------------------------------------------------------------------------------------------- images
def load_smid(root):
    """Image paths and mean ratings, in the row order of SMID's metadata.csv (only images with all three ratings)."""
    root = Path(root)
    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    meta = pd.read_csv(next(root.rglob("metadata.csv")))
    meta = meta.assign(image=meta.file_name.map(lambda s: Path(str(s)).stem))
    meta = meta[meta.image.isin(images)].drop_duplicates("image")
    labels = pd.DataFrame({"image": meta.image, "valence": meta.valence_mean,
                           "arousal": meta.arousal_mean, "morality": meta.moral_mean}).dropna().reset_index(drop=True)
    print(f"SMID: {len(labels)} images with all ratings")
    return [images[s] for s in labels.image], labels


def batches(paths, size):
    for i in tqdm(range(0, len(paths), size)):
        yield i, [Image.open(p).convert("RGB") for p in paths[i:i + size]]


def clip_embed(model, processor, ims):
    out = model.get_image_features(pixel_values=processor(images=ims, return_tensors="pt")["pixel_values"].to(DEVICE, DTYPE))
    if not torch.is_tensor(out):  # newer transformers return an output object
        emb = getattr(out, "image_embeds", None)
        out = emb if emb is not None else out.pooler_output
    return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()


# ------------------------------------------------------------------------------------------------------- DINOv2
@torch.no_grad()
def dinov2_features(name, paths, layers=None, batch_size=32):
    """Per layer: CLS token concatenated with the mean patch token. Returns [n_layers, n_images, 2 x width]."""
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(name)               # resize 256, centre-crop 224
    model = AutoModel.from_pretrained(name).to(DEVICE, DTYPE).eval()
    feats = None
    for i, ims in batches(paths, batch_size):
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEVICE, DTYPE)
        hidden = model(pixel_values=x, output_hidden_states=True).hidden_states[1:]   # drop the patch embedding
        hidden = [hidden[l] for l in layers] if layers else hidden
        batch = np.stack([torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float().cpu().numpy() for h in hidden])
        if feats is None:
            feats = np.zeros((batch.shape[0], len(paths), batch.shape[2]), np.float16)
        feats[:, i:i + len(ims)] = batch
    del model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return feats


def cmd_smid(a, cache):
    paths, labels = load_smid(a.smid_dir)
    cache.features.mkdir(parents=True, exist_ok=True)
    labels.to_csv(cache.features / "smid_labels.csv", index=False)
    for m, name in MODELS.items():
        f = cache.features / f"{m}_smid.npy"
        if f.exists():
            print("exists:", f); continue
        np.save(f, dinov2_features(name, paths))
        print("saved", f)


# ------------------------------------------------------------------------------------------------------- controls
def lowlevel(im):
    """57 statistics: RGB and HSV means/SDs, luminance mean/SD, edge mean/SD, colourfulness,
    quadrant luminance, and luminance (16), saturation (8) and hue (12) histograms."""
    im = im.resize((224, 224))
    a = np.asarray(im, dtype=np.float32) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(lum)
    edge = np.hypot(gx, gy)
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    quad = [lum[i * 112:(i + 1) * 112, j * 112:(j + 1) * 112].mean() for i in range(2) for j in range(2)]
    hist = lambda x, bins: np.histogram(x, bins, (0, 1))[0] / lum.size
    return np.array([*a.mean((0, 1)), *a.std((0, 1)), *hsv.mean((0, 1)), *hsv.std((0, 1)), lum.mean(), lum.std(),
                     edge.mean(), edge.std(), colorful, *quad,
                     *hist(lum, 16), *hist(hsv[..., 1], 8), *hist(hsv[..., 0], 12)], dtype=np.float32)


@torch.no_grad()
def cmd_controls(a, cache):
    from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPModel, CLIPProcessor
    paths, labels = load_smid(a.smid_dir)
    assert list(labels.image) == list(cache.labels().image), "image order differs from the DINOv2 features"
    clip, cproc = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE, DTYPE).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)
    sup = AutoModelForImageClassification.from_pretrained(SUPERVISED_NAME).to(DEVICE, DTYPE).eval()
    sproc = AutoImageProcessor.from_pretrained(SUPERVISED_NAME)
    out = {"lowlevel": [], "sup_logits": [], "sup_feats": [], "clip_img": []}
    for _, ims in batches(paths, 64):
        out["lowlevel"] += [lowlevel(im) for im in ims]
        o = sup(pixel_values=sproc(images=ims, return_tensors="pt")["pixel_values"].to(DEVICE, DTYPE), output_hidden_states=True)
        out["sup_logits"].append(o.logits.float().cpu().numpy())
        out["sup_feats"].append(o.hidden_states[-1][:, 0].float().cpu().numpy())
        out["clip_img"].append(clip_embed(clip, cproc, ims))
    cache.extra.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        np.save(cache.extra / f"{k}.npy", np.stack(v) if k == "lowlevel" else np.concatenate(v))
        print("saved", cache.extra / f"{k}.npy")


# ------------------------------------------------------------------------------------------------------- COCO
@torch.no_grad()
def cmd_coco(a, cache):
    from transformers import CLIPModel, CLIPProcessor
    img_dir = Path(a.coco_dir) / "val2017"
    if not img_dir.exists():
        os.makedirs(a.coco_dir, exist_ok=True)
        z = Path(a.coco_dir) / "val2017.zip"
        assert os.system(f"wget -q -O {z} {COCO_URL}") == 0, "COCO download failed"
        zipfile.ZipFile(z).extractall(a.coco_dir); z.unlink()
    paths = sorted(img_dir.glob("*.jpg"))
    cache.coco.mkdir(parents=True, exist_ok=True)
    np.save(cache.coco / "coco_dinov2G_L37.npy", dinov2_features(MODELS["G"], paths, layers=[BEST_LAYER])[0])
    clip, cproc = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE, DTYPE).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)
    np.save(cache.coco / "coco_clip.npy", np.concatenate([clip_embed(clip, cproc, ims) for _, ims in batches(paths, 64)]))
    print("saved COCO features for", len(paths), "images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["smid", "controls", "coco"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--smid-dir", default="data/smid")
    ap.add_argument("--coco-dir", default="data/coco")
    a = ap.parse_args()
    {"smid": cmd_smid, "controls": cmd_controls, "coco": cmd_coco}[a.what](a, Cache(a.cache))
