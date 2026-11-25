"""
GPU features for round 3.

  --random-smid : a randomly initialised DINOv2-G (same architecture, no training) on SMID, layers 13/25/37/40,
                  same CLS + mean-patch readout as extract_features.py. Tests whether architecture alone gives the signal.
  --places N    : DINOv2-G layer 37 and CLIP ViT-L/14 features for N random Places365 validation images, used with
                  the cached COCO features to fit a larger DINOv2 -> CLIP map.

  python extract_round3.py --labels features/smid_labels.csv --smid-dir data/smid --random-smid --places 20000 --out round3
"""
import argparse, os, random, tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel

PLACES_URL = "http://data.csail.mit.edu/places/places365/val_256.tar"
RANDOM_LAYERS = [12, 24, 36, 39]   # 0-based; 36 = the trained model's best valence layer


@torch.no_grad()
def random_init_smid(smid_dir, labels_csv, out, batch=32):
    from extract_features import load_smid
    paths, labels = load_smid(smid_dir, {})
    assert list(labels.image) == list(pd.read_csv(labels_csv).image), "image order differs from the DINOv2 features"
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained("facebook/dinov2-giant")
    model = AutoModel.from_config(cfg).to("cuda", torch.float16).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    feats = {l: [] for l in RANDOM_LAYERS}
    for i in tqdm(range(0, len(paths), batch), desc="random-init DINOv2-G"):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        hs = model(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"].to("cuda", torch.float16),
                   output_hidden_states=True).hidden_states
        for l in RANDOM_LAYERS:
            h = hs[l + 1]
            feats[l].append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float().cpu().numpy())
    np.save(out / "random_init_G_smid.npy", np.stack([np.concatenate(feats[l]) for l in RANDOM_LAYERS]).astype(np.float16))
    np.save(out / "random_init_layers.npy", np.array(RANDOM_LAYERS))
    del model; torch.cuda.empty_cache()


def places_features(n, out, workdir="places"):
    from coco_translation import coco_features
    img_root = Path(workdir) / "val_256"
    if not img_root.exists():
        os.makedirs(workdir, exist_ok=True)
        tar = Path(workdir) / "val_256.tar"
        assert os.system(f"wget -q -O {tar} {PLACES_URL}") == 0, "Places365 download failed"
        tarfile.open(tar).extractall(workdir); tar.unlink()
    files = sorted(img_root.glob("*.jpg"))
    random.Random(0).shuffle(files)
    sub = Path(workdir) / f"subset_{n}"; sub.mkdir(exist_ok=True)
    for f in files[:n]:
        link = sub / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    D, C = coco_features(sub, layer=36)
    np.save(out / "places_dinov2G_L37.npy", D.astype(np.float16)); np.save(out / "places_clip.npy", C.astype(np.float32))
    print("Places features", D.shape, C.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--smid-dir", default=None)
    ap.add_argument("--random-smid", action="store_true")
    ap.add_argument("--places", type=int, default=0)
    ap.add_argument("--out", default="round3")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.random_smid and not (out / "random_init_G_smid.npy").exists():
        random_init_smid(a.smid_dir, a.labels, out)
    if a.places and not (out / "places_clip.npy").exists():
        places_features(a.places, out)
    print("round-3 features in", out, sorted(p.name for p in out.glob("*.npy")))


if __name__ == "__main__":
    main()
