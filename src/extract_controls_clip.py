"""
Extra features for the "beyond objects" control (a2) and the CLIP-space analysis (b3).

For every SMID image, in exactly the same order as the DINOv2 features (smid_labels.csv):
  lowlevel.npy   [N, 57]   colour / brightness / contrast / edge statistics (no learning)
  sup_logits.npy [N, 1000] ImageNet class logits of a supervised ViT-B/16 (google/vit-base-patch16-224)
  sup_feats.npy  [N, 768]  that ViT's final CLS feature ("everything ImageNet supervision learned")
  clip_img.npy   [N, 768]  CLIP ViT-L/14 image embeddings (L2-normalised)
Plus CLIP text embeddings for the prompt pairs and the word list in text_sets.json:
  clip_text.npy  [T, 768]  and text_index.json (the texts, in row order)

Usage (Colab, GPU):
  python extract_controls_clip.py --smid-dir data/smid --labels features/smid_labels.csv --out features_extra
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPModel, CLIPProcessor

from extract_features import load_smid

PROMPT_TEMPLATE = "a photo of {}"
PAIRS = {
    "affect": [("something pleasant", "something unpleasant"), ("a happy scene", "a sad scene"),
               ("a beautiful scene", "an ugly scene"), ("a positive situation", "a negative situation"),
               ("people enjoying themselves", "people suffering")],
    "moral": [("a morally good act", "a morally bad act"), ("someone being kind", "someone being cruel"),
              ("an ethical action", "an unethical action"), ("people helping each other", "people harming each other"),
              ("a fair situation", "an unfair situation"), ("an honest person", "a dishonest person")],
    "arousal": [("a calm scene", "an intense scene"), ("a peaceful moment", "a chaotic moment")],
}
WORDS = (
    # affect
    "joy happiness love peace comfort beauty celebration fun laughter friendship sadness grief fear anger pain "
    "suffering despair loneliness disgust horror misery anxiety "
    # moral
    "kindness generosity charity compassion honesty justice fairness loyalty respect care cruelty violence abuse "
    "cheating theft lying corruption injustice betrayal exploitation crime murder war terrorism poverty racism "
    "discrimination pollution neglect vandalism"
).split() + [
    # scenes
    "a city street", "a kitchen", "an office", "a forest", "a beach", "a mountain", "a river", "a highway",
    "a parking lot", "a classroom", "a hospital", "a church", "a farm", "a factory", "a market", "a stadium",
    "a prison", "a cemetery", "a garden", "a playground", "a wedding", "a funeral", "a protest", "a party",
    "a concert", "a parade", "a battlefield", "a slum", "a mansion", "a desert",
    # objects and people
    "a car", "a bicycle", "a dog", "a cat", "a horse", "a bird", "a flower", "a tree", "food", "a cake", "a gun",
    "a knife", "money", "a phone", "a book", "a computer", "a chair", "a building", "a bridge", "a train",
    "an airplane", "a boat", "trash", "smoke", "fire", "blood", "a cigarette", "alcohol", "drugs", "a toy",
    "a baby", "a child", "an elderly person", "a soldier", "a police officer", "a doctor", "a nurse", "a teacher",
    "an athlete", "a crowd", "a family", "a couple",
    # visual properties
    "the color red", "the color blue", "the color green", "the color yellow", "darkness", "bright light",
    "vivid colors", "gray tones", "a sunny day", "rain", "night", "an old photograph", "a black and white photo",
    "a blurry image", "a close-up", "an aerial view",
]


def lowlevel(im):
    im = im.resize((224, 224))
    a = np.asarray(im, dtype=np.float32) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(lum)
    edge = np.hypot(gx, gy)
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    quad = [lum[i * 112:(i + 1) * 112, j * 112:(j + 1) * 112].mean() for i in range(2) for j in range(2)]
    n = lum.size
    return np.array([*a.mean((0, 1)), *a.std((0, 1)), *hsv.mean((0, 1)), *hsv.std((0, 1)), lum.mean(), lum.std(),
                     edge.mean(), edge.std(), colorful, *quad,
                     *np.histogram(lum, 16, (0, 1))[0] / n, *np.histogram(hsv[..., 1], 8, (0, 1))[0] / n,
                     *np.histogram(hsv[..., 0], 12, (0, 1))[0] / n], dtype=np.float32)


def as_emb(o):
    """CLIP get_*_features returns a tensor in older transformers and an output object in newer ones."""
    if torch.is_tensor(o):
        return o
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(o, k, None)
        if v is not None:
            return v
    return o[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smid-dir", required=True)
    ap.add_argument("--labels", required=True, help="smid_labels.csv written with the DINOv2 features")
    ap.add_argument("--out", default="features_extra")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32

    paths, labels = load_smid(a.smid_dir, {})
    ref = pd.read_csv(a.labels)
    assert list(labels.image) == list(ref.image), "image order differs from the DINOv2 features"
    print(f"{len(paths)} images, order matches DINOv2 features")

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev, dt).eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    sup = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(dev, dt).eval()
    sproc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    ll, lg, sf, ci = [], [], [], []
    for i in tqdm(range(0, len(paths), a.batch_size)):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + a.batch_size]]
        ll += [lowlevel(im) for im in ims]
        o = sup(pixel_values=sproc(images=ims, return_tensors="pt")["pixel_values"].to(dev, dt), output_hidden_states=True)
        lg.append(o.logits.float().cpu().numpy()); sf.append(o.hidden_states[-1][:, 0].float().cpu().numpy())
        e = as_emb(clip.get_image_features(pixel_values=cproc(images=ims, return_tensors="pt")["pixel_values"].to(dev, dt))).float()
        ci.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    np.save(out / "lowlevel.npy", np.stack(ll)); np.save(out / "sup_logits.npy", np.concatenate(lg))
    np.save(out / "sup_feats.npy", np.concatenate(sf)); np.save(out / "clip_img.npy", np.concatenate(ci))

    texts = sorted({PROMPT_TEMPLATE.format(t) for grp in PAIRS.values() for pr in grp for t in pr} |
                   {PROMPT_TEMPLATE.format(w) for w in WORDS})
    tok = cproc(text=texts, padding=True, return_tensors="pt").to(dev)
    te = torch.nn.functional.normalize(as_emb(clip.get_text_features(**tok)).float(), dim=-1).cpu().numpy()
    assert te.shape[1] == 768 and np.concatenate(ci).shape[1] == 768
    np.save(out / "clip_text.npy", te)
    json.dump({"texts": texts, "template": PROMPT_TEMPLATE, "pairs": PAIRS, "words": WORDS}, open(out / "text_index.json", "w"))
    print("saved:", {f.name: np.load(f, mmap_mode="r").shape for f in sorted(out.glob("*.npy"))})


if __name__ == "__main__":
    main()
