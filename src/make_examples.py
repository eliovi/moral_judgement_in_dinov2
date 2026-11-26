"""Candidate example images for the photo figures (needs the SMID images, so run it where they are downloaded).

Candidates are images where DINOv2's held-out probe and people agree:
  strip_*                 five bins along DINOv2's predicted morality, from the bottom 6% to the top 6%
  good_but_unpleasant     high predicted morality beyond valence, low valence rating
  pleasant_not_moral      low predicted morality beyond valence, high valence rating
Images most rated as purity-relevant (sexual or disgusting content) are left out. Thumbnails go to a JSON file.

The final picks in results/examples/picks.csv were chosen by hand from these candidates: no graphic images and no
identifiable public figures. figures.py reads the picks and the thumbnails in results/examples/thumbs/.

  python src/make_examples.py --cache CACHE --smid-dir data/smid --metadata metadata.csv --out results
"""
import base64, io, json
from pathlib import Path

import numpy as np
import pandas as pd

from common import Cache, cli, load_metadata, oof, parse, targets
from extract import load_smid


def thumbnail(path, size=256):
    from PIL import Image
    im = Image.open(path).convert("RGB"); im.thumbnail((size, size))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = cli(__doc__)
    ap.add_argument("--smid-dir", required=True)
    ap.add_argument("--metadata", required=True)
    a = parse(ap)
    cache = Cache(a.cache)
    lab = cache.labels(); meta = load_metadata(a.metadata, lab)
    paths, lab2 = load_smid(a.smid_dir)
    assert list(lab2.image) == list(lab.image)
    T = targets(lab)
    dino = cache.dinov2()
    pct = lambda x: pd.Series(x).rank(pct=True).to_numpy()
    pred_m, true_m = pct(oof(dino, T["morality"])), pct(T["morality"])
    pred_b, true_b = pct(oof(dino, T["morality_beyond_valence"])), pct(T["morality_beyond_valence"])
    val = T["valence"]
    ok = pct(meta.purity_mean.to_numpy(float)) < 0.9

    candidates = {}
    for lo, hi in [(0, .06), (.2, .3), (.45, .55), (.7, .8), (.94, 1)]:
        idx = np.flatnonzero(ok & (pred_m >= lo) & (pred_m <= hi) & (np.abs(pred_m - true_m) < .1))
        candidates[f"strip_{lo:.2f}-{hi:.2f}"] = idx[np.argsort(pred_m[idx])][:8]
    idx = np.flatnonzero(ok & (pred_b > .85) & (true_b > .75) & (val < 2.6))
    candidates["good_but_unpleasant"] = idx[np.argsort(-pred_b[idx])][:12]
    idx = np.flatnonzero(ok & (pred_b < .15) & (true_b < .25) & (val > 3.4))
    candidates["pleasant_not_moral"] = idx[np.argsort(pred_b[idx])][:12]

    out = {k: [{"image": lab.image[i], "valence_rating": round(float(val[i]), 2), "morality_rating": round(float(T["morality"][i]), 2),
                "dinov2_morality_percentile": round(float(pred_m[i]), 3),
                "dinov2_morality_beyond_valence_percentile": round(float(pred_b[i]), 3), "jpg": thumbnail(paths[i])}
               for i in ids] for k, ids in candidates.items()}
    path = Path(a.out) / "examples" / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"))
    print("saved", path, {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
