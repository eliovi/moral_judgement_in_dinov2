"""Test 6: do a vision-only model (DINOv2) and a caption-trained model (CLIP) share the good/bad direction?

  transfer_by_size_depth  For each DINOv2 size and 5 depths, cross-fitted over SMID: a DINOv2 -> CLIP map fit on the
                          training images WITHOUT ratings, and a rating probe trained on CLIP. CLIP's probe is applied to
                          the translated held-out DINOv2 features. Also: the rating from DINOv2's own top 10 principal
                          components only (how prominent the direction is in DINOv2's unsupervised structure).
  two_way_coco            Maps fit on unrated COCO photos only (DINOv2-g layer 37): CLIP's probe on DINOv2 translated
                          into CLIP, and DINOv2's probe on CLIP translated into DINOv2.
  mutual_knn              Overall similarity of DINOv2 to CLIP (mutual 10-nearest-neighbour overlap, Huh et al. 2024),
                          by size and depth, with the supervised ViT as a reference.

  python src/e6_shared.py --cache CACHE --out results
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import (BRIGHTNESS, EDGE_DENSITY, MODELS, Cache, cli, fit_map, oof, parse, probe, rho, rho_ci,
                    save_json, targets, unit)

DEPTHS = [0.25, 0.5, 0.75, 0.9, 1.0]


def clip_direction_in_dinov2(X, C, T, folds=5):
    """CLIP's rating probe applied to held-out DINOv2 features translated into CLIP space (map fit without ratings)."""
    pred = {t: np.zeros(len(X)) for t in T}
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(C):
        translated = unit(fit_map(X[tr], C[tr]).predict(X[te]))
        for t, y in T.items():
            pred[t][te] = fit_map(C[tr], y[tr]).predict(translated)
    return {t: rho(y, pred[t]) for t, y in T.items()}


def mutual_knn(A, B, k=10):
    def neighbours(X):
        X = unit(X); S = X @ X.T; np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]
    na, nb = neighbours(A), neighbours(B)
    return round(float(np.mean([len(set(na[i]) & set(nb[i])) / k for i in range(len(A))])), 4)


def main():
    a = parse(cli(__doc__))
    cache = Cache(a.cache)
    T = targets(cache.labels())
    C = cache.clip()
    ll, sup = cache.extra_features("lowlevel"), cache.extra_features("sup_feats")
    res = {"clip_own_probe": {t: rho(y, oof(C, y)) for t, y in T.items()}, "transfer_by_size_depth": [], "mutual_knn": [],
           "mutual_knn_supervised_vit_vs_clip": mutual_knn(sup, C)}

    print("transfer and prominence by size and depth", flush=True)
    for m in MODELS:
        F = cache.dinov2(m, layer=None)
        n_layers = F.shape[0]
        for d in DEPTHS:
            l = max(0, int(round(d * n_layers)) - 1)
            X = np.asarray(F[l], dtype=np.float32)
            top10 = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(X))
            row = {"model": m, "layer": l + 1, "depth": round((l + 1) / n_layers, 3)}
            row |= {f"clip_direction_{t}": r for t, r in clip_direction_in_dinov2(X, C, T).items()}
            row |= {f"top10pc_{t}": rho(y, oof(top10, y, k=10)) for t, y in T.items()}
            res["transfer_by_size_depth"].append(row)
            print(" ", row, flush=True)
        for l in sorted(set(range(0, n_layers, max(1, n_layers // 8))) | {n_layers - 1}):
            X = np.asarray(F[l], dtype=np.float32)
            res["mutual_knn"].append({"model": m, "layer": l + 1, "depth": round((l + 1) / n_layers, 3),
                                      "vs_clip": mutual_knn(X, C), "vs_supervised_vit": mutual_knn(X, sup)})
        save_json(res, f"{a.out}/e6_shared.json")

    print("prominence comparators (DINOv2-g layer 37)", flush=True)
    top10 = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(cache.dinov2()))
    res["top10pc_comparators_G37"] = {name: rho(y, oof(top10, y, k=10)) for name, y in
                                      {"brightness": ll[:, BRIGHTNESS], "edge_density": ll[:, EDGE_DENSITY]}.items()}

    print("two-way transfer through maps fit on COCO", flush=True)
    dino = cache.dinov2()
    Dc, Cc = cache.coco_pair()
    dino_in_clip = unit(fit_map(Dc, Cc, k=512).predict(dino))
    clip_in_dino = fit_map(Cc, Dc.astype(np.float64), k=512).predict(C).astype(np.float32)
    res["two_way_coco"] = {}
    for t, y in T.items():
        p_clip, p_dino = np.zeros_like(y), np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(C):
            p_clip[te] = probe(min(256, C.shape[1])).fit(C[tr], y[tr]).predict(dino_in_clip[te])
            p_dino[te] = probe(min(256, dino.shape[1])).fit(dino[tr], y[tr]).predict(clip_in_dino[te])
        res["two_way_coco"][t] = {"clip_probe_on_translated_dinov2": rho_ci(y, p_clip),
                                  "dinov2_probe_on_translated_clip": rho_ci(y, p_dino),
                                  "dinov2_own_probe": rho_ci(y, oof(dino, y))}
        print(" ", t, {k: v["rho"] for k, v in res["two_way_coco"][t].items()}, flush=True)
    save_json(res, f"{a.out}/e6_shared.json")


if __name__ == "__main__":
    main()
