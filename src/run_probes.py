"""
Linear probes on cached DINOv2 features: how decodable is affective valence
(plus arousal, morality) compared with ordinary semantic categories, per layer and per model size?

Tasks
  Binary (all subsampled to the same N, balanced, metric = AUROC):
    smid:valence / arousal / morality  -> top third vs bottom third of the human rating
    semantic:<class>                   -> one class vs the rest
  Continuous (full SMID, metric = Spearman rho and R^2 of out-of-fold predictions):
    valence / arousal / morality

Probe per fold: StandardScaler -> PCA(k, fit on train) -> RidgeCV (alpha by efficient LOO).

Usage:
  python run_probes.py --features features --out results
  python run_probes.py --synthetic --out results_synthetic     # pipeline smoke test, no data needed
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARAMS = {"S": 22e6, "B": 86e6, "L": 304e6, "G": 1.1e9}
RATINGS = ["valence", "arousal", "morality"]
ALPHAS = np.logspace(-1, 5, 13)


def probe(k):
    return make_pipeline(StandardScaler(), PCA(k, random_state=0), RidgeCV(alphas=ALPHAS))


def oof_predict(X, y, folds, k, stratify, seed):
    k = min(k, X.shape[1], int(len(y) * (folds - 1) / folds) - 1)
    cv = (StratifiedKFold if stratify else KFold)(folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in cv.split(X, y if stratify else None):
        pred[te] = probe(k).fit(X[tr], y[tr]).predict(X[te])
    return pred


def boot_auc(y, s, n, rng):
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def build_binary_tasks(smid, sem_y, classes, n_binary, rng):
    """Return {task_name: (dataset, indices, binary_labels)}, every task balanced with n_binary samples."""
    tasks = {}
    half = n_binary // 2
    for r in RATINGS:
        v = smid[r].to_numpy()
        lo, hi = np.where(v <= np.quantile(v, 1 / 3))[0], np.where(v >= np.quantile(v, 2 / 3))[0]
        h = min(half, len(lo), len(hi))
        idx = np.r_[rng.choice(lo, h, replace=False), rng.choice(hi, h, replace=False)]
        tasks[f"smid:{r}"] = ("smid", idx, np.r_[np.zeros(h), np.ones(h)])
    for ci, c in enumerate(classes):
        pos, neg = np.where(sem_y == ci)[0], np.where(sem_y != ci)[0]
        h = min(half, len(pos), len(neg))
        idx = np.r_[rng.choice(neg, h, replace=False), rng.choice(pos, h, replace=False)]
        tasks[f"semantic:{c}"] = ("semantic", idx, np.r_[np.zeros(h), np.ones(h)])
    sizes = {t: len(v[1]) for t, v in tasks.items()}
    if len(set(sizes.values())) > 1:
        print(f"[warn] unequal binary task sizes (lower --n-binary to equalise): {sizes}")
    return tasks


def run(feats, smid, sem_y, classes, models, a):
    rng = np.random.default_rng(a.seed)
    tasks = build_binary_tasks(smid, sem_y, classes, a.n_binary, rng)
    rows = []
    for m in models:
        F = {"smid": feats[m]["smid"], "semantic": feats[m]["semantic"]}
        n_layers = F["smid"].shape[0]
        layers = sorted(set(range(0, n_layers, a.layer_stride)) | {n_layers - 1})
        for L in layers:
            depth = (L + 1) / n_layers
            for t, (ds, idx, yb) in tasks.items():
                X = F[ds][L, idx].astype(np.float32)
                s = oof_predict(X, yb, a.folds, a.pca, True, a.seed)
                lo, hi = boot_auc(yb, s, a.boot, rng)
                rows.append(dict(model=m, params=PARAMS[m], layer=L, depth=depth, task=t,
                                 kind="binary", metric="auroc", value=roc_auc_score(yb, s), lo=lo, hi=hi))
            X = F["smid"][L].astype(np.float32)
            for r in RATINGS:
                y = smid[r].to_numpy()
                p = oof_predict(X, y, a.folds, a.pca, False, a.seed)
                rows.append(dict(model=m, params=PARAMS[m], layer=L, depth=depth, task=f"smid:{r}",
                                 kind="continuous", metric="spearman", value=spearmanr(y, p)[0]))
                rows.append(dict(model=m, params=PARAMS[m], layer=L, depth=depth, task=f"smid:{r}",
                                 kind="continuous", metric="r2", value=r2_score(y, p)))
            print(f"{m} layer {L + 1}/{n_layers} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- plots
COLORS = {"S": "#9ecae1", "B": "#4292c6", "L": "#2171b5", "G": "#08306b"}
TASK_COLORS = {"smid:valence": "#d94801", "smid:arousal": "#fd8d3c",
               "smid:morality": "#7a0177", "semantic (mean)": "#238b45"}


def add_semantic_mean(df):
    sem = df[df.task.str.startswith("semantic:")]
    agg = sem.groupby(["model", "params", "layer", "depth", "kind", "metric"], as_index=False)[
        ["value", "lo", "hi"]].mean()
    return pd.concat([df, agg.assign(task="semantic (mean)")], ignore_index=True)


def plot_layers(df, out):
    d = df[(df.metric == "auroc") & df.task.isin(TASK_COLORS)]
    tasks = list(TASK_COLORS)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.4), sharey=True)
    for ax, t in zip(axes, tasks):
        for m in d.model.unique():
            s = d[(d.task == t) & (d.model == m)].sort_values("depth")
            ax.plot(s.depth, s.value, "-o", ms=3, color=COLORS.get(m), label=f"DINOv2-{m}")
        ax.set_title(t); ax.set_xlabel("relative depth"); ax.grid(alpha=.3)
    axes[0].set_ylabel("AUROC (out-of-fold)"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig1_auroc_by_layer.png", dpi=150); plt.close(fig)


def best_layer(df, metric):
    d = df[df.metric == metric]
    return d.loc[d.groupby(["model", "task"]).value.idxmax()]


def plot_scaling(df, out):
    b = best_layer(df, "auroc")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for t, c in TASK_COLORS.items():
        s = b[b.task == t].sort_values("params")
        if s.empty:
            continue
        ax.errorbar(s.params, s.value, yerr=[s.value - s.lo, s.hi - s.value], fmt="-o", color=c,
                    capsize=3, label=t)
    ax.set_xscale("log"); ax.set_xlabel("parameters"); ax.set_ylabel("best-layer AUROC")
    ax.set_title("Decodability vs DINOv2 size"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig2_scaling_auroc.png", dpi=150); plt.close(fig)

    b = best_layer(df, "spearman")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for t in [f"smid:{r}" for r in RATINGS]:
        s = b[b.task == t].sort_values("params")
        ax.plot(s.params, s.value, "-o", color=TASK_COLORS[t], label=t)
    ax.set_xscale("log"); ax.set_xlabel("parameters"); ax.set_ylabel("best-layer Spearman ρ")
    ax.set_title("Continuous ratings (full SMID)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig3_scaling_continuous.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------- synthetic
def synthetic(models, n_smid=900, n_sem=1000, n_cls=5, seed=0):
    """Fake features with planted signals: semantic easy & early, valence weaker & late."""
    rng = np.random.default_rng(seed)
    dims = {"S": 384, "B": 768, "L": 1024, "G": 1536}
    nl = {"S": 12, "B": 12, "L": 24, "G": 40}
    z = rng.normal(size=(n_smid, 3))  # latent valence, arousal, morality
    z[:, 2] = 0.7 * z[:, 0] + 0.7 * z[:, 2]
    smid = pd.DataFrame({"image": [f"img{i}" for i in range(n_smid)],
                         **{r: 3 + z[:, i] + 0.5 * rng.normal(size=n_smid) for i, r in enumerate(RATINGS)}})
    sem_y = rng.integers(0, n_cls, n_sem)
    feats = {}
    for m in models:
        D, L = 2 * dims[m], nl[m]
        size_gain = np.log10(PARAMS[m]) - 7
        W = rng.normal(size=(3, D)); C = rng.normal(size=(n_cls, D))
        fs, fm = [], []
        for l in range(L):
            d = (l + 1) / L
            fs.append(rng.normal(size=(n_smid, D)) + (0.04 * size_gain * d ** 2) * z @ W)
            fm.append(rng.normal(size=(n_sem, D)) + (0.15 * min(1, 2 * d)) * C[sem_y])
        feats[m] = {"smid": np.stack(fs).astype(np.float16), "semantic": np.stack(fm).astype(np.float16)}
    return feats, smid, sem_y, [f"class{i}" for i in range(n_cls)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=["S", "B", "L", "G"])
    ap.add_argument("--n-binary", type=int, default=600, help="samples per binary task (balanced)")
    ap.add_argument("--pca", type=int, default=256, help="PCA dims (same for all sizes; also speeds up)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        feats, smid, sem_y, classes = synthetic(a.models)
    else:
        fd = Path(a.features)
        smid = pd.read_csv(fd / "smid_labels.csv")
        sem_y = np.load(fd / "semantic_labels.npy")
        classes = json.load(open(fd / "semantic_classes.json"))
        models = [m for m in a.models if (fd / f"{m}_smid.npy").exists()]
        feats = {m: {t: np.load(fd / f"{m}_{t}.npy", mmap_mode="r") for t in ("smid", "semantic")}
                 for m in models}
        a.models = models

    df = add_semantic_mean(run(feats, smid, sem_y, classes, a.models, a))
    df.to_csv(out / "results.csv", index=False)
    plot_layers(df, out); plot_scaling(df, out)

    summary = best_layer(df, "auroc")
    summary = summary[summary.task.isin(TASK_COLORS)].pivot(index="task", columns="model", values="value")
    print("\nBest-layer AUROC:\n", summary[[m for m in a.models if m in summary.columns]].round(3))
    print(f"\nWrote {out}/results.csv and figures.")


if __name__ == "__main__":
    main()
