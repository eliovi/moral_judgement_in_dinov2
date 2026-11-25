"""Render the writeup figures.

  python src/make_figures.py                       # main results figure from results/probe_results.csv
  python src/make_figures.py --thumbs DIR --gallery results/gallery.json   # also the SMID gallery figure (needs thumbnails)
"""
import argparse, json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888', 'axes.labelcolor': '#333', 'xtick.color': '#555', 'ytick.color': '#555'})
TASK_C = {'valence': '#d95f02', 'morality': '#7570b3', 'arousal': '#e6ab02', 'semantic': '#1b9e77'}
MODEL_C = {'S': '#c6dbef', 'B': '#6baed6', 'L': '#2171b5', 'G': '#08306b'}
PARAMS = {'S': 22e6, 'B': 86e6, 'L': 304e6, 'G': 1.1e9}
ap = argparse.ArgumentParser()
ap.add_argument('--results', default='results/probe_results.csv')
ap.add_argument('--out', default='figures')
ap.add_argument('--thumbs', default=None, help='folder of <scale>_<i>.jpg thumbnails (not in the repo: SMID images are not redistributed here)')
ap.add_argument('--gallery', default=None, help='gallery JSON with rows/items as written by the Colab gallery step')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
r = pd.read_csv(args.results)
r = r[~r.task.str.match(r'semantic:n')].copy()
r['task'] = r.task.str.replace('smid:', '', regex=False).str.replace('semantic (mean)', 'semantic', regex=False)

# ---------- Figure 1: main results (3 panels)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={'width_ratios': [1.25, 1, 1]})
a = r[(r.metric == 'auroc') & (r.model == 'G')]
for t, c in TASK_C.items():
    s = a[a.task == t].sort_values('depth')
    ax[0].plot(s.depth, s.value, '-o', ms=3, color=c, lw=2, label=t)
ax[0].set(xlabel='relative depth (layer / 40)', ylabel='AUROC (out-of-fold)', title='A. Layer by layer, DINOv2-G', ylim=(0.55, 1.01))
ax[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax[0].legend(frameon=False, loc='lower right')
best = (r[r.metric == 'auroc'].sort_values('value').groupby(['model', 'task']).tail(1))
for t, c in TASK_C.items():
    s = best[best.task == t].assign(p=lambda d: d.model.map(PARAMS)).sort_values('p')
    ax[1].errorbar(s.p, s.value, yerr=[s.value - s.lo, s.hi - s.value], fmt='-o', color=c, ms=5, lw=2, capsize=3, label=t)
ax[1].set_xscale('log'); ax[1].set_xticks(list(PARAMS.values())); ax[1].set_xticklabels(['S\n22M', 'B\n86M', 'L\n304M', 'G\n1.1B']); ax[1].minorticks_off()
ax[1].set(ylabel='best-layer AUROC (95% CI)', title='B. Top vs bottom third, by model size', ylim=(0.82, 1.01))
sp = (r[r.metric == 'spearman'].sort_values('value').groupby(['model', 'task']).tail(1))
for t in ['valence', 'morality', 'arousal']:
    s = sp[sp.task == t].assign(p=lambda d: d.model.map(PARAMS)).sort_values('p')
    ax[2].plot(s.p, s.value, '-o', color=TASK_C[t], ms=5, lw=2, label=t)
    ax[2].annotate(f'{s.value.iloc[-1]:.2f}', (s.p.iloc[-1], s.value.iloc[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=TASK_C[t])
ax[2].set_xscale('log'); ax[2].set_xticks(list(PARAMS.values())); ax[2].set_xticklabels(['S\n22M', 'B\n86M', 'L\n304M', 'G\n1.1B']); ax[2].minorticks_off()
ax[2].set(ylabel='best-layer Spearman ρ', title='C. Continuous human ratings', xlim=(15e6, 2.2e9))
for x in ax: x.grid(axis='y', color='#eee')
fig.suptitle('Linear probes on frozen DINOv2 features decode valence and morality of SMID images', x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.text(0.01, -0.04, 'Probe: standardize → PCA(256) → ridge, 5-fold out-of-fold. Binary tasks: 600 balanced images, 200-sample bootstrap CIs. '
         'Continuous: 2,940 images. Semantic = mean of 10 Imagenette one-vs-rest tasks.', fontsize=8, color='#666')
fig.tight_layout(); fig.savefig(f'{args.out}/fig_main_results.png', dpi=200, bbox_inches='tight'); plt.close(fig)

# ---------- Figure 2: gallery subset (valence, morality, arousal)
if not (args.thumbs and args.gallery):
    print('main figure written; skip gallery (pass --thumbs and --gallery)'); raise SystemExit
g = json.load(open(args.gallery))
rows = [('valence', 'Valence', 'unpleasant', 'pleasant'), ('moral', 'Morality', 'immoral', 'moral'), ('arousal', 'Arousal', 'calm', 'intense')]
pct = ['3rd', '25th', '50th', '75th', '97th']
fig, ax = plt.subplots(3, 5, figsize=(13, 6.9))
for ri, (key, label, lo, hi) in enumerate(rows):
    items = next(x for x in g['rows'] if x['scale'] == key)['items']
    for ci, it in enumerate(items):
        im = Image.open(f'{args.thumbs}/{key}_{ci}.jpg'); w, h = im.size
        tw, th = (w, int(w * 3 / 4)) if h >= w * 3 / 4 else (int(h * 4 / 3), h)
        im = im.crop(((w - tw) // 2, (h - th) // 2, (w - tw) // 2 + tw, (h - th) // 2 + th))
        a_ = ax[ri, ci]; a_.imshow(im); a_.set_xticks([]); a_.set_yticks([])
        for s_ in a_.spines.values(): s_.set_visible(False)
        a_.set_xlabel(f'{it[key]:.2f}  ·  {pct[ci]} pct', fontsize=9, color='#444')
    ax[ri, 0].set_ylabel(f'{label}\n{lo} → {hi}', fontsize=11, rotation=0, ha='right', va='center', labelpad=12)
fig.suptitle('SMID images across the human rating scale (mean rating 1–5, percentile within SMID)', x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.text(0.01, 0.005, 'Images: Socio-Moral Image Database (Crone, Bode, Murawski & Laham, 2018, PLOS ONE). Individual images are Creative Commons–licensed; see SMID for per-image credits.',
         fontsize=8, color='#666')
fig.subplots_adjust(left=0.13, right=0.995, top=0.93, bottom=0.06, wspace=0.04, hspace=0.18); fig.savefig(f'{args.out}/fig_gallery_valence_morality_arousal.png', dpi=140, bbox_inches='tight'); plt.close(fig)
print('ok')
