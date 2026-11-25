"""Figure for steps 10-12: run from the repo root."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888', 'xtick.color': '#555', 'ytick.color': '#555'})
TC = {'valence': '#d95f02', 'morality': '#7570b3', 'morality_beyond_valence': '#a6761d', 'arousal': '#e6ab02'}
p = pd.read_csv('results/controls_and_language/probes_by_features.csv'); b = pd.read_csv('results/controls_and_language/beyond_controls.csv'); l = pd.read_csv('results/controls_and_language/language_axes.csv')
fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw={'width_ratios': [1.15, 1, 1.05]})

# A
feats = ['lowlevel', 'objects', 'supervised_vit', 'dinov2', 'clip']
lab = {'lowlevel': 'Low-level stats', 'objects': 'ImageNet object scores', 'supervised_vit': 'Supervised ViT features', 'dinov2': 'DINOv2-G (no labels)', 'clip': 'CLIP ViT-L/14'}
tg = [('valence', 'valence'), ('morality', 'morality'), ('morality_beyond_valence', 'morality beyond valence')]
y = np.arange(len(feats)); h = 0.26
for i, (t, name) in enumerate(tg):
    d = p[p.target == t].set_index('features').loc[feats]
    ax[0].barh(y + (1 - i) * h, d.rho, h, color=TC[t], label=name, xerr=[d.rho - d.lo, d.hi - d.rho], error_kw={'lw': 0.8, 'ecolor': '#444'})
ax[0].set_yticks(y); ax[0].set_yticklabels([lab[f] for f in feats]); ax[0].invert_yaxis(); ax[0].set_xlim(0, 1)
ax[0].set_xlabel('Spearman ρ, probe vs human rating (95% CI)'); ax[0].set_title('A. What each representation predicts', loc='left')
ax[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)

# B
ctrl = ['none', 'lowlevel', 'lowlevel+objects', 'lowlevel+objects+supervised']
clab = ['none', 'low-level', '+ object\nscores', '+ supervised\nViT features']
for t, ls in [('valence', '-'), ('morality', '--')]:
    for m, c in [('dinov2', '#08306b'), ('clip', '#1b9e77')]:
        vals = [p[(p.features == m) & (p.target == t)].rho.iloc[0]] + [b[(b.controls == k) & (b.target == t) & (b.model == m)].rho_on_residual.iloc[0] for k in ctrl[1:]]
        ax[1].plot(range(4), vals, ls, marker='o', color=c, lw=2, ms=5, label=f"{'DINOv2-G' if m == 'dinov2' else 'CLIP'}, {t}")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(clab); ax[1].set_ylim(0, 1)
ax[1].set_ylabel('ρ with the rating left after controls'); ax[1].set_title('B. Signal left after removing controls', loc='left')
ax[1].legend(frameon=False, fontsize=8.5, loc='lower left'); ax[1].grid(axis='y', color='#eee')

# C
spaces = ['clip', 'dinov2_map_fit_on_smid', 'dinov2_map_fit_on_coco']
slab = ['CLIP', 'DINOv2→CLIP\nmap fit\non SMID', 'DINOv2→CLIP\nmap fit\non COCO']
x = np.arange(3); w = 0.2
bars = [('affect', 'valence_given_morality', 'affect prompts: valence | morality', '#d95f02', 1.0),
        ('affect', 'morality_given_valence', 'affect prompts: morality | valence', '#d95f02', 0.35),
        ('moral', 'valence_given_morality', 'moral prompts: valence | morality', '#7570b3', 0.35),
        ('moral', 'morality_given_valence', 'moral prompts: morality | valence', '#7570b3', 1.0)]
for i, (a, col, name, c, alpha) in enumerate(bars):
    v = [l[(l.space == s) & (l.axis == a)][col].iloc[0] for s in spaces]
    ax[2].bar(x + (i - 1.5) * w, v, w, color=c, alpha=alpha, label=name)
ax[2].axhline(0, color='#888', lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels(slab)
ax[2].set_ylabel('partial Spearman ρ'); ax[2].set_title('C. Language axes, controlling for the other rating', loc='left')
ax[2].legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2); ax[2].set_ylim(-0.3, 0.62); ax[2].grid(axis='y', color='#eee')

fig.suptitle('DINOv2 predicts valence and morality beyond object identity, and its features recover CLIP\'s affect-language axis', x=0.01, ha='left', fontsize=12, fontweight='bold')
fig.text(0.01, -0.1, 'All probes: standardize → PCA(256) → ridge, 5-fold out-of-fold, 2,940 SMID images, DINOv2-G layer 37. "Morality beyond valence": morality rating minus its linear fit on valence. '
         'B: each rating is first predicted from the controls (cross-fitted); the probe is then trained on what is left. C: images are scored along CLIP text directions, mean of 5 affect and 6 moral prompt pairs; '
         'DINOv2 features are mapped into CLIP image space with a linear map fit either on SMID images (no ratings used) or on 5,000 COCO val2017 photos.', fontsize=8, color='#666', wrap=True)
fig.tight_layout(); fig.savefig('figures/fig_controls_and_language.png', dpi=190, bbox_inches='tight'); print('ok')
