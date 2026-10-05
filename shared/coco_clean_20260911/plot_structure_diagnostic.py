"""Standalone figure from completed diagnostics, with matched-cohort labels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
source = root / 'diagnostics/structure_main300_20260911'
out = root.parent.parent / 'refine-logs/coco-structure/figures'
out.mkdir(exist_ok=True)
summary = json.loads((source / 'STRUCTURE_SUMMARY.json').read_text())
paired = json.loads((source / 'PAIRED_STRUCTURE_ANALYSIS.json').read_text())
groups = {r['group']: r for r in summary['stage_same_iou']}
colors = {'high': '#B04B36', 'low': '#356989'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, axs = plt.subplots(1, 3, figsize=(15.8, 4.7), gridspec_kw={'width_ratios': [1, 1.15, 1.2]})
for group, offset, label in [('high', -.16, 'High ICI (788 GT)'),
                              ('low', .16, 'Other (2,847 GT)')]:
    v = groups[group]
    x = np.arange(2) + offset
    bars = axs[0].bar(x, [v['bbox75'], v['segm75']], width=.3,
                      color=colors[group], label=label)
    axs[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
axs[0].set(xticks=[0, 1], xticklabels=['Box R75', 'Mask R75'],
           ylabel='GT recovery (%)', ylim=(0, 100), title='A. Same IoU threshold: 0.75')
axs[0].legend(loc='upper right', frameon=False, fontsize=9)
keys = ['raw_geometry', 'argmax_class', 'score', 'nms', 'top300', 'eval100']
for group, label in [('high', 'High ICI'), ('low', 'Other')]:
    axs[1].plot(range(len(keys)), [groups[group][k + '_availability75_pct'] for k in keys],
                '-o', color=colors[group], label=label, markersize=4)
axs[1].set(xticks=range(len(keys)), xticklabels=['Raw', 'Class', 'Score', 'NMS', '300', '100/class'],
           ylabel='GT with an available box at IoU >= 0.75 (%)', ylim=(68, 89),
           title='B. Candidate availability, not recall')
axs[1].tick_params(axis='x', labelrotation=30)
axs[1].legend(frameon=False, fontsize=9)
cohort = paired['proto_high_same_adjacent']['both_bbox50_matched']
names = ['Actual target score', 'Target - neighbor score', 'Coordinate oracle', 'Prototype oracle']
metrics = ['actual_a_auc', 'actual_pair_difference_auc', 'coordinate_auc', 'auc']
for j, key in enumerate(metrics):
    row = cohort[key]
    axs[2].errorbar(row['mean'], j, xerr=[[row['mean'] - row['ci_low']],
                                       [row['ci_high'] - row['mean']]],
                    fmt='o', color='#356989' if j < 2 else '#777777', capsize=3)
    axs[2].text(row['mean'], j - .19, f"{row['mean']:.4f}", ha='center', fontsize=9)
axs[2].set(yticks=range(4), yticklabels=names, ylim=(-.5, 3.5), xlim=(.90, 1.005),
           xlabel='Own-vs-neighbor pixel AUC', title='C. Same 167 high-ICI adjacent pairs')
axs[2].invert_yaxis()
for ax in axs:
    ax.set_axisbelow(True)
    ax.grid(axis='y' if ax is not axs[2] else 'x', alpha=.16)
fig.suptitle('Frozen pretrained model | exploratory COCO structure diagnostic (300 images)',
             fontsize=14, x=.04, ha='left')
fig.text(.04, .015,
         'C: GT-selected pairs and within-image oracle probes; pointwise 95% image-bootstrap intervals. '
         'Pixel AUC is not AP. High/other groups are not composition-matched.', fontsize=9, color='#444444')
fig.tight_layout(rect=[0, .055, 1, .91], w_pad=2)
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(out / f'STRUCTURE_DIAGNOSTIC_20260911.{ext}', dpi=180, bbox_inches='tight')
print(out / 'STRUCTURE_DIAGNOSTIC_20260911.png')
