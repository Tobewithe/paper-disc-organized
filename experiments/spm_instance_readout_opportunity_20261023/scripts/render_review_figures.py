"""Presentation-only replot of accepted summary data; original Run stays immutable."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
run = root / 'runs/RUN_SPM_INSTANCE_OPPORTUNITY_4d95693a2383'
out = root / 'presentation'
out.mkdir(exist_ok=True)
opportunity = json.loads((run / 'OPPORTUNITY_SUMMARY.json').read_text())
coverage = json.loads((run / 'TRAINING_INSTANCE_COVERAGE.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
labels = ['Trained\n128D', 'Random\n128D', 'Raw\n512D*']
colors = ['#0072B2', '#E69F00', '#999999']
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), layout='constrained')
for ax, key, title, gate, limits in zip(
    axes,
    ['mean_iou_delta', 'mask75_fraction', 'prototype_coverage'],
    ['Mean IoU change', 'Fraction repaired to Mask IoU >= 0.75', 'GT support available'],
    [.05, .20, .90],
    [(-.27, .095), (0, .26), (0, 1.16)],
):
    values = [opportunity[t]['failure'][key] for t in ['T', 'U', 'R']]
    ax.bar(range(3), values, color=colors, width=.65)
    ax.set_xticks(range(3), labels)
    ax.axhline(0, color='#777777', lw=.6)
    ax.axhline(gate, color='black', ls='--', lw=1)
    ax.set_ylim(*limits)
    ax.set_title(title, fontsize=10, pad=12)
    ax.text(.02, .97, f'Frozen requirement: {gate:g}', transform=ax.transAxes, va='top', fontsize=8)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.3f}', (i, value), xytext=(0, 5 if value >= 0 else -7),
                    textcoords='offset points', ha='center', va='bottom' if value >= 0 else 'top')
fig.suptitle('GT-supported instance-mean readout | 305 fixed failures\nLimited readout diagnostic; not a universal upper bound. *Raw feature dimension differs.', fontsize=11)
fig.savefig(out / '01_GT_supported_readout.png', dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
sizes = ['S', 'M', 'L']
totals = [coverage['by_size'][s]['total'] for s in sizes]
sampled = [coverage['by_size'][s]['sampled'] for s in sizes]
axes[0].bar(sizes, totals, color='#cccccc', label='All training instances', width=.6)
axes[0].bar(sizes, sampled, color='#0072B2', label='Selected in fixed episodes', width=.6)
for i, (a, b) in enumerate(zip(totals, sampled)):
    axes[0].text(i, a + 180, f'{b}/{a}\n{b/a:.1%}', ha='center', va='bottom')
axes[0].set_ylim(0, max(totals) * 1.25)
axes[0].set_title('Coverage by object area')
axes[0].set_ylabel('Number of instances')
axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, fontsize=8, frameon=False)
counts = {}
for frame in coverage['frames']:
    key = (frame['instances'], frame['sampled_instances'])
    counts[key] = counts.get(key, 0) + 1
for (n, selected), count in counts.items():
    axes[1].scatter(n, selected, s=16 + count ** .5 * 4, alpha=.7, color='#0072B2')
axes[1].plot([0, 30], [0, 30], '--', c='gray', label='All instances selected')
axes[1].axhline(8, c='#D55E00', ls=':', label='Per-image cap: 8')
axes[1].set(xlabel='GT instances per image', ylabel='Instances in its fixed episode',
            ylim=(0, 31), title='Coverage in dense images')
axes[1].legend(loc='upper left', fontsize=8)
fig.suptitle('Actual training extraction coverage | 1272 real training images\nFixed-episode coverage, not optimizer revisit frequency', fontsize=11)
fig.savefig(out / '02_training_coverage.png', dpi=160)
plt.close(fig)
manifest = {
    'scope': 'presentation-only; unchanged accepted summary values',
    'source_run_id': run.name,
    'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.png'))},
    'input_sha256': {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                     for name in ['OPPORTUNITY_SUMMARY.json', 'TRAINING_INSTANCE_COVERAGE.json']},
}
(out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'figures': list(manifest['files']), 'original_run_modified': False}))
