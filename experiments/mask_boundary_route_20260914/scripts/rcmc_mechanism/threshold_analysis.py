"""Summarize the frozen threshold panel; GT-selected thresholds are diagnostics."""
from pathlib import Path
import argparse
import csv
import json
from collections import defaultdict
from common import setup, atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol, output = setup(args.protocol, args.output)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = Path(args.input)
    rows = json.loads((source / 'instances.json').read_text())
    by_threshold = defaultdict(list)
    with (source / 'threshold_pixels.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['variant'] == 'threshold':
                by_threshold[(row['role'], float(row['threshold']))].append(row)
    thresholds = protocol['thresholds']
    methods = ['response', 'smooth', 'coverage_loss_0.01', 'coverage_loss_0.02', 'best_grid']
    result = {'source_run': source.name, 'seed': 20260920, 'bootstrap_repeats': 2000,
              'resampling_unit': 'image, keeping all selected objects in the resampled image',
              'groups': {}, 'partitions': {}, 'curves': {},
              'limitations': [
                  'The target panel is category/size-stratified, not population prevalence.',
                  'Coverage-budget and best-grid thresholds use per-instance GT; they are not deployable methods.',
                  'Image partitions come from previously explored validation images, not new blind data.',
                  'Success controls have high coverage by construction; their damage rate cannot represent all successful objects.',
                  'Exploratory intervals are pointwise, not multiplicity-adjusted, and are not AP intervals.']}

    def values(row, method):
        selected = row[method]
        iou = selected['best_iou'] if method.startswith('coverage_loss_') else selected['iou']
        repair = float(row['baseline_iou'] < .75 <= iou)
        return [repair, iou - row['baseline_iou'],
                float(iou < .75 <= row['baseline_iou'])]

    for role in ['target', 'control']:
        group = [r for r in rows if r['role'] == role]
        ids = sorted({r['image_id'] for r in group})
        lookup = {iid: i for i, iid in enumerate(ids)}
        counts = np.zeros(len(ids), dtype=np.int64)
        totals = np.zeros((len(ids), len(methods), 3))
        for row in group:
            i = lookup[row['image_id']]
            counts[i] += 1
            for j, method in enumerate(methods):
                totals[i, j] += values(row, method)
        rng = np.random.default_rng(20260920)
        sample = rng.integers(0, len(ids), size=(2000, len(ids)))
        estimates = totals[sample].sum(axis=1) / counts[sample].sum(axis=1)[:, None, None]
        stats = {'n': len(group), 'images': len(ids),
                 'baseline_iou': float(np.mean([r['baseline_iou'] for r in group])),
                 'baseline_coverage': float(np.mean([r['baseline_coverage'] for r in group])), 'methods': {}}
        for j, method in enumerate(methods):
            mean = totals[:, j].sum(axis=0) / len(group)
            lo, hi = np.quantile(estimates[:, j], [.025, .975], axis=0)
            data = {key: {'mean': float(mean[k]), 'image_bootstrap_ci95': [float(lo[k]), float(hi[k])]}
                    for k, key in enumerate(['repair_rate', 'iou_gain', 'damage_rate'])}
            data['repair_count'] = int(totals[:, j, 0].sum())
            data['damage_count'] = int(totals[:, j, 2].sum())
            if not method.startswith('coverage_loss_'):
                data['mean_coverage_loss'] = float(np.mean([r['baseline_coverage'] - r[method]['coverage'] for r in group]))
            else:
                data['max_fp_removed_fraction_mean'] = float(np.mean([r[method]['max_fp_removed_fraction'] for r in group
                                                                         if r[method]['max_fp_removed_fraction'] is not None]))
            stats['methods'][method] = data
        pair = estimates[:, methods.index('coverage_loss_0.02')] - estimates[:, methods.index('response')]
        difference = (totals[:, methods.index('coverage_loss_0.02')].sum(axis=0)
                      - totals[:, methods.index('response')].sum(axis=0)) / len(group)
        stats['diagnostic_2pp_minus_response'] = {
            key: {'mean': float(difference[k]), 'image_bootstrap_ci95': np.quantile(pair[:, k], [.025, .975]).tolist()}
            for k, key in enumerate(['repair_rate', 'iou_gain', 'damage_rate'])}
        result['groups'][role] = stats
        result['partitions'][role] = {}
        for partition in ['exploratory', 'confirmation']:
            selected = [r for r in group if r['partition'] == partition]
            result['partitions'][role][partition] = {'n': len(selected), 'methods': {
                method: {'repairs': int(sum(values(r, method)[0] for r in selected)),
                         'mean_iou_gain': float(np.mean([values(r, method)[1] for r in selected]))}
                for method in methods}}
        baseline = {(int(r['image_id']), int(r['annotation_id'])): r for r in by_threshold[(role, 0.0)]}
        curve = []
        for threshold in thresholds:
            selected = by_threshold[(role, float(threshold))]
            fp = []; coverage = []; ious = []
            for row in selected:
                base = baseline[(int(row['image_id']), int(row['annotation_id']))]
                if float(base['fp']):
                    fp.append((float(base['fp']) - float(row['fp'])) / float(base['fp']))
                coverage.append(float(base['coverage']) - float(row['coverage']))
                ious.append(float(row['iou']))
            curve.append({'threshold': threshold, 'mean_fp_removed_fraction': float(np.mean(fp)),
                          'mean_coverage_loss': float(np.mean(coverage)), 'mean_iou': float(np.mean(ious))})
        result['curves'][role] = curve
    atomic(output / 'SUMMARY.json', result)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
    for role, color, label in [('target', '#c65542', 'Target failures (n=256)'),
                               ('control', '#357da8', 'High-coverage successes (n=217)')]:
        curve = result['curves'][role]
        axes[0].plot(thresholds, [r['mean_iou'] * 100 for r in curve], 'o-', color=color, label=label, ms=3)
        axes[1].plot([r['mean_coverage_loss'] * 100 for r in curve],
                     [r['mean_fp_removed_fraction'] * 100 for r in curve], 'o-', color=color, ms=3, label=label)
    axes[0].set(xlabel='Common logit threshold', ylabel='Mean mask IoU (%)', title='A. Fixed common thresholds')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Mean target coverage lost (percentage points)', ylabel='Mean FP removed (%)',
                title='B. Removal / coverage trade-off')
    labels = ['RCMC', 'Fixed\nsmooth', 'GT oracle\nloss <= 1 pp', 'GT oracle\nloss <= 2 pp', 'GT oracle\nunrestricted']
    means = []; errors = [[], []]
    for method in methods:
        metric = result['groups']['target']['methods'][method]['repair_rate']
        mean = 100 * metric['mean']; lo, hi = [v * 100 for v in metric['image_bootstrap_ci95']]
        means.append(mean); errors[0].append(mean - lo); errors[1].append(hi - mean)
    bars = axes[2].bar(range(len(methods)), means, yerr=errors, capsize=3,
                       color=['#357da8', '#679bbd', '#da9868', '#cf784e', '#b25b42'])
    axes[2].bar_label(bars, labels=[f'{v:.1f}' for v in means], padding=7, fontsize=9)
    axes[2].set_xticks(range(len(methods)), labels, fontsize=8)
    axes[2].set(ylabel='Target failures repaired at IoU 0.75 (%)', ylim=(0, 75), title='C. Existing response has usable headroom')
    fig.suptitle('Frozen YOLO26m-seg outputs: threshold diagnostics on COCO (432 images)', fontsize=13)
    fig.savefig(output / 'threshold_diagnostics.png', dpi=180)
    fig.savefig(output / 'threshold_diagnostics.pdf')
    plt.close(fig)
    print(json.dumps({'completed': True, 'targets': len([r for r in rows if r['role'] == 'target']),
                      'diagnostic_2pp_minus_response': result['groups']['target']['diagnostic_2pp_minus_response']}))


if __name__ == '__main__':
    main()
