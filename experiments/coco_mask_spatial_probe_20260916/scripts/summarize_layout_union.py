"""Summarize both benefit and harm in the previously frozen diagnostic cohort."""
import argparse
import json
from pathlib import Path

from analyze_spatial_layout import estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'instances.jsonl').read_text().splitlines()]
    result = {
        'scope': 'All 2241 historically changed GTs, including repair and damage; outcome-conditioned, not whole COCO or AP.',
        'n': len(rows), 'images': len({row['image_id'] for row in rows}),
        'arms': {}, 'local_minus_shared_template': {},
    }
    for name in rows[0]['metrics']:
        result['arms'][name] = {
            'iou': estimate(rows, [100 * row['metrics'][name]['iou'] for row in rows]),
            'mask75': sum(row['metrics'][name]['iou'] >= .75 for row in rows),
        }
    for metric in ['iou', 'coverage', 'purity']:
        result['local_minus_shared_template'][metric] = estimate(rows, [
            100 * (row['metrics']['local'][metric] - row['metrics']['shared_template'][metric]) for row in rows
        ])
    result['local_minus_shared_template']['mask75_pp'] = estimate(rows, [
        100 * (int(row['metrics']['local']['iou'] >= .75) - int(row['metrics']['shared_template']['iou'] >= .75)) for row in rows
    ])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'UNION_SUMMARY.json').write_text(json.dumps(result, indent=2, allow_nan=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prior = json.loads(args.summary.read_text())
    names = ['local_only_repair', 'scalar_only_repair', 'both_repair',
             'local_only_damage', 'scalar_only_damage', 'both_damage']
    labels = ['Local-only repair (582)', 'Scalar-only repair (134)', 'Both repair (756)',
              'Local-only harm (302)', 'Scalar-only harm (152)', 'Both harm (315)', 'All changed GTs (2241)']
    stats = [prior['groups'][name]['contrasts']['local_minus_shared_template']['iou'] for name in names]
    stats.append(result['local_minus_shared_template']['iou'])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for index, value in enumerate(stats):
        mean = value['mean']; low, high = value['ci95']
        ax.errorbar(mean, index, xerr=[[max(0, mean-low)], [max(0, high-mean)]],
                    fmt='o', capsize=4, color='#207d90' if mean >= 0 else '#a64b35')
    ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
    ax.axvline(0, color='gray', linewidth=1); ax.grid(axis='x', alpha=.2)
    ax.set_xlabel('Learned local minus shared training shape: mask IoU percentage points')
    ax.set_title('Benefits and harms of instance-specific layout\nFixed prediction identity; conditional diagnosis, not AP')
    fig.savefig(args.out / 'layout_benefits_and_harms.png', dpi=180)
    fig.savefig(args.out / 'layout_benefits_and_harms.pdf'); plt.close(fig)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
