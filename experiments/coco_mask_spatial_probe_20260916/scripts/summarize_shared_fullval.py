"""Full-val outcome differences, reproduction and image-cluster recall intervals."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for key in ['source', 'reference', 'annotations', 'out']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    assert (args.source / 'COMPLETE.json').exists()
    result = json.loads((args.source / 'RESULTS.json').read_text())
    matched = {name: set(ids) for name, ids in json.loads((args.source / 'MATCHED_GT75.json').read_text()).items()}
    prior = json.loads((args.reference / 'RESULTS.json').read_text())
    prior_match = json.loads((args.reference / 'MATCHED_GT75.json').read_text())
    reproduction = {}
    for name, old_name in [('baseline', 'baseline'), ('local', 'local4_center_s0')]:
        differences = {key: result[name]['metrics'][key] - value for key, value in prior[old_name]['metrics'].items()}
        reproduction[name] = {'max_absolute_metric_difference': max(abs(v) for v in differences.values()),
            'metric_differences': differences, 'same_mask75_gt_ids': matched[name] == set(prior_match[old_name])}
    coco = json.loads(args.annotations.read_text())
    ids = sorted(im['id'] for im in coco['images']); index = {iid: j for j, iid in enumerate(ids)}
    anns = [ann for ann in coco['annotations'] if not ann.get('iscrowd', 0) and not ann.get('ignore', 0)]
    def size(ann): return 'small' if ann['area'] < 1024 else 'medium' if ann['area'] < 9216 else 'large'
    pick = np.random.default_rng(20260920).integers(0, len(ids), (2000, len(ids)))
    contrasts = {}
    for left, right in [('local', 'baseline'), ('shared_template', 'baseline'), ('local', 'shared_template')]:
        groups = {}
        left_only = matched[left] - matched[right]
        right_only = matched[right] - matched[left]
        for group in ['all', 'small', 'medium', 'large']:
            selected = [ann for ann in anns if group == 'all' or size(ann) == group]
            sums = np.zeros(len(ids)); counts = np.zeros(len(ids))
            for ann in selected:
                j = index[ann['image_id']]
                sums[j] += int(ann['id'] in matched[left]) - int(ann['id'] in matched[right]); counts[j] += 1
            boot = 100 * sums[pick].sum(1) / counts[pick].sum(1)
            groups[group] = {'gt': len(selected), 'left_only_good': sum(ann['id'] in left_only for ann in selected),
                'right_only_good': sum(ann['id'] in right_only for ann in selected),
                'recall75_difference_pp': float(100 * sums.sum() / counts.sum()), 'ci95_pp': np.quantile(boot, [.025, .975]).tolist()}
        contrasts[left + '_minus_' + right] = {'mask75': groups,
            'metric_differences_pp': {key: 100 * (result[left]['metrics'][key] - result[right]['metrics'][key]) for key in result[left]['metrics']}}
    args.out.mkdir(parents=True, exist_ok=True)
    output = {'scope': 'All 5000 val2017 images, official score-ordered COCO matches; exploratory, one frozen trained seed.',
        'reproduction': reproduction, 'contrasts': contrasts, 'results': result,
        'uncertainty': 'CI applies to paired image-cluster Mask75 recall difference, not to AP or training-seed variability.'}
    (args.out / 'FULLVAL_COMPARISON.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    metrics = ['AP', 'AP50', 'AP75', 'APS', 'APM', 'APL']
    for shift, name, label, color in [(-.18, 'local', 'Instance layout', '#237b91'), (.18, 'shared_template', 'Shared shape + instance strength', '#bd8b42')]:
        d = contrasts[name + '_minus_baseline']
        axs[0].bar(np.arange(len(metrics)) + shift, [d['metric_differences_pp'][key] for key in metrics], width=.36, color=color, label=label)
        values = [d['mask75'][group] for group in ['all', 'small', 'medium', 'large']]
        means = np.array([v['recall75_difference_pp'] for v in values]); ci = np.array([v['ci95_pp'] for v in values])
        axs[1].errorbar(np.arange(4) + shift / 2, means, yerr=np.maximum(np.vstack([means-ci[:, 0], ci[:, 1]-means]), 0), fmt='o', capsize=3, color=color)
    axs[0].set_xticks(range(6), ['AP', 'AP50', 'AP75', 'AP small', 'AP med.', 'AP large'])
    axs[1].set_xticks(range(4), ['All', 'Small', 'Medium', 'Large'])
    axs[0].set_title('COCO mask AP changes'); axs[1].set_title('Mask75 recall changes (95% image bootstrap CI)')
    for ax in axs: ax.axhline(0, color='gray', linewidth=1); ax.grid(axis='y', alpha=.2); ax.set_ylabel('Change from frozen baseline, percentage points')
    axs[0].legend(fontsize=8); fig.suptitle('All 5000 COCO val2017 images; fixed boxes, scores and prototypes')
    fig.savefig(args.out / 'fullval_shared_shape.png', dpi=180); fig.savefig(args.out / 'fullval_shared_shape.pdf'); plt.close(fig)
    print(json.dumps({'reproduction': reproduction, 'local_minus_shared_template': contrasts['local_minus_shared_template']}))


if __name__ == '__main__':
    main()
