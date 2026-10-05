"""Bounded-goal decision, full-val repair/harm and paired recall uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for key in ['source', 'annotations', 'out']: parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args(); assert (args.source/'COMPLETE.json').exists()
    results = json.loads((args.source/'RESULTS.json').read_text())
    matched = {k: set(v) for k, v in json.loads((args.source/'MATCHED_GT75.json').read_text()).items()}
    gate = json.loads((args.source/'GATE.json').read_text())
    coco = json.loads(args.annotations.read_text()); image_ids = sorted(im['id'] for im in coco['images'])
    index = {iid: i for i, iid in enumerate(image_ids)}
    anns = [a for a in coco['annotations'] if not a.get('iscrowd', 0) and not a.get('ignore', 0)]
    def size(ann): return 'small' if ann['area'] < 1024 else 'medium' if ann['area'] < 9216 else 'large'
    pick = np.random.default_rng(20260920).integers(0, len(image_ids), (2000, len(image_ids)))
    comparisons = {}
    for other in ['baseline', 'scalar', 'local', 'shared_template']:
        groups = {}; left_only = matched['shared2']-matched[other]; right_only = matched[other]-matched['shared2']
        for group in ['all', 'small', 'medium', 'large']:
            selected = [a for a in anns if group == 'all' or size(a) == group]
            counts = np.zeros(len(image_ids)); sums = counts.copy()
            for ann in selected:
                i = index[ann['image_id']]; counts[i] += 1
                sums[i] += int(ann['id'] in left_only)-int(ann['id'] in right_only)
            boot = 100*sums[pick].sum(1)/counts[pick].sum(1)
            groups[group] = {'gt': len(selected), 'new_only_good': sum(a['id'] in left_only for a in selected),
                'other_only_good': sum(a['id'] in right_only for a in selected),
                'recall75_difference_pp': float(100*sums.sum()/counts.sum()), 'ci95_pp': np.quantile(boot, [.025, .975]).tolist()}
        comparisons['shared2_minus_'+other] = {'mask75': groups,
            'metrics_delta_pp': {k: 100*(results['shared2']['metrics'][k]-results[other]['metrics'][k]) for k in results['shared2']['metrics']}}
    damage_base = matched['baseline']; transitions = {}
    for name in ['scalar', 'local', 'shared_template']:
        old_repairs = matched[name]-damage_base; new_repairs = matched['shared2']-damage_base
        old_damages = damage_base-matched[name]; new_damages = damage_base-matched['shared2']
        transitions[name] = {'retained_old_repairs': len(new_repairs & old_repairs), 'lost_old_repairs': len(old_repairs-new_repairs),
            'new_repairs': len(new_repairs-old_repairs), 'avoided_old_damages': len(old_damages-new_damages),
            'new_damages': len(new_damages-old_damages), 'shared_damages': len(new_damages & old_damages)}
    goal = {'gate': gate, 'action': 'replicate seeds1/2' if gate['pass'] else 'stop this bounded candidate; no further seeds or tuning',
        'achieved_ap_gain_pp': 100*(results['shared2']['metrics']['AP']-results['baseline']['metrics']['AP']),
        'retained_reference_gain_fraction': (results['shared2']['metrics']['AP']-results['baseline']['metrics']['AP']) /
            (results['shared_template']['metrics']['AP']-results['baseline']['metrics']['AP']),
        'uncertainty': 'Paired 2000 image-cluster intervals for Mask75 recall only, not AP or seeds. val2017 repeatedly inspected; exploratory.'}
    args.out.mkdir(parents=True, exist_ok=True)
    output = {'goal': goal, 'comparisons': comparisons, 'repair_harm_transitions': transitions, 'results': results}
    (args.out/'ANALYSIS.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    methods = ['baseline', 'scalar', 'local', 'shared_template', 'shared2']
    labels = ['Frozen YOLO', 'Learned scalar', 'Learned 16-grid', 'Shared shape\n(local-head strength)', 'Independent\nbias + amplitude']
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    vals = [100*results[name]['metrics']['AP'] for name in methods]
    bars = axs[0].bar(range(5), vals, color=['#969fa4', '#a7c8ce', '#237b91', '#bd8b42', '#4267a8'])
    axs[0].bar_label(bars, labels=[f'{v:.3f}' for v in vals]); axs[0].set_ylim(min(vals)-.4, max(max(vals)+.25, 44.55))
    axs[0].axhline(44.30, color='#b74835', ls='--', label='Predeclared 44.30 gate'); axs[0].legend(fontsize=8)
    axs[0].set_xticks(range(5), labels, rotation=20, ha='right'); axs[0].set_ylabel('COCO mask AP (0–100)'); axs[0].set_title('All 5000 COCO val2017 images')
    x = np.arange(4)
    repairs = [len(matched[name]-damage_base) for name in methods[1:]]
    harms = [len(damage_base-matched[name]) for name in methods[1:]]
    axs[1].bar(x-.18, repairs, width=.36, label='Mask75 repairs', color='#237b91')
    axs[1].bar(x+.18, harms, width=.36, label='Mask75 harms', color='#b76c51')
    axs[1].set_xticks(x, labels[1:], rotation=20, ha='right'); axs[1].set_ylabel('GT instances'); axs[1].legend(fontsize=8)
    axs[1].set_title('Official score-ordered GT matching')
    fig.savefig(args.out/'shared2_fullval.png', dpi=180); fig.savefig(args.out/'shared2_fullval.pdf'); plt.close(fig)
    print(json.dumps(goal), flush=True)


if __name__ == '__main__': main()
