"""Final predeclared three-seed decision; no new model/strength selection."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, nargs=3, required=True)
    for key in ['timing', 'annotations', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    results, matches, records = [], [], []
    for seed, source in enumerate(args.sources):
        record = json.loads((source/'run.json').read_text()); assert record['status'] == 'completed'
        result = json.loads((source/'RESULTS.json').read_text())
        assert result['shared2']['seed'] == seed
        assert result['local'].get('seed', 0) == seed
        assert abs(result['baseline']['metrics']['AP']-.4351845499048806) < 1e-12
        results.append(result); matches.append({k: set(v) for k, v in json.loads((source/'MATCHED_GT75.json').read_text()).items()})
        records.append({'run_id': record['run_id'], 'source': str(source), 'seed': seed})
    for matched in matches[1:]: assert matched['baseline'] == matches[0]['baseline']
    methods = ['baseline', 'scalar', 'local', 'shared2']; summary = {}
    for method in methods:
        metrics = {}
        for name in results[0][method]['metrics']:
            values = np.array([100*result[method]['metrics'][name] for result in results])
            metrics[name] = {'values': values.tolist(), 'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1))}
        counts = {}
        for name in ['matched75', 'repaired75', 'damaged75']:
            values = []
            for matched in matches:
                good, base = matched[method], matched['baseline']
                values.append(len(good) if name == 'matched75' else len(good-base) if name == 'repaired75' else len(base-good))
            counts[name] = {'values': values, 'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))}
        summary[method] = {'metrics': metrics, 'counts': counts}
    delta_ap = {other: [100*(result['shared2']['metrics']['AP']-result[other]['metrics']['AP']) for result in results]
        for other in ['baseline', 'scalar', 'local']}
    passed = summary['shared2']['metrics']['AP']['mean'] >= 44.30 and all(value > 0 for value in delta_ap['scalar'])
    gate = {'threshold_mean_ap': 44.30, 'observed_mean_ap': summary['shared2']['metrics']['AP']['mean'],
        'paired_scalar_ap_gains': delta_ap['scalar'], 'pass': bool(passed),
        'decision': 'retain as a validated constrained mask-calibration candidate; bounded experiment ends'
                    if passed else 'stop this candidate under the preset data/training budget; no more seeds or tuning',
        'not_a_statistical_significance_gate': True}
    coco = json.loads(args.annotations.read_text()); image_ids = sorted(im['id'] for im in coco['images']); index = {iid: j for j, iid in enumerate(image_ids)}
    anns = [ann for ann in coco['annotations'] if not ann.get('iscrowd', 0) and not ann.get('ignore', 0)]
    pick = np.random.default_rng(20260920).integers(0, len(image_ids), (2000, len(image_ids)))
    def size(ann): return 'small' if ann['area'] < 1024 else 'medium' if ann['area'] < 9216 else 'large'
    recall = {}
    for other in ['baseline', 'scalar', 'local']:
        groups = {}
        for group in ['all', 'small', 'medium', 'large']:
            chosen = [ann for ann in anns if group == 'all' or size(ann) == group]
            counts = np.zeros(len(image_ids)); sums = counts.copy()
            for ann in chosen:
                j = index[ann['image_id']]; counts[j] += 1
                sums[j] += np.mean([int(ann['id'] in matched['shared2'])-int(ann['id'] in matched[other]) for matched in matches])
            samples = 100*sums[pick].sum(1)/counts[pick].sum(1)
            groups[group] = {'gt': len(chosen), 'mean_seed_recall_delta_pp': float(100*sums.sum()/counts.sum()),
                'image_only_ci95_pp': np.quantile(samples, [.025, .975]).tolist()}
        recall['shared2_minus_'+other] = groups
    timing = json.loads(args.timing.read_text())
    output = {'sources': records, 'gate': gate, 'summary': summary, 'paired_ap_differences_pp': delta_ap,
        'mean_seed_mask75_contrasts': recall, 'timing': timing,
        'limits': 'Three fixed seeds on same 800/200 images and repeatedly explored COCO val2017. SD is between seeds; bootstrap CI is image sampling conditional on these seeds, not AP CI or independent replication.'}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'FINAL_RESULTS.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.7), layout='constrained')
    labels = ['Frozen YOLO', 'Scalar', '16-grid local', 'Bias + amplitude']
    for seed, color in enumerate(['#2d728f', '#ca9242', '#864f91']):
        values = [100*results[seed][name]['metrics']['AP'] for name in methods]
        axs[0].plot(np.arange(4)+(seed-1)*.07, values, 'o-', color=color, alpha=.85, label=f'Seed {seed}')
    axs[0].axhline(44.30, color='#a8483c', ls='--', label='44.30 mean AP gate')
    axs[0].set_xticks(range(4), labels, rotation=12); axs[0].set_ylabel('COCO mask AP (0–100)'); axs[0].legend(fontsize=8)
    axs[0].set_title('Same split and 8-epoch budget')
    for shift, key, label, color in [(-.18, 'repaired75', 'Repairs', '#237b91'), (.18, 'damaged75', 'Harms', '#b76c51')]:
        values = [summary[name]['counts'][key]['mean'] for name in methods[1:]]
        sd = [summary[name]['counts'][key]['sample_sd'] for name in methods[1:]]
        axs[1].bar(np.arange(3)+shift, values, yerr=sd, width=.36, capsize=4, color=color, label=label)
    axs[1].set_xticks(range(3), labels[1:]); axs[1].set_ylabel('GT instances, mean ± seed SD'); axs[1].legend(fontsize=8)
    axs[1].set_title('Mask75 repair–harm balance')
    fig.suptitle('Independent two-output head: full 5000-image COCO evaluation')
    fig.savefig(args.out/'shared2_three_seeds.png', dpi=180); fig.savefig(args.out/'shared2_three_seeds.pdf'); plt.close(fig)
    print(json.dumps({'gate': gate, 'ap': {name: summary[name]['metrics']['AP'] for name in methods}}), flush=True)


if __name__ == '__main__': main()
