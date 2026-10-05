"""Train-domain policy selection and a separate, frozen full-COCO evaluation."""
from pathlib import Path
import argparse
import csv
import json
import time
import gc
import contextlib
import io
from common import setup, atomic, progress, load_predictions


def main():
    parser = argparse.ArgumentParser()
    for arg in ['protocol', 'output', 'models', 'phase']:
        parser.add_argument('--'+arg, required=True)
    parser.add_argument('--selection')
    args = parser.parse_args(); p, out = setup(args.protocol, args.output)
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from calibrator import Calibrator, decisions
    from data_access import load_bank, subset_outcomes

    assert args.phase in ['selection', 'evaluation']
    start = time.perf_counter()
    split = json.loads(Path(p['split']).read_text())
    root = Path(p['train_bank'] if args.phase == 'selection' else p['val_bank'])
    bank = load_bank(root)
    ids = set(split['selection_ids']) if args.phase == 'selection' else set(bank['ids'])
    if args.phase == 'evaluation':
        assert len(ids) == 4500 and not ids & (set(split['fit_ids']) | set(split['selection_ids']))
    cal = Calibrator(args.models, p['frozen_rcmc'])
    progress(out, 'prediction', candidates=len(bank['features']), phase=args.phase)
    t = time.perf_counter(); scores = cal.predict(bank['features']); predict_seconds = time.perf_counter()-t
    eligible = bank['eligible']
    frozen_use = eligible & (scores['frozen_rcmc'] > 0)
    if args.phase == 'selection':
        reference = subset_outcomes(bank, frozen_use, ids)
        budget = reference['damages']
        policies = {}; search = []
        for name in ['direct', 'decomposed', 'decomposed_protected']:
            candidates = []
            for cutoff in p['cutoff_grid']:
                if name == 'direct':
                    use = eligible & (scores['direct'] > cutoff)
                else:
                    use = decisions(scores, eligible, cutoff, name == 'decomposed_protected', p['coverage_protection']['budget'])
                stat = subset_outcomes(bank, use, ids)
                item = {'variant': name, 'cutoff': cutoff, 'feasible': stat['damages'] <= budget, **stat}
                candidates.append(item); search.append(item)
            feasible = [r for r in candidates if r['feasible']]
            assert feasible, 'Keep-baseline cutoff should always be feasible'
            best = max(feasible, key=lambda r: (r['target_repairs'], -r['damages'], r['all_repairs'], r['iou_gain_sum'], r['cutoff']))
            policies[name] = {'cutoff': best['cutoff'], 'selection_outcomes': best,
                              'coverage_budget': .02 if name == 'decomposed_protected' else None}
        frozen = {'policies': policies, 'selection_image_ids': sorted(ids), 'damage_budget': budget,
                  'frozen_rcmc_outcomes': reference, 'search': search, 'selection_criterion': p['selection'],
                  'model_run': Path(args.models).name, 'scope': 'train2017 selection only; frozen before val scoring'}
        atomic(out / 'selection.json', frozen)
    else:
        frozen = json.loads(Path(args.selection).read_text())
        assert frozen['model_run'] == Path(args.models).name
        assert set(frozen['selection_image_ids']) == set(split['selection_ids'])
        policies = frozen['policies']
    flags = {'baseline': np.zeros(len(eligible), bool), 'frozen_rcmc': frozen_use}
    flags['direct'] = eligible & (scores['direct'] > policies['direct']['cutoff'])
    for name in ['decomposed', 'decomposed_protected']:
        flags[name] = decisions(scores, eligible, policies[name]['cutoff'], name.endswith('protected'), .02)
    key_array = np.asarray(bank['keys'], dtype=np.int64)
    np.savez_compressed(out / 'candidate_decisions.npz', keys=key_array, **flags)
    np.savez_compressed(out / 'predicted_components.npz', keys=key_array, **scores)
    quality = {}
    for field, truth in [('purity', 'baseline_purity'), ('coverage', 'baseline_coverage'),
                         ('removed_purity', 'removed_purity'), ('coverage_cost', 'coverage_cost'),
                         ('removed_fp_per_gt', 'fp_benefit'), ('gain', None), ('direct', None)]:
        rows = [r for r in bank['matched'] if r['image_id'] in ids
                and (field != 'removed_purity' or r['removed_tp'] + r['removed_fp'] > 1e-9)]
        pred = np.asarray([scores[field][r['index']] for r in rows])
        target = np.asarray([r[truth] if truth else r['trial_iou']-r['baseline_iou'] for r in rows])
        quality[field] = {'n': len(rows), 'mae': float(np.mean(np.abs(pred-target))),
                          'mean_prediction': float(pred.mean()), 'mean_truth': float(target.mean()),
                          'rmse': float(np.sqrt(np.mean((pred-target)**2)))}
        edges = np.unique(np.quantile(pred, np.linspace(0, 1, 11)))
        bins = []
        for k in range(len(edges)-1):
            mask = (pred >= edges[k]) & ((pred <= edges[k+1]) if k == len(edges)-2 else (pred < edges[k+1]))
            if mask.any():
                bins.append({'n': int(mask.sum()), 'prediction': float(pred[mask].mean()), 'truth': float(target[mask].mean())})
        quality[field]['quantile_calibration'] = bins
    atomic(out / 'component_quality.json', quality)
    outcomes = {name: subset_outcomes(bank, use, ids) for name, use in flags.items()}
    f = (out / 'instance_decisions.csv').open('w', newline='', encoding='utf-8'); writer = None
    for r in bank['matched']:
        if r['image_id'] not in ids: continue
        i = r['index']; row = dict(r)
        for name, use in flags.items(): row['use_'+name] = int(use[i])
        for name in ['gain','coverage_cost','removed_fp_per_gt','direct','frozen_rcmc']:
            row['predicted_'+name] = float(scores[name][i])
        if writer is None: writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
        writer.writerow(row)
    f.close()
    progress(out, 'load_rle_banks', phase=args.phase)
    base, basekeys = load_predictions(root/'predictions_official_zero.json')
    _, smooth = load_predictions(root/'predictions_smooth_gated.json')
    assert smooth.keys() <= basekeys.keys()
    for key, r in smooth.items():
        assert (r['category_id'], r['score']) == (basekeys[key]['category_id'], basekeys[key]['score'])
    indices = [bank['lookup'][(r['image_id'],r['candidate_index'])] for r in base]
    for r, i in zip(base, indices):
        assert bool(eligible[i]) == ((r['image_id'],r['candidate_index']) in smooth)
    annotations = p['train_annotations'] if args.phase == 'selection' else p['val_annotations']
    coco = COCO(annotations); metrics = {}
    for name, use in flags.items():
        progress(out, 'cocoeval', phase=args.phase, variant=name)
        output = []
        for r, i in zip(base, indices):
            if r['image_id'] not in ids: continue
            selected = smooth[(r['image_id'],r['candidate_index'])] if use[i] else r
            output.append({k: selected[k] for k in ['image_id','category_id','score','segmentation']})
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            detections = coco.loadRes(output); ev = COCOeval(coco, detections, 'segm'); ev.params.imgIds = sorted(ids)
            ev.evaluate(); ev.accumulate(); ev.summarize()
        values = [float(v) for v in ev.stats]
        metrics[name] = {'stats': values, 'ap': values[0]*100, 'ap75': values[2]*100,
                         'prediction_count': len(output), 'cocoeval_seconds': time.perf_counter()-t}
        (out / ('cocoeval_'+name+'.log')).write_text(log.getvalue(), encoding='utf-8')
        atomic(out / 'metrics_partial.json', metrics)
        del ev, detections, output; gc.collect()
        if args.phase == 'evaluation' and name in ['baseline','frozen_rcmc']:
            expected = .4337882998311341 if name == 'baseline' else .4393954420856892
            assert abs(values[0]-expected) < 1e-8, (name, values[0], expected)
    result = {'phase': args.phase, 'images': len(ids), 'policies': policies, 'outcomes': outcomes,
              'metrics': metrics, 'model_run': Path(args.models).name, 'bank': str(root),
              'ordinary_gt': sum(not a.get('iscrowd',0) for iid in ids for a in coco.imgToAnns[iid]),
              'prediction_seconds_all_candidates': predict_seconds, 'component_quality': quality,
              'elapsed_seconds': time.perf_counter()-start,
              'all_variants_preserve_scores_categories_boxes_counts': True,
              'limitations': p['scope_limits']}
    atomic(out / 'SUMMARY.json', result); progress(out, 'completed', phase=args.phase, metrics=metrics)


if __name__ == '__main__': main()
