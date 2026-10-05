"""Summarize fixed-head COCO pilots without rerunning inference or training.

Pointwise image-cluster bootstrap intervals condition on the three observed
training seeds. They do not measure training-population uncertainty or correct
for repeated model selection on COCO val2017. AP is summarized across seeds;
it is not bootstrapped or averaged from per-image AP.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


GROUPS = ['initial', 'bce_global', 'bce_local', 'dice_global', 'dice_local', 'plain_bce']
CONTRASTS = [('bce_global', 'initial'), ('dice_global', 'initial'),
             ('dice_global', 'bce_global'), ('bce_local', 'bce_global'),
             ('dice_local', 'dice_global')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bbox_iou(a, b):
    intersection = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return intersection / max(a[2]*a[3]+b[2]*b[3]-intersection, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    args = parser.parse_args()
    folder = args.input
    complete = json.loads((folder/'COMPLETE.json').read_text())
    protocol = json.loads((folder/'protocol.json').read_text())
    assert complete['status'] == 'COMPLETE' and complete['images'] == 5000
    image_ids = np.array(sorted(protocol['images']))
    assert len(np.unique(image_ids)) == 5000
    with (folder/'summary.csv').open() as handle:
        summary = list(csv.DictReader(handle))
    summary_by_arm = {row['arm']:row for row in summary}
    metrics = ['mask_ap', 'mask_ap50', 'mask_ap75', 'r75_all', 'r75_high', 'r75_low', 'pair_recovery75']
    tables = []
    for group in GROUPS:
        rows = [row for row in summary if row['arm']=='initial'] if group=='initial' else [row for row in summary if row['arm'].startswith(group+'_s')]
        assert len(rows) == (1 if group == 'initial' else 3)
        tables.append(dict(group=group, seeds=len(rows), metrics={metric: dict(mean_pct=float(np.mean([float(row[metric]) for row in rows])*100), seed_sd_pp=float(np.std([float(row[metric]) for row in rows],ddof=1)*100) if len(rows)>1 else None) for metric in metrics}))

    numerators, denominators, details = [], [], []
    recovery_data = None
    recovery_initial = None
    spatial_tables = []
    for filename, fields in [('gt_recovery.csv', ['mask_recovered75']),
                             ('spatial.csv', ['coverage', 'same_neighbor', 'background', 'mask_iou'])]:
        by_arm = {}
        with (folder/filename).open() as handle:
            for row in csv.DictReader(handle):
                values = [float(row[field]=='True') if field=='mask_recovered75' else float(row[field]) for field in fields]
                by_arm.setdefault(row['arm'], []).append([int(row['target_annotation']),int(row['image_id']),float(row['target_ici']),*values])
        arrays = {}
        for arm, rows in by_arm.items():
            part = np.array(rows, dtype=float)
            arrays[arm] = part[np.argsort(part[:,0])]
        del by_arm
        initial = arrays['initial']
        ids = initial[:,0].astype(np.int64)
        assert len(np.unique(ids))==len(ids)
        image_index = np.searchsorted(image_ids, initial[:,1])
        assert np.array_equal(image_ids[image_index], initial[:,1])
        data = {}
        for arm, part in arrays.items():
            assert np.array_equal(part[:,0], ids), (filename, arm, 'cohort mismatch')
            assert np.array_equal(part[:,1], initial[:,1])
            assert np.allclose(part[:,2], initial[:,2], rtol=0, atol=0)
            data[arm] = part[:,3:]
            assert np.isfinite(data[arm]).all()
        averaged = {group: data['initial'] if group == 'initial' else np.mean([data[f'{group}_s{seed}'] for seed in range(3)], axis=0) for group in GROUPS}
        high = initial[:,2] > .5+1e-10
        for group, selection in [('all', np.ones(len(initial), dtype=bool)), ('high', high), ('low', ~high)]:
            count = np.bincount(image_index[selection], minlength=len(image_ids))
            if filename == 'spatial.csv':
                for family, values in averaged.items():
                    spatial_tables.append(dict(family=family, group=group, targets=int(selection.sum()), metrics={field:float(values[selection,k].mean()*100) for k,field in enumerate(fields)}))
            for treatment, control in CONTRASTS:
                delta = averaged[treatment]-averaged[control]
                for k, field in enumerate(fields):
                    numerator = np.bincount(image_index[selection], weights=delta[selection,k], minlength=len(image_ids))
                    numerators.append(numerator)
                    denominators.append(count)
                    details.append(dict(contrast=f'{treatment} - {control}', group=group, metric=field, targets=int(selection.sum())))
        if filename == 'gt_recovery.csv':
            recovery_data, recovery_initial = data, initial

    annotations = json.loads(args.annotations.read_text())
    by_image = {}
    for ann in annotations['annotations']:
        if not ann.get('iscrowd', 0):
            by_image.setdefault(ann['image_id'], []).append(ann)
    target_index = {int(aid): idx for idx, aid in enumerate(recovery_initial[:,0])}
    pair_a, pair_b, pair_image = [], [], []
    for ii, iid in enumerate(image_ids):
        anns = by_image.get(int(iid), [])
        for j, a in enumerate(anns):
            for b in anns[j+1:]:
                if a['category_id'] == b['category_id'] and bbox_iou(a['bbox'], b['bbox']) > .05:
                    assert a['id'] in target_index and b['id'] in target_index
                    pair_a.append(target_index[a['id']]); pair_b.append(target_index[b['id']]); pair_image.append(ii)
    pair_a, pair_b, pair_image = map(np.array, (pair_a, pair_b, pair_image))
    assert len(pair_a) == int(summary[0]['pairs'])
    pair_values = {arm:(values[pair_a,0]*values[pair_b,0]) for arm,values in recovery_data.items()}
    for arm, values in pair_values.items():
        assert np.isclose(values.mean(), float(summary_by_arm[arm]['pair_recovery75']), atol=1e-12, rtol=0)
    averaged = {group:pair_values['initial'] if group == 'initial' else np.mean([pair_values[f'{group}_s{seed}'] for seed in range(3)], axis=0) for group in GROUPS}
    count = np.bincount(pair_image, minlength=len(image_ids))
    for treatment, control in CONTRASTS:
        numerators.append(np.bincount(pair_image, weights=averaged[treatment]-averaged[control], minlength=len(image_ids)))
        denominators.append(count)
        details.append(dict(contrast=f'{treatment} - {control}', group='all_same_class_adjacent_pairs', metric='pair_recovery75', targets=len(pair_a)))

    num, den = np.stack(numerators, axis=1), np.stack(denominators, axis=1)
    rng = np.random.default_rng(20260911)
    draws = rng.multinomial(len(image_ids), np.full(len(image_ids), 1/len(image_ids)), size=2000).astype(np.float64)
    boot_den = draws@den
    assert np.all(boot_den > 0)
    boot = (draws@num)/boot_den
    means, ci = 100*num.sum(0)/den.sum(0), np.quantile(boot, [.025,.975], axis=0)*100
    for j, row in enumerate(details):
        row.update(mean_pp=float(means[j]), ci_low_pp=float(ci[0,j]), ci_high_pp=float(ci[1,j]))
    result = dict(status='COMPLETE', source_complete=complete,
                  provenance={name:sha(folder/name) for name in ['summary.csv','protocol.json','gt_recovery.csv','spatial.csv']},
                  annotations_sha256=sha(args.annotations), script_sha256=sha(__file__),
                  bootstrap='2,000 paired image-cluster multinomial draws over all 5,000 images; ratios recomputed per draw; paired three-seed means fixed. Pointwise exploratory 95% percentile intervals; no multiplicity adjustment. No AP confidence interval calculated.',
                  task_metrics=tables, spatial_metrics=spatial_tables, contrasts=details,
                  scope='Frozen network with learned coefficient residuals, original candidates fixed, zero masks may be removed after refinement. Official COCO AP; instance micro-recall at mask IoU .75 and maxDets=100 per category is not precision-matched recall. Spatial results condition on fixed bbox-matched GT. Reused val2017 is exploratory, not a pristine confirmatory test.')
    (folder/'PAIRED_ANALYSIS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for table in tables:
        print(json.dumps(table))
    for row in details:
        if row['group'] in ['high','all_same_class_adjacent_pairs']:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
