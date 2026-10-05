"""Offline paired summaries; no inference, fitting, or threshold selection."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'diagnostics/structure_main300_20260911'
REPORT = ROOT.parent.parent / 'refine-logs/coco-structure'


def read(name):
    with (SOURCE / name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truth(value):
    assert value in ('True', 'False'), value
    return value == 'True'


def main():
    receipt = json.loads((SOURCE / 'COMPLETE.json').read_text())
    summary = json.loads((SOURCE / 'STRUCTURE_SUMMARY.json').read_text())
    for name, expected in {**receipt['hashes'], **summary['files']}.items():
        assert sha(SOURCE / name) == expected, name
    ids = json.loads((SOURCE / 'protocol.json').read_text())['images']
    assert len(ids) == len(set(ids)) == 300
    lookup = {iid: k for k, iid in enumerate(ids)}
    weights = np.random.default_rng(20260911).multinomial(
        len(ids), np.full(len(ids), 1 / len(ids)), size=2000)

    def estimate(rows, value):
        sums = np.zeros(len(ids))
        counts = np.zeros(len(ids))
        for row in rows:
            k = lookup[int(row['image_id'])]
            sums[k] += value(row)
            counts[k] += 1
        assert counts.sum() > 0
        denominator = weights @ counts
        boot = (weights @ sums)[denominator > 0] / denominator[denominator > 0]
        return dict(mean=float(sums.sum() / counts.sum()),
                    ci_low=float(np.quantile(boot, .025)),
                    ci_high=float(np.quantile(boot, .975)),
                    n=len(rows), images=int((counts > 0).sum()),
                    valid_bootstrap=len(boot))

    instances = read('instances_same_iou.csv')
    pairs = read('pairs.csv')
    points = read('feature_pair_means.csv')
    assert len(instances) == 3635 and len(pairs) == 1174
    witness = read('witness.csv')
    assert {int(r['image_id']) for r in witness} == set(ids)
    assert all(truth(r['raw_index_replay']) and float(r['cache_max_error']) == 0
               for r in witness)
    assert all(truth(r['bbox50']) == truth(r['official_bbox50_matched'])
               and truth(r['segm75']) == truth(r['official_mask75_recovered'])
               for r in instances), 'New official matches differ from cached attribution/recovery'
    byid = {int(r['annotation_id']): r for r in instances}
    stage = {}
    for group in ['all', 'high', 'low']:
        rr = [r for r in instances if group == 'all' or
              (float(r['ici_same']) > .5 + 1e-10) == (group == 'high')]
        stage[group] = {
            'score_to_nms_loss75': estimate(rr, lambda r:
                float(truth(r['score_available75'])) - float(truth(r['nms_available75']))),
            'bbox75_and_mask_fail75_fraction_all_gt': estimate(rr, lambda r:
                float(truth(r['bbox75']) and not truth(r['segm75']))),
        }
    pair_stats = {}
    for group in ['all', 'high', 'low']:
        rr = [r for r in pairs if group == 'all' or
              (float(r['ici_max']) > .5 + 1e-10) == (group == 'high')]
        item = {}
        for key in ['raw_geometry_two_distinct', 'score_two_distinct',
                    'nms_two_distinct', 'eval100_two_distinct',
                    'official_bbox50_both', 'official_mask75_both']:
            item[key] = estimate(rr, lambda r, key=key: float(truth(r[key])))
        for key in ['bbox75', 'segm75']:
            item['official_' + key + '_both'] = estimate(rr, lambda r, key=key:
                float(truth(byid[int(r['annotation_a'])][key]) and
                      truth(byid[int(r['annotation_b'])][key])))
        pair_stats[group] = item

    pp = [r for r in points if r['pair_type'] == 'same_adjacent' and
          r['task'] == 'own_neighbor' and r['layer'] == 'proto' and truth(r['high'])]
    cohorts = {}
    for name, rows in [('all_eligible', pp),
                       ('a_bbox50_matched', [r for r in pp if r['actual_a_auc']]),
                       ('both_bbox50_matched', [r for r in pp if r['actual_pair_difference_auc']])]:
        metrics = ['auc', 'coordinate_auc', 'shuffled_train_auc']
        if name != 'all_eligible':
            metrics += ['actual_a_auc']
        if name == 'both_bbox50_matched':
            metrics += ['actual_pair_difference_auc', 'coefficient_cosine']
        values = {metric: estimate(rows, lambda r, metric=metric: float(r[metric]))
                  for metric in metrics}
        contrasts = {'oracle_minus_coordinate': ('auc', 'coordinate_auc')}
        if name != 'all_eligible':
            contrasts['oracle_minus_actual_a'] = ('auc', 'actual_a_auc')
        if name == 'both_bbox50_matched':
            contrasts.update(pair_difference_minus_actual_a=(
                'actual_pair_difference_auc', 'actual_a_auc'),
                oracle_minus_pair_difference=('auc', 'actual_pair_difference_auc'))
        for contrast, (a, b) in contrasts.items():
            values[contrast] = estimate(rows, lambda r, a=a, b=b: float(r[a]) - float(r[b]))
        cohorts[name] = values
    output = dict(
        scope='Frozen pretrained model. Dense-enriched reused 300 val images. Pointwise 2000 image-cluster bootstrap intervals; no multiple-comparison correction. Three projection seeds averaged within pair, not model replicates.',
        stages=stage, pairs=pair_stats, proto_high_same_adjacent=cohorts,
        integrity=dict(main_receipt_hashes_verified=True,
                       summary_hashes_verified=True, exact_replay_images=len(witness),
                       official_matches_equal_prior=True,
                       instances=len(instances), pairs=len(pairs)),
        source_sha256={name: sha(SOURCE / name) for name in [
            'instances_same_iou.csv', 'pairs.csv', 'feature_pair_means.csv']},
        reporter_sha256=sha(Path(__file__)))
    (SOURCE / 'PAIRED_STRUCTURE_ANALYSIS.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(stage=stage, pair_high=pair_stats['high'], proto=cohorts), indent=2))


if __name__ == '__main__':
    main()
