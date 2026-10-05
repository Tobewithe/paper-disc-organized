"""Exact identities and labels from the already-evaluated COCO mask banks."""
import csv
import json
from pathlib import Path
import numpy as np
from calibrator import bank_features


def load_bank(root):
    root = Path(root)
    rows = list(csv.DictReader((root / 'candidate_records.csv').open(newline='', encoding='utf-8')))
    keys = [(int(r['image_id']), int(r['candidate_index'])) for r in rows]
    lookup = {key: i for i, key in enumerate(keys)}
    assert len(keys) == len(lookup)
    x = bank_features(rows)
    area = np.asarray([float(r['baseline_area']) for r in rows])
    trial_area = np.asarray([float(r['smooth_area']) for r in rows])
    assert np.isfinite(x).all()
    assert (trial_area <= area + 1e-6).all()
    eligible = (area > 0) & (trial_area > 0)
    matched = []
    for row in csv.DictReader((root / 'instance_records.csv').open(newline='', encoding='utf-8')):
        if row['variant'] != 'smooth_gated':
            continue
        key = (int(row['image_id']), int(row['candidate_index']))
        i = lookup[key]
        g = float(row['gt_area']); c = float(row['baseline_recall']); p = float(row['baseline_purity'])
        j = float(row['baseline_iou']); a = float(row['removed_tp']); b = float(row['removed_fp'])
        assert g > 0 and area[i] > 0 and 0 <= p <= 1 + 1e-8 and 0 <= c <= 1 + 1e-8
        assert abs(c * g - p * area[i]) < 1e-5
        assert abs(j - p * area[i] / (g + (1 - p) * area[i])) < 1e-8
        assert a >= -1e-5 and b >= -1e-5
        assert abs(a + b - (area[i] - trial_area[i])) < 1e-5
        if not eligible[i]:
            a = b = 0.
        j1 = float(row['iou']) if eligible[i] else j
        c1 = c - a / g
        assert abs(j1 - (c * g - a) / (g + (1 - p) * area[i] - b)) < 1e-8
        matched.append({'index': i, 'image_id': key[0], 'candidate_index': key[1],
                        'annotation_id': int(row['annotation_id']), 'category_id': int(row['category_id']),
                        'box_iou': float(row['box_iou']), 'gt_area': g, 'baseline_iou': j,
                        'baseline_coverage': c, 'baseline_purity': p, 'trial_iou': j1,
                        'trial_coverage': c1, 'removed_tp': a, 'removed_fp': b,
                        'removed_purity': a / (a + b) if a + b > 1e-9 else 0.,
                        'coverage_cost': a / g, 'fp_benefit': b / g,
                        'target': bool(float(row['box_iou']) >= .75 and c >= .95 and j < .75)})
    assert len({(r['image_id'], r['candidate_index']) for r in matched}) == len(matched)
    assert len({(r['image_id'], r['annotation_id']) for r in matched}) == len(matched)
    return {'rows': rows, 'keys': keys, 'lookup': lookup, 'features': x, 'area': area,
            'eligible': eligible, 'matched': matched, 'ids': json.loads((root / 'image_ids.json').read_text())}


def subset_outcomes(bank, use, image_ids):
    selected = [r for r in bank['matched'] if r['image_id'] in image_ids]
    stats = {'matched': len(selected), 'targets': 0, 'successes': 0, 'target_repairs': 0,
             'all_repairs': 0, 'damages': 0, 'iou_gain_sum': 0., 'coverage_loss_sum': 0.}
    for r in selected:
        after = r['trial_iou'] if use[r['index']] else r['baseline_iou']
        repaired = r['baseline_iou'] < .75 <= after
        damaged = after < .75 <= r['baseline_iou']
        stats['targets'] += int(r['target']); stats['successes'] += int(r['baseline_iou'] >= .75)
        stats['target_repairs'] += int(r['target'] and repaired)
        stats['all_repairs'] += int(repaired); stats['damages'] += int(damaged)
        stats['iou_gain_sum'] += after - r['baseline_iou']
        stats['coverage_loss_sum'] += r['coverage_cost'] if use[r['index']] else 0.
    stats['net_repairs'] = stats['all_repairs'] - stats['damages']
    return stats
