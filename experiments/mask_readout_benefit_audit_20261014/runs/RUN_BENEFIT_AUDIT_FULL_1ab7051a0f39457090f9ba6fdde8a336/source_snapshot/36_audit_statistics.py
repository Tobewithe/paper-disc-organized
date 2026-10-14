"""Frozen, model-independent statistics for the benefit audit."""
from collections import Counter
import hashlib
import math
import numpy as np

FEATURES = ('preserved_foreground', 'native_uncertainty', 'same_class_neighbor_iou')
STAGES = ('soft_roi', 'binary_roi', 'binary_proto_full', 'binary_input_full', 'binary_original')
STAGE_CLASSES = ('soft_roi_already_down', 'proto_binary_reversal', 'full_support_reversal',
                 'proto_to_input_reversal', 'input_to_original_reversal', 'matching_despite_fixed_mask75')


def counted(tp, fp, fn, tn):
    if any(type(v) is not int or v < 0 for v in (tp, fp, fn, tn)):
        raise ValueError('nonnegative integer counts required')
    union = tp + fp + fn
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, pixels=tp+fp+fn+tn,
                empty_union=union == 0, iou=tp/union if union else 1.0)


def failure_stage(row):
    if not row['task_damaged']:
        return None
    stages = row['stages']
    if stages['binary_original']['after']['iou'] >= .75:
        return 'matching_despite_fixed_mask75'
    for name, label in zip(STAGES, STAGE_CLASSES[:5]):
        delta = stages[name]['delta']
        if delta is not None and delta < (-1e-6 if name == 'soft_roi' else -1e-12):
            return label
    raise ValueError('lost successful fixed detection without a negative endpoint')


def weighted_auc(scores, positive, weights=None):
    scores, positive = np.asarray(scores, dtype=float), np.asarray(positive, dtype=bool)
    weights = np.ones(len(scores)) if weights is None else np.asarray(weights, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or np.any(weights < 0):
        raise ValueError('invalid AUC input')
    _, groups = np.unique(scores, return_inverse=True)
    yes = np.bincount(groups, weights=weights*positive)
    no = np.bincount(groups, weights=weights*(~positive))
    denominator = yes.sum()*no.sum()
    if denominator == 0:
        return None
    return float(np.dot(yes, np.cumsum(no)-.5*no)/denominator)


def signal_summary(rows, image_ids):
    dev = [r for r in rows if r['role'] == 'dev_bbox75']
    nonzero = [r for r in dev if abs(r['stages']['binary_original']['delta']) > 1e-12]
    positive = np.asarray([r['stages']['binary_original']['delta'] > 0 for r in nonzero])
    ids = sorted(image_ids)
    index = {v: i for i, v in enumerate(ids)}
    groups = np.array([index[r['image_id']] for r in nonzero], dtype=int)
    halves = sorted(ids, key=lambda i: hashlib.sha256(f'benefit-audit-v1:{i}'.encode()).hexdigest())
    first = set(halves[:len(halves)//2])
    left = np.array([r['image_id'] in first for r in nonzero], dtype=bool)
    positives = sum(bool(x) for x in positive)
    negatives = len(nonzero)-positives
    positive_images = len({r['image_id'] for r, y in zip(nonzero, positive) if y})
    negative_images = len({r['image_id'] for r, y in zip(nonzero, positive) if not y})
    adequate = positives >= 50 and negatives >= 50 and min(positive_images, negative_images) >= 30
    rng = np.random.default_rng(1729)
    multiplicities = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
                              for _ in range(2000)])
    output = {}
    for feature in FEATURES:
        scores = np.array([r['signals'][feature] for r in nonzero], dtype=float)
        point = weighted_auc(scores, positive) if len(scores) else None
        halves_auc = [weighted_auc(scores[m], positive[m]) if m.any() else None for m in (left, ~left)]
        draws = [weighted_auc(scores, positive, w[groups]) for w in multiplicities] if len(scores) else []
        valid = [x for x in draws if x is not None]
        ci = [float(v) for v in np.quantile(valid, [.05/6, 1-.05/6], method='linear')] if valid else None
        passed = (adequate and point is not None and point >= .60 and ci is not None and ci[0] > .50
                  and all(v is not None and v >= .55 for v in halves_auc) and len(valid) == 2000)
        output[feature] = dict(auc=point, familywise95_interval=ci, half_auc=halves_auc,
                               valid_bootstrap_draws=len(valid), priority_gate=bool(passed))
    return dict(total_pairs=len(dev), gain_pairs=positives, harm_pairs=negatives,
                tied_pairs=len(dev)-len(nonzero), gain_images=positive_images, harm_images=negative_images,
                adequate_counts=adequate, features=output,
                bootstrap=dict(resamples=2000, seed=1729, clusters=len(ids), quantiles=[.05/6, 1-.05/6],
                               method='image_cluster_percentile_linear_Bonferroni_three_fixed_directions'),
                halves=dict(first=halves[:len(halves)//2], second=halves[len(halves)//2:]))


def summarize(rows, dev_ids):
    damage = [r for r in rows if r['role'] == 'success_segm75' and r['task_damaged']]
    signatures = Counter(failure_stage(r) for r in damage)
    means = {}
    for role in ('dev_bbox75', 'success_segm75'):
        selected = [r for r in rows if r['role'] == role]
        means[role] = {name: {'valid_pairs': sum(r['stages'][name]['delta'] is not None for r in selected),
            'pair_mean_delta': float(np.mean([r['stages'][name]['delta'] for r in selected
                                              if r['stages'][name]['delta'] is not None]))
                               if any(r['stages'][name]['delta'] is not None for r in selected) else None}
                       for name in STAGES}
    return dict(pair_roles=dict(Counter(r['role'] for r in rows)), damaged_success_GT=len(damage),
                damage_stage_signatures={k: signatures[k] for k in STAGE_CLASSES},
                stage_pair_means=means, signals=signal_summary(rows, dev_ids),
                empty_ROI_pairs=sum(r['empty_roi'] for r in rows),
                empty_native_foreground_pairs=sum(r['signals']['native_foreground_pixels'] == 0 for r in rows))


def decide(summary):
    signals = summary['signals']
    passing = [f for f in FEATURES if signals['features'][f]['priority_gate']]
    if passing:
        return dict(status='supported', direction='selective_application', selected_signal=passing[0],
                    next_action='Freeze one zero-training selective-application baseline protocol with full COCO matching and original S2 gates; compare the existing RCMC selection principle; no training yet')
    counts = summary['damage_stage_signatures']
    largest = max(STAGE_CLASSES, key=lambda name: counts[name])
    if largest == 'matching_despite_fixed_mask75':
        focus = 'preserve_valid_detection_competition'
    elif largest in STAGE_CLASSES[:2]:
        focus = 'proto_mask_fidelity_and_threshold_quality'
    else:
        focus = 'native_output_geometry_consistent_correction'
    return dict(status='not_supported' if signals['adequate_counts'] else 'insufficient_precision',
                direction=focus, selected_signal=None, dominant_damage_signature=largest,
                next_action='Freeze one single-variable correction-design protocol addressing this observed stage; no additional gate-feature search or training in this audit')
