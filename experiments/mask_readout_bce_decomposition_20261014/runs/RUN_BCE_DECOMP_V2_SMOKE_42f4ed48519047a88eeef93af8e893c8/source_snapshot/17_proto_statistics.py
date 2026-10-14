"""Small deterministic definitions, independent of GPU and model code."""
from collections import Counter
import math


def direction(delta, tolerance=1e-12):
    if not math.isfinite(delta):
        raise ValueError('non-finite effect')
    return 'down' if delta < -tolerance else 'up' if delta > tolerance else 'tie'


def counted_iou(tp, fp, fn, tn):
    values = (tp, fp, fn, tn)
    if any(type(v) is not int or v < 0 for v in values) or not sum(values):
        raise ValueError('nonempty ROI and nonnegative integer counts required')
    union = tp+fp+fn
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'roi_pixels': sum(values),
            'gt_pixels': tp+fn, 'union_pixels': union, 'empty_union': union == 0,
            'iou': tp/union if union else 1.0}


def tensor_counts(logits, targets, roi):
    import torch
    if logits.shape != targets.shape or logits.shape != roi.shape or roi.dtype != torch.bool:
        raise ValueError('grid or ROI mismatch')
    if not torch.isfinite(logits).all().item() or not ((targets == 0) | (targets == 1)).all().item():
        raise ValueError('nonfinite logits or nonbinary GT')
    p, g = logits > 0, targets == 1
    counts = torch.stack([((p & g) & roi).sum((1, 2)), ((p & ~g) & roi).sum((1, 2)),
                          ((~p & g) & roi).sum((1, 2)), ((~p & ~g) & roi).sum((1, 2))], dim=1)
    return [counted_iou(*row) for row in counts.cpu().tolist()]


def pair_key(row):
    return row['image_id'], row['annotation_id'], row['training_output_id']


def summarize(rows):
    abnormal = [r for r in rows if r['direction'] == 'loss_down_iou_down']
    counts = {name: sum(r['proto_direction'] == name for r in abnormal) for name in ('down', 'tie', 'up')}
    transitions = Counter(r['direction']+'_proto_'+r['proto_direction'] for r in rows)
    return {'supervised_pairs': len(rows), 'known_discrepant_pairs': len(abnormal),
            'discrepant_proto_directions': counts,
            'early_reversal_fraction': counts['down']/len(abnormal) if abnormal else None,
            'all_stage_direction_counts': dict(transitions),
            'empty_union_counts': {arm: sum(r[arm]['empty_union'] for r in rows) for arm in ('zero_proto', 'B_proto')},
            'empty_GT_in_ROI_pairs': sum(r['zero_proto']['gt_pixels'] == 0 for r in rows),
            'unit': 'fixed supervised detection-GT pairs; IoU fraction_0_to_1',
            'mean_delta_proto_iou': math.fsum(r['delta_proto_iou'] for r in rows)/len(rows) if rows else None}


def adjudicate(summary):
    if not summary['known_discrepant_pairs']:
        raise ValueError('no discrepancy population')
    early = summary['early_reversal_fraction'] > .5
    return {'majority_threshold_strict': .5, 'majority_early_reversal': early,
            'localization': 'majority_already_reverses_on_same_proto_ROI' if early else
                            'majority_first_negative_binary_change_after_proto_ROI',
            'next_action': 'Zero-training foreground/background BCE and TP/FP/FN decomposition on the fixed early-reversal pairs' if early else
                           'Zero-training staged native upsampling/cropping/original-image-support audit on the remaining pairs',
            'prior_S2_negative_decision_retained': True,
            'scope': 'observed stage boundary on fixed fit population; not causal attribution to a decoder defect'}
