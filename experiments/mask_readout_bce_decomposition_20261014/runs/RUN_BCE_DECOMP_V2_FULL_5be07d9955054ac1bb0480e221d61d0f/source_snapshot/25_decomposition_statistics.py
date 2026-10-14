"""Fixed ROI accounting; never changes the model, targets, losses or thresholds."""
import math
from collections import Counter

GROUPS = ('TN_to_TN', 'TN_to_FP', 'FP_to_TN', 'FP_to_FP',
          'FN_to_FN', 'FN_to_TP', 'TP_to_FN', 'TP_to_TP')
STABLE = ('TN_to_TN', 'FP_to_FP', 'FN_to_FN', 'TP_to_TP')
HARMFUL = ('TN_to_FP', 'TP_to_FN')
CORRECTED = ('FP_to_TN', 'FN_to_TP')


def count_endpoints(groups):
    n = {k: groups[k]['pixels'] for k in GROUPS}
    return ({'tp': n['TP_to_TP']+n['TP_to_FN'], 'fn': n['FN_to_FN']+n['FN_to_TP'],
             'fp': n['FP_to_FP']+n['FP_to_TN'], 'tn': n['TN_to_TN']+n['TN_to_FP']},
            {'tp': n['TP_to_TP']+n['FN_to_TP'], 'fn': n['FN_to_FN']+n['TP_to_FN'],
             'fp': n['FP_to_FP']+n['TN_to_FP'], 'tn': n['TN_to_TN']+n['FP_to_TN']})


def iou_terms(before, after):
    gt = before['tp']+before['fn']
    if gt != after['tp']+after['fn']:
        raise ValueError('GT pixel count changed')
    u0, u1 = gt+before['fp'], gt+after['fp']
    j0, j1 = before['tp']/u0 if u0 else 1., after['tp']/u1 if u1 else 1.
    if u0 and u1:
        fg = (after['tp']-before['tp'])*(.5/u0+.5/u1)
        bg = .5*(before['tp']+after['tp'])*(1/u1-1/u0)
    else:
        if gt:
            raise ValueError('nonempty GT with empty union')
        fg, bg = 0., j1-j0
    if abs(fg+bg-(j1-j0)) > 1e-12:
        raise ValueError('IoU decomposition failed')
    harms = max(-fg, 0.), max(-bg, 0.)
    dominant = 'foreground' if harms[0]-harms[1] > 1e-12 else 'background' if harms[1]-harms[0] > 1e-12 else 'tie'
    return {'foreground_TP_term': fg, 'background_FP_term': bg, 'delta_iou': j1-j0,
            'empty_union_boundary': not (u0 and u1), 'dominant_negative_term': dominant}


def derive(groups):
    before, after = count_endpoints(groups)
    sums = {field: math.fsum(groups[g][field] for g in GROUPS)
            for field in ('zero_loss', 'B_loss', 'delta_loss', 'decrease_credit', 'increase_debit')}
    stable = math.fsum(groups[g]['decrease_credit'] for g in STABLE)
    corrected = math.fsum(groups[g]['decrease_credit'] for g in CORRECTED)
    fg = {field: math.fsum(groups[g][field] for g in GROUPS[4:]) for field in sums}
    bg = {field: math.fsum(groups[g][field] for g in GROUPS[:4]) for field in sums}
    share = stable/sums['decrease_credit'] if sums['decrease_credit'] else None
    return {'totals': sums, 'foreground': fg, 'background': bg, 'stable_decrease_credit': stable,
            'corrected_decrease_credit': corrected, 'stable_credit_share': share,
            'stable_credit_majority': share is not None and share > .5,
            'counts_zero': before, 'counts_B': after, 'iou_decomposition': iou_terms(before, after)}


def partition(zero_logits, B_logits, targets, roi, zero_pixel_bce, B_pixel_bce):
    import torch
    tensors = (B_logits, targets, roi, zero_pixel_bce, B_pixel_bce)
    if any(x.shape != zero_logits.shape for x in tensors) or roi.dtype != torch.bool:
        raise ValueError('partition grid mismatch')
    if not ((targets == 0) | (targets == 1)).all().item():
        raise ValueError('nonbinary GT')
    area = roi.sum((1, 2))
    if not (area > 0).all().item():
        raise ValueError('empty ROI')
    code = 4*targets.long()+2*(zero_logits > 0).long()+(B_logits > 0).long()
    a, b = zero_pixel_bce.double(), B_pixel_bce.double()
    gain, cost = (a-b).clamp_min(0), (b-a).clamp_min(0)
    by_group = []
    for index in range(8):
        mask = (code == index) & roi
        values = torch.stack((mask.sum((1, 2)).double(), (a*mask).sum((1, 2))/area,
            (b*mask).sum((1, 2))/area, ((b-a)*mask).sum((1, 2))/area,
            (gain*mask).sum((1, 2))/area, (cost*mask).sum((1, 2))/area), 1)
        by_group.append(values.cpu().tolist())
    result = []
    for row in range(len(area)):
        groups = {}
        for name, values in zip(GROUPS, by_group):
            n, loss0, loss1, delta, credit, debit = values[row]
            groups[name] = {'pixels': int(n), 'zero_loss': loss0, 'B_loss': loss1,
                            'delta_loss': delta, 'decrease_credit': credit, 'increase_debit': debit}
        if sum(g['pixels'] for g in groups.values()) != int(area[row]):
            raise ValueError('partition not exhaustive')
        for g in groups.values():
            if abs(g['delta_loss']-(g['increase_debit']-g['decrease_credit'])) > 1e-12:
                raise ValueError('signed credit closure failed')
        result.append({'groups': groups, **derive(groups)})
    return result


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError('empty target population')
    fields = ('zero_loss', 'B_loss', 'delta_loss', 'decrease_credit', 'increase_debit')
    means = {name: {f: math.fsum(r['decomposition'][name][f] for r in rows)/n for f in fields}
             for name in ('totals', 'foreground', 'background')}
    groups = {g: {'pixels': sum(r['decomposition']['groups'][g]['pixels'] for r in rows),
                  **{f: math.fsum(r['decomposition']['groups'][g][f] for r in rows)/n for f in fields}}
              for g in GROUPS}
    majority = sum(r['decomposition']['stable_credit_majority'] for r in rows)
    credits = math.fsum(r['decomposition']['totals']['decrease_credit'] for r in rows)
    stable_credits = math.fsum(r['decomposition']['stable_decrease_credit'] for r in rows)
    return {'target_pairs': n, 'target_images': len({r['image_id'] for r in rows}),
            'stable_credit_majority_pairs': majority, 'stable_credit_majority_fraction': majority/n,
            'equal_pair_mean_stable_credit_share': math.fsum(r['decomposition']['stable_credit_share'] for r in rows)/n,
            'ratio_of_summed_pair_normalized_stable_credit': stable_credits/credits,
            'equal_pair_mean_contributions': means, 'groups': groups,
            'IoU_negative_term_dominance': dict(Counter(r['decomposition']['iou_decomposition']['dominant_negative_term'] for r in rows)),
            'IoU_terms_mean': {k: math.fsum(r['decomposition']['iou_decomposition'][k] for r in rows)/n
                               for k in ('foreground_TP_term', 'background_FP_term', 'delta_iou')},
            'empty_union_boundary_pairs': sum(r['decomposition']['iou_decomposition']['empty_union_boundary'] for r in rows),
            'max_BCE_delta_closure_abs_error': max(r['BCE_delta_closure_abs_error'] for r in rows),
            'unit': 'fixed-pair diagnostic; all loss contributions normalized by original ROI pixels then pair mean'}


def adjudicate(summary):
    supported = summary['stable_credit_majority_fraction'] > .5
    return {'gate_id': 'BCE_STABLE_CONFIDENCE_MAJORITY', 'strict_pair_credit_threshold': .5,
            'strict_population_threshold': .5, 'hypothesis_supported': supported,
            'localization': 'stable_binary_label_confidence_gains_dominate_BCE_decrease' if supported else
                            'stable_confidence_majority_hypothesis_rejected',
            'next_action': 'Preregister a new single-variable S2 pilot replacing only GT BCE with same-ROI soft-IoU supervision; keep preservation, initialization, 2k/3epoch budget and original efficacy/damage gates; do not train in this diagnostic' if supported else
                           'Audit staged native decoding and evaluation support on the retained 1791 non-early-reversal pairs; no loss change',
            'prior_S2_negative_decision_retained': True, 'scope': 'closed arithmetic accounting on selected fixed fit pairs, not causal intervention'}
