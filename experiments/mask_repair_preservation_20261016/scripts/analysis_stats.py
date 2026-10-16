"""Image-macro paired statistics; no re-selection of targets or comparators."""
from collections import defaultdict
import numpy as np


def summarize(rows, cfg, scientific=True):
    if not rows or len({r['output_detection_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicated panel')
    ids = sorted({r['image_id'] for r in rows})
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['image_id']].append(r)
    sampling = np.random.default_rng(cfg['statistics']['seed']).integers(
        0, len(ids), size=(cfg['statistics']['bootstrap_replicates'], len(ids)))

    def macro(fn):
        return np.asarray([np.mean([fn(r) for r in grouped[i]]) for i in ids], dtype=np.float64)

    def receipt(v):
        draws = v[sampling].mean(axis=1)
        return dict(mean=float(v.mean()), ci95=[float(x) for x in np.quantile(draws, [.025, .975], method='linear')])

    arms = tuple(a for a in ('N', 'C', 'E', 'EK', 'E_equal', 'R_G', 'R_S1') if all(a in r['arms'] for r in rows))
    if scientific and not {'R_G', 'R_S1'}.issubset(arms):
        raise ValueError('missing preselected existing references')
    ious = {a: macro(lambda r: r['arms'][a]['original_iou']) for a in arms}
    metrics = {a: dict(original_iou=receipt(ious[a]), delta_vs_native=receipt(ious[a]-ious['N']),
                       repaired_instances=sum(r['arms'][a]['original_iou'] >= .75 for r in rows)) for a in arms}
    comparisons = {name: receipt(ious[a]-ious[b]) for name, a, b in (
        ('EK_minus_C', 'EK', 'C'), ('EK_minus_E_equal', 'EK', 'E_equal'),
        ('EK_minus_E', 'EK', 'E'), ('EK_minus_R_G', 'EK', 'R_G'), ('EK_minus_R_S1', 'EK', 'R_S1')) if a in ious and b in ious}
    damage = {a: macro(lambda r: r['arms'][a]['tp_to_fn']/max(1, r['native_tp'])) for a in ('C', 'E', 'EK', 'E_equal')}
    decrease = receipt(damage['E_equal']-damage['EK'])
    denom = float(damage['E_equal'].mean())
    relative = decrease['mean']/denom if denom > 0 else None
    control_success = {r['output_detection_id'] for r in rows if r['arms']['C']['original_iou'] >= .75}
    protected_success = {r['output_detection_id'] for r in rows if r['arms']['EK']['original_iou'] >= .75}
    retained = len(control_success & protected_success)
    numerical = all(r['numerical_valid'] and all(r['arms'][a]['solver']['solver_status']=='converged'
        for a in ('C','E','EK')) for r in rows)
    full = len(rows)==cfg['panel']['pairs'] and len(ids)==cfg['panel']['images']
    g = cfg['gates']
    point = dict(utility=comparisons['EK_minus_C']['mean']>=g['min_increment_over_C'],
        protection=relative is not None and relative>=g['min_relative_tp_loss_reduction'],
        equal_iou=comparisons['EK_minus_E_equal']['mean']>=-g['max_iou_loss_to_equal'],
        retention=bool(control_success) and retained/len(control_success)>=g['min_C_witness_retention'])
    intervals = dict(utility=comparisons['EK_minus_C']['ci95'][0]>0, protection=decrease['ci95'][0]>0)
    if not scientific:
        decision='ENGINEERING_ONLY'
    elif not full or not numerical:
        decision='INCONCLUSIVE_NUMERICAL'
    elif all(point.values()) and all(intervals.values()):
        decision='FINITE_MECHANISM_SUPPORTED'
    elif not all(point.values()):
        decision='CONFIGURATION_REJECTED'
    else:
        decision='INSUFFICIENT_PRECISION'
    return dict(pairs=len(rows), images=len(ids), unit='IoU_0_to_1_not_AP',
        reduction='instance_mean_within_image_then_equal_image_mean', metrics=metrics, comparisons=comparisons,
        tp_loss={a:receipt(v) for a,v in damage.items()}, protection_decrease=decrease,
        relative_tp_loss_reduction=relative, control_successes=len(control_success), retained_C_successes=retained,
        added_vs_C=sorted(protected_success-control_success), lost_vs_C=sorted(control_success-protected_success),
        native_tp_zero_instances=sum(r['native_tp']==0 for r in rows),
        point_gates=point, interval_gates=intervals, numerical_valid=numerical,
        denominator_complete=full, decision=decision,
        deployment_claim=False, reused_development_panel=True, new_training_steps=0, base_model_forward_calls=0)
