"""Frozen image-macro evidence and investment rules for the exposed 256-pair panel."""
from collections import defaultdict
import numpy as np


def summarize(rows, cfg, scientific):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['image_id']].append(r)
    ids = sorted(grouped)
    if not rows or len({r['output_detection_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate panel')
    indices = np.random.default_rng(cfg['statistics']['seed']).integers(0, len(ids), size=(cfg['statistics']['bootstrap_replicates'], len(ids)))
    def macro(fn):
        return np.array([np.mean([fn(r) for r in grouped[i]]) for i in ids])
    def receipt(v):
        return dict(mean=float(v.mean()), ci95=np.quantile(v[indices].mean(axis=1), [.025,.975], method='linear').tolist())
    arms = [a for a in ('N','U','G','C','R') if all(a in r['arms'] for r in rows)]
    values = {a:macro(lambda r:r['arms'][a]['original_iou']) for a in arms}
    metrics = {a:dict(original_iou=receipt(v), delta_vs_N=receipt(v-values['N']), repaired=sum(r['arms'][a]['original_iou']>=.75 for r in rows)) for a,v in values.items()}
    comparisons = {'R_minus_'+a:receipt(values['R']-values[a]) for a in arms if a!='R'}
    denom = values['C']-values['U']; gain = values['R']-values['U']
    draw_d = denom[indices].mean(axis=1); draw_g = gain[indices].mean(axis=1)
    reference_valid = bool(denom.mean()>0 and (draw_d>0).all())
    q = dict(value=float(gain.mean()/denom.mean()), ci95=np.quantile(draw_g/draw_d,[.025,.975],method='linear').tolist()) if reference_valid else None
    ccounts = np.array([sum(r['arms']['C']['original_iou']>=.75 for r in grouped[i]) for i in ids])
    retained = np.array([sum(r['arms']['C']['original_iou']>=.75 and r['arms']['R']['original_iou']>=.75 for r in grouped[i]) for i in ids])
    draw_c = ccounts[indices].sum(axis=1)
    capture_valid = bool(ccounts.sum()>0 and (draw_c>0).all())
    capture = dict(numerator=int(retained.sum()), denominator=int(ccounts.sum()), value=float(retained.sum()/ccounts.sum()), ci95=np.quantile(retained[indices].sum(axis=1)/draw_c,[.025,.975],method='linear').tolist()) if capture_valid else None
    numerical = all(r['numerical_valid'] for r in rows)
    complete = len(rows)==cfg['panel']['pairs'] and len(ids)==cfg['panel']['images']
    g=cfg['gates']
    if not scientific:decision='ENGINEERING_ONLY'
    elif not complete or not numerical:decision='INCONCLUSIVE_NUMERICAL'
    elif not reference_valid or not capture_valid:decision='INSUFFICIENT_REFERENCE'
    elif q['value']>=g['amplitude_capture'] and capture['value']>=g['amplitude_capture'] and comparisons['R_minus_U']['mean']>=g['min_gain_over_U'] and comparisons['R_minus_U']['ci95'][0]>0:decision='AMPLITUDE_PRIORITY'
    elif q['ci95'][1]<g['direction_upper_capture'] and capture['ci95'][1]<g['direction_upper_capture']:decision='DIRECTION_PRIORITY'
    else:decision='MIXED_DIRECTION_AMPLITUDE'
    alphas=np.array([r['arms']['R']['alpha'] for r in rows])
    return dict(pairs=len(rows),images=len(ids),unit='original_mask_IoU_0_to_1_not_AP',reduction='instance_mean_within_image_then_equal_image_mean',
        metrics=metrics,comparisons=comparisons,opportunity_capture=q,C_witness_capture=capture,numerical_valid=numerical,reference_valid=reference_valid and capture_valid,
        denominator_complete=complete,decision=decision,alpha=dict(min=float(alphas.min()),median=float(np.median(alphas)),max=float(alphas.max()),zero=int((alphas==0).sum()),one=int((alphas==1).sum()),at_max=int((alphas==cfg['ray']['alpha_max']).sum())),
        C_witness_lost=[r['output_detection_id'] for r in rows if r['arms']['C']['original_iou']>=.75 and r['arms']['R']['original_iou']<.75],
        new_training_steps=0,base_model_forward_calls=0,GT_assisted_selection=True,deployment_claim=False,reused_development_panel=True)
