"""Frozen pair/image reductions, independent of solver and decoder."""
import numpy as np


def analyze(rows, cfg, scope, output_indices=None):
    ids = sorted({r['image_id'] for r in rows})
    groups = [[r for r in rows if r['image_id']==i] for i in ids]
    rng = np.random.Generator(np.random.PCG64(cfg['bootstrap_seed']))
    samples = rng.integers(len(ids), size=(cfg['bootstrap_replicates'],len(ids)))
    if output_indices is not None:
        np.savez_compressed(output_indices, image_ids=np.array(ids), samples=samples)
    def interval(values):
        a = np.asarray(values,dtype=float)
        return {'mean':float(a.mean()),'ci95':np.quantile(a[samples].mean(1),[.025,.975],method='linear').tolist()}
    def by_image(fn):
        return [np.mean([fn(r) for r in g]) for g in groups]
    def stress(r, center, arm=None):
        return np.mean([p[center]['iou'] for p in r['pressure'] if arm is None or p['arm']==arm])
    main = interval(by_image(lambda r:stress(r,'W')-stress(r,'C')))
    arms = {a:interval(by_image(lambda r:stress(r,'W',a)-stress(r,'C',a))) for a in ['S','R','M']}
    valid_margin = all(r['margin_C'] is not None and r['margin_W'] is not None for r in rows)
    margin = interval(by_image(lambda r:r['margin_W']-r['margin_C'])) if valid_margin else None
    numerator_C, numerator_W, denominator = [], [], []
    for g in groups:
        eligible = [r for r in g if r['endpoints']['C']['iou']>=.75]
        denominator.append(len(eligible))
        for center, target in [('C',numerator_C),('W',numerator_W)]:
            target.append(sum(np.mean([p[center]['iou']>=.75 for p in r['pressure']]) for r in eligible))
    den = np.asarray(denominator,dtype=float); nc=np.array(numerator_C); nw=np.array(numerator_W)
    witness = None
    if den.sum()>0:
        bd = den[samples].sum(1)
        gains = (nw-nc)[samples].sum(1)/np.where(bd>0,bd,np.nan)
        if not np.isfinite(gains).all():
            raise ValueError('bootstrap sample without any C witness')
        witness = {'C_witnesses':int(den.sum()),'C_retention':float(nc.sum()/den.sum()),'W_retention':float(nw.sum()/den.sum()),
            'gain':float((nw-nc).sum()/den.sum()),'ci95':np.quantile(gains,[.025,.975],method='linear').tolist()}
    refs={}
    for center in ['C','W']:
        for ref in ['N','G_old']:
            if all(r['endpoints'].get(ref) is not None for r in rows):
                refs[f'{center}-{ref}']=interval(by_image(lambda r:r['endpoints'][center]['iou']-r['endpoints'][ref]['iou']))
                refs[f'{center}_stress-{ref}']=interval(by_image(lambda r:stress(r,center)-r['endpoints'][ref]['iou']))
            else:
                refs[f'{center}-{ref}']=None
    valid=sum(r['accepted'] for r in rows)
    numerical=valid/len(rows)>=cfg['gates']['min_valid_fraction'] and all(r['proxy_valid'] for r in rows) and valid_margin
    positive=numerical and main['mean']>=cfg['gates']['mean_stress_iou_gain'] and main['ci95'][0]>0 and witness is not None and witness['gain']>=cfg['gates']['witness_retention_gain'] and witness['ci95'][0]>0 and all(a['mean']>=0 for a in arms.values()) and margin['ci95'][0]>0
    if scope=='smoke':
        decision='ENGINEERING_ONLY'
    elif not numerical:
        decision='INCONCLUSIVE_NUMERICAL'
    elif positive:
        decision='STABILITY_DESIGN_SUPPORTED_REFERENCE_PENDING'
    elif main['ci95'][1]<cfg['gates']['mean_stress_iou_gain'] or (witness and witness['ci95'][1]<cfg['gates']['witness_retention_gain']):
        decision='NO_MATERIAL_STABILITY_GAIN_AT_SCOPE'
    else:
        decision='INCONCLUSIVE_NOT_READY'
    return {'pairs':len(rows),'images':len(ids),'accepted_centers':valid,'fallback_centers':len(rows)-valid,
            'numerical_valid':numerical,'stress_iou_gain':main,'per_direction_gain':arms,'geometric_margin_gain':margin,
            'witness_retention':witness,'reference_comparisons':refs,'G_budget':cfg['G_budget'],
            'G_budget_same_panel_gain':None,'G_budget_comparison_status':'unavailable_checkpoint',
            'decision':decision,'training_readiness':'not_ready_G_budget_reference_pending' if positive else 'not_recommended_at_scope',
            'new_training_steps':0,'base_model_forward_calls':0,'adapter_forward_calls':0,'scope':'GT_assisted_fixed_detection_not_AP'}
