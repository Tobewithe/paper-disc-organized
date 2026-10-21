"""Image-equal paired summaries; the frozen intersection gate."""
from collections import defaultdict
import math
import numpy as np


def gate_decision(summary, cfg):
    s, comparison = summary['support'], summary['comparisons']
    hard = dict(
        witness_retention=s['C_witness_retained_by_O_B'] >= math.ceil(cfg['min_C_witness_retention'] * s['C_witnesses']),
        locality=s['editable_fraction_image_macro'] <= cfg['max_editable_fraction'],
        real_damage=summary['task_damage']['G_B']['rate'] <= cfg['max_success_damage'],
        no_more_damage_than_G=summary['task_damage']['G_B']['damaged'] <= summary['task_damage']['G']['damaged'])
    soft = {k: comparison[k]['mean'] >= cfg['min_effect'] and comparison[k]['ci95'][0] > 0
            for k in ('O_B-G', 'O_B-O_R')}
    if not all(hard.values()) or any(comparison[k]['ci95'][1] < cfg['min_effect'] for k in soft):
        decision = 'NEGATIVE_CURRENT_REGION'
    elif all(soft.values()):
        decision = 'NECESSARY_CONDITIONS_PASSED'
    else:
        decision = 'INCONCLUSIVE_NOT_READY'
    return dict(**hard, **soft), decision


def summarize(rows, damage, cfg):
    ids = sorted({r['image_id'] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[r['image_id']].append(r)
    names = ['N', 'G', 'C', 'O_B', 'C_B', 'G_B', 'O_R']
    values = np.array([[np.mean([r['O_R'] if k == 'O_R' else r['arms'][k]['iou'] for r in by[i]])
                        for k in names] for i in ids], dtype=np.float64)
    draws = np.random.Generator(np.random.PCG64(cfg['seed'])).integers(
        0, len(ids), size=(cfg['bootstrap_replicates'], len(ids)))
    comparisons = {}
    todo = [(k, base) for k in names if k != 'N' for base in ('N', 'G') if k != base]
    todo += [('O_B', 'O_R'), ('C_B', 'C')]
    for a, b in todo:
        v = values[:, names.index(a)] - values[:, names.index(b)]
        ci = np.quantile(v[draws].mean(axis=1), [.025, .975], method='linear')
        comparisons[a + '-' + b] = dict(mean=float(v.mean()), ci95=ci.tolist())
    cs = [r for r in rows if r['arms']['C']['iou'] >= .75]
    support = dict(C_witnesses=len(cs),
                   C_witness_retained_by_O_B=sum(r['arms']['O_B']['iou'] >= .75 for r in cs),
                   C_witness_retained_by_C_B=sum(r['arms']['C_B']['iou'] >= .75 for r in cs),
                   editable_fraction_image_macro=float(np.mean([np.mean([r['editable_fraction'] for r in by[i]]) for i in ids])),
                   empty_rois=sum(r['empty_roi'] for r in rows))
    result = dict(pairs=len(rows), images=len(ids), unit='IoU_0_to_1', reduction='within-image then image-equal',
                  absolute={k: float(values[:, j].mean()) for j, k in enumerate(names)},
                  mask75={k: sum(r['arms'][k]['iou'] >= .75 for r in rows) for k in names if k != 'O_R'},
                  comparisons=comparisons, support=support, task_damage=damage)
    result['gates'], result['decision'] = gate_decision(result, cfg)
    return result, draws, ids
