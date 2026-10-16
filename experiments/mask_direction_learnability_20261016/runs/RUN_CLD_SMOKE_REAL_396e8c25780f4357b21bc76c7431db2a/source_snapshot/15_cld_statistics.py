"""CLD-v1 image-macro paired inference, fixed denominators and decisions."""
from collections import defaultdict
import numpy as np


def effects(rows, comparisons):
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_id']].append(row)
    ids = sorted(groups)
    if len(ids) < 2:
        raise ValueError('insufficient images')
    draws = np.random.default_rng(1729).integers(0, len(ids), size=(2000, len(ids)))
    result = {}
    for left, right in comparisons:
        values = np.asarray([np.mean([r[left] - r[right] for r in groups[i]]) for i in ids], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite endpoint')
        result[left + '-' + right] = dict(mean=float(values.mean()), ci95=np.quantile(values[draws].mean(1), [.025, .975], method='linear').tolist(),
                                         images=len(ids), instances=len(rows), unit='IoU_fraction_0_to_1')
    return result, ids, draws


def summarize(direction, pairs, tasks):
    held = [r for r in direction if r['split'] == 'heldout']
    fit = [r for r in direction if r['split'] == 'fit']
    target = [r for r in pairs if r['population'] == 'T']
    if (len(held), len(fit), len(pairs), len(target), sum(r['N'] for r in tasks)) != (581, 2510, 2246, 490, 1808):
        raise ValueError('frozen population differs')
    comparisons = [('R', 'P'), ('R', 'J'), ('R', 'N'), ('R', 'C')]
    d, di, dd = effects(held, comparisons)
    f, fi, fd = effects(fit, comparisons)
    t, ti, td = effects(target, [('M', 'N'), ('M', 'G'), ('M', 'S')])
    a, ai, ad = effects(pairs, [('M', 'N'), ('M', 'G'), ('M', 'S')])
    if (len(di), len(ti), len(ai)) != (208, 203, 394):
        raise ValueError('frozen image denominator differs')
    census = {name: dict(damaged=sum(r['N'] and not r[name] for r in tasks), recovered=sum(not r['N'] and r[name] for r in tasks),
                        native_successes=1808) for name in ('M', 'S', 'G')}
    kd = all(d[key]['mean'] >= .005 and d[key]['ci95'][0] > 0 for key in ('R-P', 'R-J', 'R-N'))
    ku = t['M-N']['mean'] >= .005 and t['M-N']['ci95'][0] > 0 and a['M-N']['mean'] >= -.001 and t['M-G']['ci95'][0] > 0 and census['M']['damaged'] <= 18
    dir_negative = any(d[key]['ci95'][1] < .005 for key in ('R-P', 'R-J', 'R-N'))
    use_negative = t['M-N']['ci95'][1] < .005 or t['M-G']['ci95'][1] <= 0 or census['M']['damaged'] > 18 or a['M-N']['mean'] < -.001
    decision = 'POSITIVE_INTERNAL_SCREEN' if kd and ku else 'NEGATIVE_AT_BUDGET' if dir_negative or use_negative else 'INCONCLUSIVE_AT_BUDGET'
    descriptive = {}
    for label, rows in (('fit', fit), ('heldout', held), ('heldout_gate_on', [r for r in held if r['gate']])):
        descriptive[label] = dict(instances=len(rows), reaches_075={k: sum(r[k] >= .75 for r in rows) for k in ('N','C','R','P','J')},
                                  zero_directions={k: sum(r['zero_' + k] for r in rows) for k in ('R','P','J')},
                                  mean_functional_cosine={k: float(np.mean([r['cosine_' + k] for r in rows])) for k in ('R','P','J')})
    if descriptive['heldout_gate_on']['instances'] != 62:
        raise ValueError('gate-on teacher denominator differs')
    result = dict(direction=d, fit_direction=f, target=t, overall=a, census=census, descriptive=descriptive,
                  mechanism_pass=kd, utility_pass=ku, mechanism_negative=dir_negative, utility_negative=use_negative,
                  decision=decision, output_identity_guard='passed_required_by_producer', scope='internal_development_not_formal_S2_or_AP',
                  old_G_comparison='historical_output_reference_trained_on_all_2000_not_matched_holdout_training')
    return result, dict(direction=dd, fit_direction=fd, target=td, overall=ad), dict(direction=di, fit_direction=fi, target=ti, overall=ai)
