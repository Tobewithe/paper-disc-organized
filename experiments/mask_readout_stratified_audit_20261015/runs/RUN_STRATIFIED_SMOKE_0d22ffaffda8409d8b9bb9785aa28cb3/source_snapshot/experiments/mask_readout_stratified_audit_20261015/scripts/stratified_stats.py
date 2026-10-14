"""Frozen exploratory strata; no model, fitting, threshold search or COCO rematching."""
from collections import Counter
import hashlib
import numpy as np
from scipy.stats import beta

STAGES = ('soft_roi', 'binary_roi', 'binary_proto_full', 'binary_input_full', 'binary_original')
SIGNATURES = ('soft_roi_already_down', 'proto_binary_reversal', 'full_support_reversal',
              'proto_to_input_reversal', 'input_to_original_reversal', 'matching_despite_fixed_mask75')
AREAS = ('S', 'M', 'L')
CROWDS = ('Low', 'Mid', 'High')


def area_group(value):
    if not np.isfinite(value) or value < 0:
        raise ValueError('invalid area')
    return 'S' if value < 1024 else 'M' if value < 9216 else 'L'


def crowd_group(value):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid crowd IoU')
    return 'Low' if value < .1 else 'Mid' if value < .5 else 'High'


def neighbors(boxes, classes, raw_ids, ranks):
    boxes = np.asarray(boxes, dtype=np.float64)
    classes, raw_ids = np.asarray(classes), np.asarray(raw_ids)
    out = {}
    for rank in ranks:
        box = boxes[rank]
        xy = np.maximum(0., np.minimum(box[2:], boxes[:, 2:])-np.maximum(box[:2], boxes[:, :2]))
        inter = xy.prod(1)
        area = np.maximum(0., boxes[:, 2:]-boxes[:, :2]).prod(1)
        union = area + np.maximum(0., box[2:]-box[:2]).prod()-inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        different = raw_ids != raw_ids[rank]
        same = different & (classes == classes[rank])
        out[rank] = dict(all_class_neighbor_iou=float(iou[different].max()) if different.any() else 0.,
                         same_class_neighbor_iou=float(iou[same].max()) if same.any() else 0.,
                         different_raw_neighbors=int(different.sum()), same_class_neighbors=int(same.sum()))
    return out


def bootstrap_frame(image_ids):
    ids = sorted(image_ids)
    rng = np.random.default_rng(1729)
    weights = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
                        for _ in range(2000)]).astype(float)
    ordered = sorted(ids, key=lambda i: hashlib.sha256(f'stratified-audit-v1:{i}'.encode()).hexdigest())
    return dict(ids=ids, index={v: i for i, v in enumerate(ids)}, weights=weights,
                halves=(set(ordered[:len(ids)//2]), set(ordered[len(ids)//2:])))


def interval(draws, quantiles=(.025, .975)):
    valid = np.asarray(draws)[np.isfinite(draws)]
    return dict(interval=[float(x) for x in np.quantile(valid, quantiles, method='linear')] if len(valid) else None,
                valid_draws=int(len(valid)))


def aggregate(rows, values, frame, macro):
    valid = [(r, v) for r, v in zip(rows, values) if v is not None]
    totals = np.zeros(len(frame['ids']))
    counts = np.zeros_like(totals)
    for r, v in valid:
        idx = frame['index'][r['image_id']]
        totals[idx] += v
        counts[idx] += 1
    if macro:
        totals = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
        counts = (counts > 0).astype(float)
    den = frame['weights'] @ counts
    draws = np.divide(frame['weights'] @ totals, den, out=np.full(2000, np.nan), where=den > 0)
    return (float(totals.sum()/counts.sum()) if counts.sum() else None), draws


def rate(rows, values, frame):
    n, k = len(values), sum(values)
    point, draws = aggregate(rows, values, frame, False)
    exact = ([0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
              1. if k == n else float(beta.ppf(.975, k+1, n-k))] if n else None)
    return dict(numerator=int(k), denominator=n, rate=point, cluster95=interval(draws),
                exact_binomial95_supplement=exact, degenerate_cluster_rate=bool(n and k in (0, n)))


def cell_summary(rows, frame, cohort):
    stages = {}
    for stage in STAGES:
        values = [r['stages'][stage]['delta'] for r in rows]
        good = [v for v in values if v is not None]
        mean, draws = aggregate(rows, values, frame, True)
        stages[stage] = dict(valid_pairs=len(good), pair_mean=float(np.mean(good)) if good else None,
                             image_macro_mean=mean, image_macro_cluster95=interval(draws))
    delta = [r['stages']['binary_original']['delta'] for r in rows]
    data = dict(n=len(rows), images=len({r['image_id'] for r in rows}),
                gains=sum(d > 1e-12 for d in delta), harms=sum(d < -1e-12 for d in delta),
                ties=sum(abs(d) <= 1e-12 for d in delta), stages=stages,
                fixed_detection_reaches075=rate(rows, [r['stages']['binary_original']['after']['iou'] >= .75 for r in rows], frame))
    if cohort == 'target_T':
        data['task_repair'] = rate(rows, [r['task_repaired'] for r in rows], frame)
    if cohort == 'success_panel':
        data['task_damage'] = rate(rows, [r['task_damaged'] for r in rows], frame)
        counts = Counter(r['damage_stage_signature'] for r in rows if r['task_damaged'])
        data['damaged_stage_counts'] = {s: counts[s] for s in SIGNATURES}
        data['damaged_stage_denominator'] = len(rows)
    return data


def adjusted_contrast(rows, frame, area_key, factor, endpoint):
    # Positive contrast: High-Low crowding, or Small-(Medium+Large) area.
    adjustment = 'all_crowd_group' if factor == 'area' else area_key
    levels = CROWDS if factor == 'area' else AREAS
    side = (lambda r: 1 if r[area_key] == 'S' else 0) if factor == 'area' else (
        lambda r: 1 if r['all_crowd_group'] == 'High' else 0 if r['all_crowd_group'] == 'Low' else -1)
    value = (lambda r: float(r['task_damaged'])) if endpoint == 'damage' else (
        lambda r: r['stages']['binary_original']['delta'])
    macro = endpoint != 'damage'
    fixed_weights = {lv: sum(r[adjustment] == lv for r in rows)/len(rows) for lv in levels}
    point, draws, halves = 0., np.zeros(2000), [0., 0.]
    adequate, half_adequate, support = True, True, {}
    for lv in levels:
        samples = [[r for r in rows if r[adjustment] == lv and side(r) == s] for s in (0, 1)]
        counts = [dict(n=len(g), images=len({r['image_id'] for r in g})) for g in samples]
        adequate &= all(x['n'] >= 10 and x['images'] >= 5 for x in counts)
        estimates = [aggregate(g, [value(r) for r in g], frame, macro) for g in samples]
        if any(v[0] is None for v in estimates):
            point = float('nan')
        else:
            point += fixed_weights[lv]*(estimates[1][0]-estimates[0][0])
        draws += fixed_weights[lv]*(estimates[1][1]-estimates[0][1])
        half_counts = []
        for h, ids in enumerate(frame['halves']):
            groups = [[r for r in g if r['image_id'] in ids] for g in samples]
            hc = [dict(n=len(g), images=len({r['image_id'] for r in g})) for g in groups]
            half_counts.append(hc)
            half_adequate &= all(x['n'] >= 5 and x['images'] >= 3 for x in hc)
            values = []
            for g in groups:
                if not g:
                    values.append(float('nan'))
                elif macro:
                    img = {}
                    for r in g:
                        img.setdefault(r['image_id'], []).append(value(r))
                    values.append(float(np.mean([np.mean(v) for v in img.values()])))
                else:
                    values.append(float(np.mean([value(r) for r in g])))
            halves[h] += fixed_weights[lv]*(values[1]-values[0])
        support[lv] = dict(weight=fixed_weights[lv], full=counts, halves=half_counts)
    ci = interval(draws, (.05/16, 1-.05/16))
    finite = bool(np.isfinite(point) and all(np.isfinite(halves)))
    passed = bool(adequate and half_adequate and finite and ci['valid_draws'] == 2000 and
                  (ci['interval'][0] > 0 or ci['interval'][1] < 0) and
                  all(v*point > 0 for v in halves) and abs(point) >= (.02 if endpoint == 'damage' else .005))
    return dict(area_definition=area_key, factor=factor, endpoint=endpoint,
                contrast='High-minus-Low' if factor == 'crowd' else 'S-minus-ML',
                estimate=point if np.isfinite(point) else None, familywise95=ci,
                half_estimates=[v if np.isfinite(v) else None for v in halves],
                adequate_full_counts=bool(adequate), adequate_half_counts=bool(half_adequate),
                support=support, stable_exploratory_heterogeneity=passed)


def analyze(rows, dev_ids, panel_ids, scientific=True):
    frames = {'dev_all': bootstrap_frame(dev_ids), 'success_panel': bootstrap_frame(panel_ids)}
    frames['target_T'] = frames['dev_all']
    cohorts = {'dev_all': [r for r in rows if r['role'] == 'dev_bbox75'],
               'target_T': [r for r in rows if r['role'] == 'dev_bbox75' and r['population'] == 'T'],
               'success_panel': [r for r in rows if r['role'] == 'success_segm75']}
    tables = []
    for cohort, selected in cohorts.items():
        for area_key in ('gt_area_group', 'native_area_group'):
            for crowd_key in ('all_crowd_group', 'same_crowd_group'):
                for a, c in [(None, None)]+[(x, None) for x in AREAS]+[(None, x) for x in CROWDS]+[(a,c) for a in AREAS for c in CROWDS]:
                    group = [r for r in selected if (a is None or r[area_key] == a) and (c is None or r[crowd_key] == c)]
                    tables.append(dict(cohort=cohort, area_definition=area_key, crowd_definition=crowd_key,
                                       area=a or 'All', crowd=c or 'All', **cell_summary(group, frames[cohort], cohort)))
    contrasts = []
    if scientific:
        for area_key in ('gt_area_group', 'native_area_group'):
            for factor in ('crowd', 'area'):
                for endpoint, cohort in (('target_delta', 'target_T'), ('damage', 'success_panel')):
                    contrasts.append(adjusted_contrast(cohorts[cohort], frames[cohort], area_key, factor, endpoint))
    crossing = Counter((r['role'], r['gt_area_group'], r['native_area_group']) for r in rows)
    return dict(cohort_sizes={k:len(v) for k,v in cohorts.items()}, tables=tables, contrasts=contrasts,
                area_cross_tab=[dict(role=k[0], gt_area=k[1], native_area=k[2], n=v) for k,v in sorted(crossing.items())],
                boundary_counts={key: {str(bound):sum(r[key] == bound for r in rows) for bound in (1024,9216)}
                                 for key in ('gt_area','native_foreground_area')},
                inference='Exploratory reused development data; intervals do not confer independent confirmation')


def decide(summary):
    supported = []
    for endpoint in ('damage', 'target_delta'):
        for factor in ('crowd', 'area'):
            pair = [r for r in summary['contrasts'] if r['factor'] == factor and r['endpoint'] == endpoint]
            if len(pair) == 2 and all(r['stable_exploratory_heterogeneity'] for r in pair) and pair[0]['estimate']*pair[1]['estimate'] > 0:
                supported.append(dict(factor=factor, endpoint=endpoint, direction_sign=int(np.sign(pair[0]['estimate']))))
    if supported:
        chosen = supported[0]
        direction = 'neighbor_conditioned_coefficient_correction' if chosen['factor'] == 'crowd' else 'native_area_conditioned_selective_application'
        next_action = ('Freeze a single-variable real-versus-shuffled-neighbor protocol; no new gate or loss, no automatic training'
                       if chosen['factor'] == 'crowd' else
                       'Freeze one native-mask-area-conditioned mixed-output evaluation protocol using complete predictions and original S2 gates; no training')
        return dict(status='exploratory_route_supported', direction=direction, chosen=chosen,
                    supported=supported, next_action=next_action, original_S2='failed_unchanged')
    return dict(status='conditional_route_not_supported_by_frozen_rule', supported=[],
                direction='native_input_geometry_aligned_coefficient_objective',
                next_action='Freeze a native-input-geometry GT-objective-only protocol with explicit threshold/resize surrogate, unchanged preservation term and original S2 gates; no automatic training',
                original_S2='failed_unchanged')
