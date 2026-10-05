"""Frozen native semantic response, spatial controls and original-basis projection.

Linux CUDA server only. The source receives prediction fields only. COCO labels
are used after prediction for fixed-candidate evaluation and descriptive strata.
No checkpoint, threshold, support, coefficient head or original weights are fitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

if '--config' in sys.argv:
    _config = json.loads(Path(sys.argv[sys.argv.index('--config') + 1]).read_text(encoding='utf-8-sig'))
    sys.path.insert(0, _config['source_python'])

import torch
from pycocotools.coco import COCO
from ultralytics.utils import ops
from online_runtime import dump, load_json, load_asset, load_index, resolve_runtime_config, sha256
from semantic_source import SemanticSource
from full_projection import prepare_prototype, solve_full_projection
import evaluation_metrics as em

ARMS = ('A', 'SEM', 'MIX', 'PROJ', 'ROT_MIX', 'ROT_PROJ', 'AGN_PROJ')
PRIMARY_PAIRS = (('PROJ', 'A'), ('PROJ', 'MIX'), ('PROJ', 'ROT_PROJ'))
SECONDARY_PAIRS = (('MIX', 'A'), ('MIX', 'ROT_MIX'), ('PROJ', 'AGN_PROJ'), ('SEM', 'A'))
PAIRS = PRIMARY_PAIRS + SECONDARY_PAIRS
METRICS = ('iou', 'mask75', 'coverage', 'auc', 'fpr')
KEYS = ('split', 'image_id', 'annotation_id', 'branch', 'raw_id', 'pyramid_level', 'target_gt_idx')
COCO80 = (1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,
          33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
          60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90)
em.ARMS = ARMS


def key(row):
    return tuple(row[k] for k in KEYS)


def ensure_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Fixed resource limit; retain incomplete Run, never shrink cohort')


def prediction_support(image, index, device):
    return ops.crop_mask(torch.ones((1, 640, 640), device=device),
                         image['boxes'][index:index+1].to(device))[0].bool()


@torch.no_grad()
def rotate_within_support(logits, support):
    """Exact integer-grid permutation, preserving the support histogram.

    The unrounded prediction determines the official crop. Its nonempty integer
    support is a rectangle; only that rectangle is rotated. Empty supports stay
    unchanged. No interpolation, target label or instance geometry enters here.
    """
    rotated = logits.clone()
    indices = support.nonzero()
    if not len(indices):
        return rotated, dict(support_pixels=0, empty_support=True, rectangle_yxyx=None,
                             histogram_preserved_exact=True, inverse_permutation_exact=True,
                             spatial_change_rms=0., spatial_change_mean_absolute=0.,
                             spatial_change_nonzero_fraction=0., spatial_control_activated=False)
    top, left = map(int, indices.min(0).values.tolist())
    bottom, right = (int(v)+1 for v in indices.max(0).values.tolist())
    if int(support.sum()) != (bottom-top)*(right-left) or not bool(support[top:bottom, left:right].all()):
        raise AssertionError('Prediction crop is not the declared integer rectangle')
    patch = logits[top:bottom, left:right]
    rotated[top:bottom, left:right] = torch.flip(patch, (0, 1))
    inverse_exact = torch.equal(torch.flip(rotated[top:bottom, left:right], (0, 1)), patch)
    outside_exact = torch.equal(rotated[~support], logits[~support])
    if not inverse_exact or not outside_exact:
        raise AssertionError('Rotation altered values or pixels outside the support')
    delta = (rotated[support]-logits[support]).double()
    return rotated, dict(support_pixels=int(support.sum()), empty_support=False,
        rectangle_yxyx=[top, left, bottom, right], histogram_preserved_exact=True,
        histogram_proof='bijective torch.flip on the exact rectangular integer support',
        inverse_permutation_exact=inverse_exact, outside_support_exact=outside_exact,
        spatial_change_rms=float(delta.square().mean().sqrt()),
        spatial_change_mean_absolute=float(delta.abs().mean()),
        spatial_change_nonzero_fraction=float(delta.ne(0).double().mean()),
        spatial_control_activated=bool(delta.ne(0).any()))


def logit_stats(logits, support):
    values = logits[support].double()
    if not len(values):
        return dict(pixels=0, mean=None, std=None, minimum=None, maximum=None, positive_fraction=None)
    return dict(pixels=len(values), mean=float(values.mean()), std=float(values.std(unbiased=False)),
                minimum=float(values.min()), maximum=float(values.max()),
                positive_fraction=float(values.gt(0).double().mean()))


@torch.no_grad()
def direct_metrics(logits, image, index, coco, annotation_id):
    if tuple(logits.shape) != (640, 640) or not bool(torch.isfinite(logits).all()):
        raise AssertionError('Invalid continuous source')
    support = prediction_support(image, index, logits.device)
    binary = (logits.gt(0) & support).byte()[None]
    manual = ops.crop_mask(logits[None].clone(), image['boxes'][index:index+1].to(logits.device)).gt(0).byte()
    if not torch.equal(binary, manual):
        raise AssertionError('Crop/threshold baseline mismatch')
    mask = em._scale_binary(binary, tuple(image['original_shape']), image['ratio_pad'])[0]
    annotation = coco.anns[int(annotation_id)]
    if int(annotation['image_id']) != int(image['image_id']):
        raise AssertionError('Annotation identity changed')
    gt = torch.as_tensor(coco.annToMask(annotation).astype(bool), device=logits.device)
    intersection = int((mask & gt).sum())
    iou = intersection/max(1, int((mask | gt).sum()))
    auc, fpr = em._pixel_auc_fpr(logits, em._padded_gt(gt, image['ratio_pad'], (640, 640)), support)
    return dict(iou=iou, mask75=int(iou >= .75), coverage=intersection/max(1, int(gt.sum())),
                auc=auc, fpr=fpr, empty_mask=not bool(mask.any()))


def evaluation_strata(image, index, coco, predicted_class):
    """GT-derived strata ONLY; called after all source responses are computed.

    A same-class neighbor is a different ordinary annotation of the recipient's
    GT category whose original COCO box has positive-area overlap with the frozen
    recipient prediction box. Prediction-box inverse geometry uses saved ratio_pad.
    This is box proximity, not mask overlap or an assertion of physical occlusion.
    """
    ann = coco.anns[int(image['rows'][index]['annotation_id'])]
    if not 0 <= int(predicted_class) < len(COCO80):
        raise AssertionError('Predicted class is not a contiguous COCO80 index')
    box = list(map(float, image['boxes'][index].detach().cpu().tolist()))
    oh, ow = map(int, image['original_shape'])
    if image['ratio_pad'] is None:
        rh = rw = min(640/oh, 640/ow)
        left, top = (640-round(ow*rw))/2, (640-round(oh*rh))/2
    else:
        ratios, pads = image['ratio_pad']
        rh, rw = (float(ratios), float(ratios)) if isinstance(ratios, (float, int)) else map(float, ratios)
        left, top = map(float, pads)
    if rh <= 0 or rw <= 0:
        raise AssertionError('Invalid frozen letterbox scale')
    pred_original = [max(0., min(float(ow), (box[0]-left)/rw)),
                     max(0., min(float(oh), (box[1]-top)/rh)),
                     max(0., min(float(ow), (box[2]-left)/rw)),
                     max(0., min(float(oh), (box[3]-top)/rh))]
    neighbors = []
    for other in coco.imgToAnns[int(image['image_id'])]:
        if int(other['id']) == int(ann['id']) or other.get('iscrowd', 0) or int(other['category_id']) != int(ann['category_id']):
            continue
        x, y, w, h = map(float, other['bbox'])
        if w > 0 and h > 0 and min(pred_original[2], x+w) > max(pred_original[0], x) and min(pred_original[3], y+h) > max(pred_original[1], y):
            neighbors.append(int(other['id']))
    return dict(predicted_class=int(predicted_class), predicted_category_id=COCO80[int(predicted_class)],
        predicted_class_correct=COCO80[int(predicted_class)] == int(ann['category_id']),
        sameclass_neighbor=bool(neighbors), sameclass_neighbor_annotation_ids=sorted(neighbors),
        sameclass_neighbor_count=len(neighbors), frozen_prediction_box_original=pred_original,
        stratum_GT_used_for_evaluation_only=True)


def summarize(rows, out, population):
    groupspec = {'all': lambda r: True, 'box_good_mask_bad': lambda r: bool(r['box_good_mask_bad']),
        'original_success': lambda r: bool(r['mask75_A']), 'original_failure': lambda r: not r['mask75_A'],
        'predclass_correct': lambda r: bool(r['predicted_class_correct']),
        'predclass_incorrect': lambda r: not r['predicted_class_correct'],
        'sameclass_neighbor': lambda r: bool(r['sameclass_neighbor']),
        'no_sameclass_neighbor': lambda r: not r['sameclass_neighbor']}
    for level in range(3):
        groupspec['P'+str(3+level)] = lambda r, level=level: r['pyramid_level'] == level
    for size in ('small', 'medium', 'large'):
        groupspec[size] = lambda r, size=size: r['size_group'] == size
    tables, per_image = {}, []
    for name, select in groupspec.items():
        selected = [r for r in rows if select(r)]
        groups = em.image_groups(selected)
        table = dict(images=len(groups), candidates=len(selected), candidate={}, image_macro={},
                     undefined={}, comparisons={}, empty_masks={a: sum(bool(r.get('empty_mask_'+a)) for r in selected) for a in ARMS})
        for metric in METRICS:
            table['candidate'][metric] = {a: em.avg([r.get(metric+'_'+a) for r in selected]) for a in ARMS}
            table['image_macro'][metric] = {a: em.avg([em.avg([r.get(metric+'_'+a) for r in g]) for g in groups]) for a in ARMS}
            table['undefined'][metric] = {a: sum(not em.finite(r.get(metric+'_'+a)) for r in selected) for a in ARMS}
        for arm, reference in PAIRS:
            pair = {m: em.paired(groups, arm, reference, m, 20261004, 1000) for m in METRICS}
            repair = sum(r['mask75_'+arm] == 1 and r['mask75_'+reference] == 0 for r in selected)
            damage = sum(r['mask75_'+arm] == 0 and r['mask75_'+reference] == 1 for r in selected)
            pair['crossings'] = dict(repair=repair, damage=damage, net=repair-damage)
            table['comparisons'][arm+'_minus_'+reference] = pair
        for group in groups:
            record = dict(split='dev', image_id=group[0]['image_id'], group=name, candidates=len(group))
            for metric in METRICS:
                for arm in ARMS:
                    record[metric+'_'+arm] = em.avg([r.get(metric+'_'+arm) for r in group])
                for arm, reference in PAIRS:
                    record['delta_'+metric+'_'+arm+'_minus_'+reference] = em.avg([
                        r[metric+'_'+arm]-r[metric+'_'+reference] for r in group
                        if em.finite(r.get(metric+'_'+arm)) and em.finite(r.get(metric+'_'+reference))])
            per_image.append(record)
        tables[name] = table
    em.append_rows(out/'PER_IMAGE.jsonl', per_image)
    decisions = {}
    for arm in ('MIX', 'PROJ'):
        all_ = tables['all']['comparisons'][arm+'_minus_A']
        target = tables['box_good_mask_bad']['comparisons'][arm+'_minus_A']
        delta, target_delta = all_['iou']['image_macro'], target['iou']['image_macro']
        overall = (em.finite(delta['delta']) and delta['delta'] >= .002 and
                   em.finite(delta['ci95'][0]) and delta['ci95'][0] > 0 and all_['crossings']['net'] > 0)
        subgroup = (em.finite(target_delta['delta']) and target_delta['delta'] >= .005 and
                    em.finite(target_delta['ci95'][0]) and target_delta['ci95'][0] > 0 and
                    em.finite(delta['delta']) and delta['delta'] >= -.001)
        decisions[arm] = dict(prespecified_overall_signal=bool(overall), prespecified_target_signal=bool(subgroup),
                             other_metrics_require_effect_tradeoff_review=True)
    result = dict(schema='native-semantic-response-projection-v1', population=population, arms=ARMS,
        primary_pairs=PRIMARY_PAIRS, secondary_pairs=SECONDARY_PAIRS, tables=tables,
        bootstrap=dict(draws=1000, seed=20261004, unit='paired image'), decisions=decisions,
        no_training=True, scope='Historical dev / fixed GT-conditioned official candidates; not blind test or COCO AP',
        automatic_followup=False)
    dump(out/'SUMMARY.json', em.clean(result))
    def pp(value):
        return 'undefined' if not em.finite(value) else f'{value*100:+.4f}'
    lines = ['# Native semantic response and original-basis projection', '',
        'Frozen original YOLO26m-seg. No training or parameter selection; all reported mask effects are percentage points.',
        f"{population['planned_images']} planned dev images, {population['effective_images']} effective images, {len(rows)} fixed official candidates.", '',
        'A is the original mask. SEM directly decodes the predicted-class semantic logits; MIX averages original and semantic logits equally. PROJ applies the fixed full-support ridge solver to half their logit difference. ROT controls rotate only the semantic response inside the prediction box. AGN_PROJ substitutes the maximum class logit.', '',
        '| Group | Pair | Image macro IoU delta [95% CI] | Candidate IoU delta | Repair / damage | Coverage delta | AUC delta | FPR delta |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, table in tables.items():
        for pair, record in table['comparisons'].items():
            delta, crossings = record['iou']['image_macro'], record['crossings']
            lines.append(f"| {name} | {pair} | {pp(delta['delta'])} [{pp(delta['ci95'][0])}, {pp(delta['ci95'][1])}] | {pp(record['iou']['candidate']['delta'])} | {crossings['repair']} / {crossings['damage']} | {pp(record['coverage']['image_macro']['delta'])} | {pp(record['auc']['image_macro']['delta'])} | {pp(record['fpr']['image_macro']['delta'])} |")
    lines += ['', 'Interpretation boundaries:',
        '- The three primary comparisons are PROJ-A, PROJ-MIX and PROJ-ROT_PROJ. Multiple intervals are exploratory, without a confirmatory familywise-error claim.',
        '- PROJ exceeding MIX only supports this fixed ridge/alpha projection against this direct mixture. PROJ exceeding ROT_PROJ supports this within-box spatial arrangement; it is not proof of general instance ownership.',
        '- Rotation preserves the exact support-value histogram but may change little for nearly symmetric/constant maps. SOURCE_ROWS records RMS, mean absolute and nonzero spatial changes; an inactive contrast cannot reject spatial information.',
        '- AGN takes a maximum across classes, which changes logit scale/distribution. PROJ versus AGN_PROJ alone cannot identify a pure class-specific mechanism.',
        '- Same-class-neighbor strata use other noncrowd GT boxes with positive-area overlap with the recipient prediction box. They describe proximity, not actual mask overlap or occlusion. Predicted-class correctness and neighbor labels enter evaluation only.',
        '- No candidate is removed for an empty mask or undefined AUC. Failure-group damage is structurally zero versus A and does not exclude continuous degradation.',
        '- These reused dev images are not a new blind test. Fixed TAL candidates are not complete deployment outputs or COCO AP. No training or automatic follow-up is authorized by a positive submetric.',
        '', 'Prespecified decisions: '+json.dumps(decisions, ensure_ascii=False)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


@torch.no_grad()
def run(args):
    if os.name == 'nt' or not torch.cuda.is_available():
        raise RuntimeError('Authorized Linux CUDA server only')
    started = time.monotonic()
    cfg = resolve_runtime_config(load_json(args.config))
    limit = 180 if args.smoke else 1800
    deadline = started + limit
    if cfg['alpha'] != .5 or cfg['projection_lambda'] != .003 or cfg['bootstrap'] != 1000:
        raise AssertionError('Frozen configuration changed')
    if int(cfg.get('bootstrap_seed', 20261004)) != 20261004:
        raise AssertionError('Frozen bootstrap seed changed')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('PER_CANDIDATE.jsonl', 'COMPLETE.json', 'FAILURE.json')):
        raise RuntimeError('Existing results; use an independent Run')
    index, split = load_index(cfg), load_json(cfg['split'])
    if len(index['dev']) != 256 or [int(e['image_id']) for e in index['dev']] != list(map(int, split['dev'])):
        raise AssertionError('Changed dev membership/order')
    if sum(bool(e['n']) for e in index['dev']) != 253 or sum(int(e['n']) for e in index['dev']) != 1816:
        raise AssertionError('Frozen official dev population changed')
    items = [e for e in index['dev'] if e['n']][:2] if args.smoke else index['dev']
    population = dict(planned_images=len(items), effective_images=sum(bool(e['n']) for e in items),
        candidates=sum(int(e['n']) for e in items), no_positive_images=[int(e['image_id']) for e in items if not e['n']], smoke=args.smoke)
    wanted, previous = {int(e['image_id']) for e in items}, {}
    for line in Path(cfg['prior_results']).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['split'] == 'dev' and int(row['image_id']) in wanted:
            if key(row) in previous:
                raise AssertionError('Duplicate historical identity')
            previous[key(row)] = row
    if len(previous) != population['candidates']:
        raise AssertionError('Historical population differs')
    source, coco = SemanticSource(cfg), COCO(cfg['annotations_train'])
    dump(out/'ENVIRONMENT.json', source.import_info)
    paths = (Path(args.config), Path(cfg['split']), Path(cfg['weights']), Path(cfg['prior_results']),
             Path(cfg['annotations_train']), Path(cfg['cache'])/'CACHE_IDENTITY.json', Path(cfg['cache'])/'INDEX.json')
    inputs = {str(path): sha256(path) for path in paths}
    dump(out/'INPUT_MANIFEST.json', dict(inputs=inputs, code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))
    dump(out/'EVALUATION_INDEX.json', population)
    rows, source_rows, image_audits, solver_audits = [], [], [], []
    seen, reference_error, direct_baseline_error, payload_fields = set(), 0., 0., set()
    zero_target_checks = 0
    for position, entry in enumerate(items):
        ensure_time(deadline)
        if not entry['n']:
            continue
        image = load_asset(cfg, int(entry['image_id']), True)
        if image['split'] != 'dev' or len(image['rows']) != int(entry['n']):
            raise AssertionError('Changed image cohort')
        for field in ('proto', 'boxes', 'c0'):
            image[field] = image[field].to('cuda').float()
        payload = {k: image[k] for k in ('input_uint8', 'boxes', 'c0', 'proto', 'raw_ids', 'levels')}
        for field in ('predicted_classes', 'predicted_scores'):
            if field in image:
                payload[field] = image[field]
        payload_fields.update(payload)
        prototype_up = prepare_prototype(image['proto'])
        coefficients = {arm: [] for arm in ('PROJ', 'ROT_PROJ', 'AGN_PROJ')}
        direct, predicted_classes, image_sources = [], [], []
        for record in source.infer(payload, deadline_monotonic=deadline):
            ensure_time(deadline)
            j = int(record['candidate_index'])
            if j != len(direct):
                raise AssertionError('Candidate order changed')
            live_base, semantic, agnostic = (record[k] for k in ('baseline_logits', 'semantic_logits', 'agnostic_logits'))
            base = em._logits(image['proto'], image['c0'][j], (640, 640))
            if any(tuple(t.shape) != (640, 640) or not bool(torch.isfinite(t).all()) for t in (live_base, base, semantic, agnostic)):
                raise AssertionError('Invalid continuous source response')
            predicted_class = int(record['predicted_class'])
            if 'predicted_classes' in image and predicted_class != int(image['predicted_classes'][j]):
                raise AssertionError('Semantic class does not match original predicted class')
            support = prediction_support(image, j, base.device)
            rotated, rotation = rotate_within_support(semantic, support)
            if j == 0:
                zero = solve_full_projection(image['proto'], image['c0'][j], image['boxes'][j], torch.zeros_like(base),
                    prototype_up=prototype_up, deadline_monotonic=deadline)
                if not torch.equal(zero['coefficients'], image['c0'][j]) or bool(zero['delta'].ne(0).any()):
                    raise AssertionError('Zero-target self projection is not identity')
                zero_target_checks += 1
            projections = {}
            for arm, signal in (('PROJ', semantic), ('ROT_PROJ', rotated), ('AGN_PROJ', agnostic)):
                solution = solve_full_projection(image['proto'], image['c0'][j], image['boxes'][j], .5*(signal-base),
                    prototype_up=prototype_up, deadline_monotonic=deadline)
                coefficients[arm].append(solution['coefficients_fp32'])
                projections[arm] = solution['diagnostics']
                solver_audits.append(dict(image_id=int(entry['image_id']), candidate_index=j, arm=arm, **solution['diagnostics']))
            # No GT has entered source selection, rotation, mixing or the solve.
            ann = image['rows'][j]['annotation_id']
            metrics = {arm: direct_metrics(z, image, j, coco, ann) for arm, z in
                (('A', base), ('SEM', semantic), ('MIX', .5*(base+semantic)), ('ROT_MIX', .5*(base+rotated)))}
            direct.append(metrics)
            predicted_classes.append(predicted_class)
            source_row = dict(image_id=int(entry['image_id']), candidate_index=j, raw_id=int(image['raw_ids'][j]),
                predicted_class=predicted_class, predicted_score=float(record['predicted_score']) if 'predicted_score' in record else None,
                projection=projections, rotation=rotation, coefficient_c0_l2=float(image['c0'][j].double().norm()),
                logits={name: logit_stats(z, support) for name, z in
                        (('baseline_c0_cached', base), ('baseline_c0_live', live_base),
                         ('semantic', semantic), ('rotated_semantic', rotated), ('agnostic', agnostic))})
            source_rows.append(source_row)
            image_sources.append(source_row)
        if len(direct) != len(image['rows']):
            raise AssertionError('Source omitted candidate')
        current = em.evaluate_image(image, dict(A=image['c0'], **{arm: torch.stack(values) for arm, values in coefficients.items()}), coco)
        if len(current) != len(image['rows']):
            raise AssertionError('Normal decoder omitted candidate')
        for j, row in enumerate(current):
            for metric in METRICS:
                value, other = row.get(metric+'_A'), direct[j]['A'].get(metric)
                if em.finite(value) != em.finite(other):
                    raise AssertionError('Direct baseline definedness differs')
                if em.finite(value):
                    direct_baseline_error = max(direct_baseline_error, abs(value-other))
            if direct_baseline_error > 1e-12:
                raise AssertionError('Direct source baseline differs from native baseline')
            for arm in ('SEM', 'MIX', 'ROT_MIX'):
                for metric, value in direct[j][arm].items():
                    row[metric+'_'+arm] = value
            identity = key(row)
            if identity in seen or identity not in previous:
                raise AssertionError('Candidate identity changed or repeated')
            seen.add(identity)
            for metric in METRICS:
                value, old = row.get(metric+'_A'), previous[identity].get(metric+'_A')
                if em.finite(value) != em.finite(old):
                    raise AssertionError('Historical baseline definedness changed')
                if em.finite(value):
                    reference_error = max(reference_error, abs(value-old))
            if reference_error > 1e-12:
                raise AssertionError('Historical A replay changed')
            row.update(evaluation_strata(image, j, coco, predicted_classes[j]))
            row['semantic_rotation'] = image_sources[j]['rotation']
        rows.extend(current)
        em.append_rows(out/'PER_CANDIDATE.jsonl', current)
        image_audits.append(dict(image_id=int(entry['image_id']), **source.last_audit))
        dump(out/'PROGRESS.json', dict(stage='evaluating', images_done=position+1, images_planned=len(items),
             candidates_done=len(rows), seconds=time.monotonic()-started))
        del prototype_up
    if len(rows) != population['candidates'] or seen != set(previous):
        raise AssertionError('Population not complete')
    ensure_time(deadline)
    source.replay.assert_unchanged()
    em.append_rows(out/'SOURCE_ROWS.jsonl', em.clean(source_rows))
    audit = dict(passed=True, population=population, reference_metric_max_error=reference_error,
        direct_baseline_metric_max_error=direct_baseline_error, identity_unique=True, no_training=True,
        original_model_unchanged=True, source_image_audits=image_audits, solver_diagnostics=solver_audits,
        GT_free_payload_fields=sorted(payload_fields), source_rows='SOURCE_ROWS.jsonl',
        self_zero_target_exact=True, self_zero_target_checks=zero_target_checks,
        rotation_histogram_preserved_exact=all(r['rotation']['histogram_preserved_exact'] for r in source_rows),
        rotation_changed_candidates=sum(r['rotation']['spatial_control_activated'] for r in source_rows),
        rotation_mean_RMS=em.avg([r['rotation']['spatial_change_rms'] for r in source_rows]),
        GT_used_only_for_candidate_membership_evaluation_and_strata=True, limit_seconds=limit)
    dump(out/'AUDIT.json', em.clean(audit))
    summary = summarize(rows, out, population)
    ensure_time(deadline)
    dump(out/'COMPLETE.json', dict(passed=True, completed=True, smoke=args.smoke, seconds=time.monotonic()-started,
        population=population, decisions=summary['decisions'], no_training=True, no_automatic_followup=True))
    print(json.dumps(dict(completed=True, seconds=time.monotonic()-started, population=population, decisions=summary['decisions'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    # A duplicate invocation must never overwrite a completed/failed attempt.
    if any((out/name).exists() for name in ('COMPLETE.json', 'FAILURE.json', 'PER_CANDIDATE.jsonl')):
        raise RuntimeError('Existing Run artifacts; refusing overwrite')
    try:
        run(args)
    except BaseException as exc:
        if not (out/'COMPLETE.json').exists() and not (out/'FAILURE.json').exists():
            dump(out/'FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(), completed=False))
        raise


if __name__ == '__main__':
    main()
