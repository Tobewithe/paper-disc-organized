"""Replay original masks, construct one fixed support, evaluate complete outputs."""
from lea_common import *
import argparse, gc, time
from collections import defaultdict
import numpy as np
import torch
from pycocotools import mask as masks
from pycocotools.coco import COCO
from adapter_data import FrozenCache
from native_geometry import native_stages
from native_output_bridge import decode_native, native_environment
from s1_geometry import iter_prediction_array
from lea_ops import encode, decode, iou, compose, gate_from_logits, pair_record, counts
from lea_eval import match75, damage_records
from lea_stats import summarize


def execute(run):
    cfg, manifest, profile, env = inspect_inputs()
    scope = run.meta['scope']
    targets = manifest[scope + '_targets']
    ids = sorted({r['image_id'] for r in targets})
    expected = manifest['denominator'][scope]
    source = FrozenCache(BASE / 'trace', ids)
    indices = sorted(source.predictions)
    needed_indices = set(indices)
    existing = {i: p for i, p in enumerate(iter_prediction_array(S3 / 'predictions_mixed.json')) if i in needed_indices}
    require(set(existing) == needed_indices, 'Existing G output rows incomplete')
    old_c = {r['output_detection_id']: r for r in read_jsonl((CFULL if scope == 'full' else CSMOKE) / 'PAIR_RESULTS.jsonl')}
    baseline_ledger = [r for r in read_jsonl(BASE / 'ledger/PER_GT_FAILURE.jsonl') if r['image_id'] in set(ids)]
    successes = [r for r in baseline_ledger if not r['is_ignored'] and r['task_mask75'] == 'success']
    by_image, success_by_image = defaultdict(list), defaultdict(list)
    for t in targets:
        by_image[t['image_id']].append(t)
    for r in successes:
        success_by_image[r['image_id']].append(r)
    run.freeze(dict(config=cfg, scope=scope, environment=env, native_profile=profile, image_ids=ids,
                    denominator=expected, candidate_policy='original_S1_fixed_final_Box75_detection',
                    decoder='unchanged_native_then_final_original_binary_support_composition',
                    region='abs(native_prethreshold_logit_scaled_to_original)<=1.5 inside serialized_bbox',
                    parameters_changed=['final_original_binary_edit_support'],
                    training_steps=0, base_model_forward_calls=0, adapter_forward_calls=0,
                    oracle_optimization_calls=0, no_new_AP=True))
    write_new(run.path / 'TARGETS.json', targets)
    write_new(run.path / 'ROW_MAPPING.json', dict(global_indices=indices, source_prediction_sha=source.manifest['predictions_sha256']))
    write_new(run.path / 'INPUT_CACHE_RECORDS.json', [source.records[i] for i in ids])
    dataset = read_json(GT)
    subset = {k: v for k, v in dataset.items() if k not in ('images', 'annotations')}
    subset['images'] = [r for r in dataset['images'] if r['id'] in set(ids)]
    subset['annotations'] = [r for r in dataset['annotations'] if r['image_id'] in set(ids)]
    del dataset
    write_new(run.path / 'ANNOTATIONS_SUBSET.json', subset)
    coco = COCO()
    coco.dataset = subset
    coco.createIndex()
    rows, support_rows, gate_rows = [], [], []
    modified = {}
    zero_rows, changed_outputs, teacher_replays = 0, 0, 0
    started = time.monotonic()
    with torch.inference_mode():
        for step, iid in enumerate(ids, 1):
            resource_check(run)
            cache, native_predictions = source.load(iid)
            rank_of = {gi: r for r, gi in enumerate(cache['prediction_indices'])}
            target_rank = {rank_of[int(t['output_detection_id'].rsplit(':', 1)[1])]: t for t in by_image[iid]}
            changed = [r for r, gi in enumerate(cache['prediction_indices'])
                       if existing[gi]['segmentation'] != native_predictions[r]['segmentation']]
            success_ranks = {rank_of[int(s['segm75_match_output_id'].rsplit(':', 1)[1])] for s in success_by_image[iid]}
            wanted = sorted(set(changed) | set(target_rank) | success_ranks)
            first, final, logits = native_stages(cache, cache['c0'], wanted)
            require(torch.equal(final, decode_native(cache).cpu()), 'Native stage/bridge mismatch')
            for rank, p in enumerate(native_predictions):
                gi = cache['prediction_indices'][rank]
                require(encode(final[rank].numpy()) == p['segmentation'], 'Zero delta mismatch')
                require({k: v for k, v in p.items() if k != 'segmentation'} ==
                        {k: v for k, v in existing[gi].items() if k != 'segmentation'}, 'G changed detection identity')
                zero_rows += 1
                modified[gi] = p
            bands, rois = {}, {}
            for rank in wanted:
                gi = cache['prediction_indices'][rank]
                band, roi = gate_from_logits(logits[rank], cache, native_predictions[rank]['bbox'], cfg['tau'])
                bands[rank], rois[rank] = band, roi
                gate_rows.append(dict(image_id=iid, rank=rank, global_index=gi,
                                      output_detection_id=source.manifest['predictions_sha256'] + ':' + str(gi),
                                      bbox=native_predictions[rank]['bbox'], B_rle=encode(band), ROI_rle=encode(roi),
                                      editable_pixels=int(band.sum()), roi_pixels=int(roi.sum())))
            for rank in changed:
                gi = cache['prediction_indices'][rank]
                n = final[rank].numpy().astype(bool)
                g = decode(existing[gi]['segmentation'])
                mixed = compose(n, g, bands[rank])
                require(not np.any((mixed != n) & ~bands[rank]), 'Outside-support edits')
                modified[gi] = dict(native_predictions[rank], segmentation=encode(mixed))
                changed_outputs += int(modified[gi]['segmentation'] != native_predictions[rank]['segmentation'])
            if target_rank:
                coefficients = cache['c0'].clone()
                for rank, t in target_rank.items():
                    c = old_c[t['output_detection_id']]
                    require(c['annotation_id'] == t['annotation_id'] and c['native_prediction'] == native_predictions[rank], 'C pair identity')
                    require(c['cache_sha256'] == source.records[iid]['cache_sha256'] and c['c0'] == cache['c0'][rank].tolist(), 'C tensor identity')
                    require(c['arms']['C']['solver']['solver_status'] == 'converged', 'C unaccepted')
                    coefficients[rank] = torch.tensor(c['arms']['C']['coefficients_fp32'])
                teacher = decode_native(cache, coefficients).cpu()
                unchanged = [r for r in range(len(native_predictions)) if r not in target_rank]
                require(torch.equal(teacher[unchanged], final[unchanged]), 'C replay alters nontarget output')
                for rank, t in target_rank.items():
                    c = old_c[t['output_detection_id']]
                    require(encode(teacher[rank].numpy()) == c['arms']['C']['original_rle'], 'C RLE replay differs')
                    gt = coco.annToRLE(coco.anns[t['annotation_id']])
                    gt = dict(size=gt['size'], counts=gt['counts'].decode('ascii'))
                    require(gt == c['GT_rle'], 'GT pair changed')
                    gi = cache['prediction_indices'][rank]
                    record = pair_record(dict(**t, rank=rank, global_index=gi,
                                              cache_sha256=source.records[iid]['cache_sha256']),
                                         native_predictions[rank]['segmentation'], c['arms']['C']['original_rle'],
                                         existing[gi]['segmentation'], gt, bands[rank], rois[rank], cfg)
                    require(record['arms']['G_B']['rle'] == modified[gi]['segmentation'], 'Pair/full output mismatch')
                    rows.append(record)
                    teacher_replays += 1
                del teacher, coefficients
            for s in success_by_image[iid]:
                gi = int(s['segm75_match_output_id'].rsplit(':', 1)[1])
                rank = rank_of[gi]
                n = final[rank].numpy().astype(bool)
                y = masks.decode(coco.annToRLE(coco.anns[s['annotation_id']])).astype(bool)
                b = bands[rank]
                worst = compose(n, ~y, b)
                support_rows.append(dict(image_id=iid, annotation_id=s['annotation_id'], global_index=gi,
                                         native_rle=encode(n), GT_rle=encode(y), native_fixed_iou=iou(n, y),
                                         worst_fixed_iou=iou(worst, y), **counts(n, n, y, b)))
            del cache, native_predictions, first, final, logits, bands, rois
            gc.collect()
            torch.cuda.empty_cache()
            if step % 10 == 0 or step == len(ids):
                run.update(phase='native_and_local_support_replay', completed_images=step,
                           completed_pairs=len(rows), native_rows_replayed=zero_rows)
                print('REPLAY', step, '/', len(ids), 'pairs', len(rows), 'seconds', round(time.monotonic()-started, 1), flush=True)
    require(len(rows) == expected['pairs'] and zero_rows == expected['native_prediction_rows'], 'Incomplete output denominator')
    jsonl(run.path / 'PAIRS.jsonl', rows)
    jsonl(run.path / 'GATES.jsonl', gate_rows)
    jsonl(run.path / 'SUCCESS_SUPPORT.jsonl', support_rows)
    jsonl(run.path / 'SOURCE_GT_LEDGER.jsonl', baseline_ledger)
    predictions = dict(N=[source.predictions[i] for i in indices], G=[existing[i] for i in indices],
                       G_B=[modified[i] for i in indices])
    all_matches = {}
    for name, preds in predictions.items():
        resource_check(run)
        write_new(run.path / (name + '_predictions.json'), preds)
        match = match75(subset, preds, indices)
        jsonl(run.path / (name + '_MATCHES.jsonl'), match)
        all_matches[name] = match
    old_map = {(r['image_id'], r['annotation_id']): r for r in baseline_ledger}
    require(set(old_map) == {(r['image_id'], r['annotation_id']) for r in all_matches['N']}, 'GT census differs')
    for r in all_matches['N']:
        old = old_map[(r['image_id'], r['annotation_id'])]
        require(r['ignored'] == old['is_ignored'] and r['success'] == (old['task_mask75'] == 'success'), 'S0 success/ignore mismatch')
        if r['success']:
            require(r['matched_global_index'] == int(old['segm75_match_output_id'].rsplit(':', 1)[1]), 'S0 match identity mismatch')
    damage, census = damage_records(all_matches)
    require(damage['G']['population'] == expected['native_success_gt'], 'Success denominator changed')
    jsonl(run.path / 'GT_CENSUS.jsonl', census)
    summary, draws, order = summarize(rows, damage, cfg)
    if scope == 'full':
        require(summary['support']['C_witnesses'] == 159, 'Original C witness count drift')
    else:
        summary['decision'] = 'ENGINEERING_SMOKE_ONLY'
    summary.update(zero_delta_exact_rows=zero_rows, teacher_replays=teacher_replays,
                   complete_predictions=len(indices), changed_G_B_outputs=changed_outputs,
                   native_success_fixed_match_exposed=sum(r['editable_tp'] + r['editable_tn'] > 0 for r in support_rows),
                   native_success_fixed_match_worst_below75=sum(r['worst_fixed_iou'] < .75 for r in support_rows),
                   support_identity_pixels_outside_changed=0, no_new_AP=True,
                   elapsed_before_verify_seconds=time.monotonic()-started)
    write_new(run.path / 'STATISTICS.json', summary)
    np.savez_compressed(run.path / 'BOOTSTRAP.npz', indices=draws, image_ids=np.array(order))
    source.verify()
    verify_files(manifest['files'])
    require(native_environment() == env, 'Post-run runtime drift')
    require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file()) <= cfg['budget']['output_bytes'], 'Output budget')
    write_new(run.path / 'CONFIG_AUDIT.json', dict(status='passed', source_files_verified=len(run.sources),
              input_files_verified_before_and_after=len(manifest['files']), cache_images_verified=len(ids),
              environment_unchanged=True, all_native_rle_exact=True, all_teacher_rle_exact=True,
              all_nonmask_fields_unchanged=True, all_edits_within_final_support=True,
              original_GT_ledger_reproduced=True, new_training_steps=0, model_forward_calls=0, adapter_forward_calls=0))
    from verify import verify
    check = verify(run.path, replay=False)
    write_new(run.path / 'INDEPENDENT_IN_RUN.json', check)
    require(check['status'] == 'passed', 'Independent in-run calculation failed')
    if scope == 'smoke':
        estimate = (time.monotonic()-started) * 243 / len(ids) * 1.25 + 120
        require(estimate <= cfg['budget']['full_seconds'], 'Projected full budget exceeded')
    run.finish(summary, status=summary['decision'], scientific=summary['decision'],
               next_action='Independent laptop replay and local verification; no automatic training')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=['smoke', 'full'], required=True)
    p.add_argument('--parent-run')
    a = p.parse_args()
    require(not list((STUDY / 'runs').glob('RUN_LEA_' + a.scope.upper() + '_*')), 'Stage already has a Run; no duplicate')
    parent = BASE
    if a.scope == 'full':
        require(a.parent_run and Path(a.parent_run).name == a.parent_run, 'Exact smoke Run ID required')
        parent = STUDY / 'runs' / a.parent_run
        require(read_json(parent / 'run.json')['execution_status'] == 'completed', 'Smoke incomplete')
        require(read_json(parent / 'INDEPENDENT_LAPTOP.json')['status'] == 'passed', 'Smoke not independently verified')
    run = Run(a.scope, parent)
    study = read_json(STUDY / 'study.json')
    try:
        with run:
            study.update(execution_status='running', artifact_status='partial', active_run_id=run.run_id, current_phase=a.scope)
            save(STUDY / 'study.json', study)
            execute(run)
    finally:
        if (run.path / 'run.json').exists():
            meta = read_json(run.path / 'run.json')
            study.update(execution_status=meta['execution_status'], scientific_status=meta['scientific_status'],
                         artifact_status=meta['artifact_status'], active_run_id=None, latest_run_id=run.run_id,
                         current_phase='awaiting_independent_verification')
            save(STUDY / 'study.json', study)


if __name__ == '__main__':
    main()
