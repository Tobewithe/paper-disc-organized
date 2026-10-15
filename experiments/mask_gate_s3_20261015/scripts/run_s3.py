"""One smoke then one full S3 evaluation, no new training or base-model forward."""
from common import *
import argparse
from collections import Counter
import contextlib
import gc
import hashlib
import json
import subprocess
import time
import numpy as np
import psutil
from coco_bootstrap import evaluate, compact, save_cache, load_cache, resampled_evaluator, ap, verify_replication


def resources(run):
    run.check_budget()
    require(psutil.Process().memory_info().rss < 7 * 1024**3, 'RSS limit 7 GiB exceeded')
    require(psutil.virtual_memory().available >= 768 * 1024**2, 'available RAM below 768 MiB')
    require(shutil.disk_usage(ROOT).free >= 6 * 1024**3, 'disk reserve below 6 GiB')


def export(run, ids, reuse_old):
    import torch
    from adapter_data import FrozenCache
    from adapter_model import CoefficientAdapter
    from native_output_bridge import decode_native, encode_rle
    from pycocotools import mask as masks
    from posttrain_audit import compare_outputs
    from s1_geometry import iter_prediction_array
    source = FrozenCache(BASE / 'trace', ids)
    old_indices = read_json(GATE / 'ROW_MAPPING.json')['original_global_indices']
    old = {i: row for i, row in zip(old_indices, iter_prediction_array(GATE / 'MIXED_predictions.json'), strict=True)}
    old_ids = set(read_json(GATE / 'EVALUATION_IMAGES.json'))
    model = CoefficientAdapter('B').cuda()
    saved = torch.load(TRAIN / 'last.pt', map_location='cpu', weights_only=True)
    require(saved['arm'] == 'B' and saved['completed_epochs'] == 3, 'wrong trained adapter')
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    del saved
    counters = Counter()
    timing = Counter()
    indices, delta_records = [], {}
    start = time.monotonic()
    output = run.path / 'predictions_mixed.json'
    native_subset = run.path / 'predictions_native_smoke.json'
    with output.open('x', encoding='utf8') as stream, (run.path / 'GATE_ROWS.jsonl').open('x', encoding='utf8') as gates, torch.inference_mode():
        stream.write('[')
        comma = False
        baseline_subset = []
        for number, image_id in enumerate(source.records):
            resources(run)
            cache, original = source.load(image_id)
            rows = len(original)
            require(len(cache['c0']) == rows, 'cache/native row mismatch')
            apply = [int(masks.area(row['segmentation'])) >= 9216 for row in original]
            areas = [int(masks.area(row['segmentation'])) for row in original]
            if rows:
                zero = decode_native(cache)
                for rank, row in enumerate(original):
                    require(encode_rle(zero[rank]) == row['segmentation'], 'zero-delta native RLE mismatch')
                counters['zero_delta_exact_rows'] += rows
                del zero
            else:
                counters['empty_output_images'] += 1
            if reuse_old and image_id in old_ids:
                mixed = [old[i] for i in cache['prediction_indices']]
                counters['reused_S2_images'] += 1
                counters['reused_S2_rows'] += rows
            elif rows:
                torch.cuda.synchronize()
                t = time.monotonic()
                delta = model(cache, 'eval:0')
                torch.cuda.synchronize()
                timing['adapter_cuda_synchronized_seconds'] += time.monotonic() - t
                require(torch.isfinite(delta).all().item(), 'nonfinite delta')
                delta_records[image_id] = delta.detach().cpu()
                t = time.monotonic()
                decoded = decode_native(cache, cache['c0'].to(delta) + delta)
                torch.cuda.synchronize()
                timing['corrected_decoder_cuda_synchronized_seconds'] += time.monotonic() - t
                mixed = [{**row, 'segmentation': encode_rle(decoded[rank]) if apply[rank] else row['segmentation']}
                         for rank, row in enumerate(original)]
                counters['adapter_forward_images'] += 1
                counters['adapter_forward_rows'] += rows
                del decoded, delta
            else:
                mixed = []
            compare_outputs(original, mixed)
            for rank, (native, actual, index) in enumerate(zip(original, mixed, cache['prediction_indices'], strict=True)):
                if index in old:
                    require(actual == old[index], 'old 1314-image candidate changed')
                    counters['exact_old_candidate_rows'] += 1
                if not apply[rank]:
                    require(actual == native, 'unselected row is not native')
                changed = actual['segmentation'] != native['segmentation']
                counters['selected_rows'] += int(apply[rank])
                counters['changed_masks'] += int(changed)
                counters['outputs'] += 1
                counters['empty_native_masks'] += int(areas[rank] == 0)
                counters['exact_9216_boundary'] += int(areas[rank] == 9216)
                gate = dict(image_id=image_id, global_index=index, native_mask_area=areas[rank],
                            apply_delta=apply[rank], mask_changed=changed, reused_S2=bool(reuse_old and image_id in old_ids))
                gates.write(json.dumps(gate) + '\n')
                if comma:
                    stream.write(',')
                stream.write(json.dumps(actual, separators=(',', ':'), allow_nan=False))
                comma = True
            indices.extend(cache['prediction_indices'])
            if run.meta['scope'] == 'smoke':
                baseline_subset.extend(original)
            counters['images'] += 1
            if (number + 1) % 100 == 0 or number + 1 == len(ids):
                stream.flush()
                run.update(phase='export', completed_images=number+1, total_images=len(ids))
                print('S3_EXPORT', number+1, '/', len(ids), 'seconds', round(time.monotonic()-start, 2), flush=True)
            del cache, original, mixed
            torch.cuda.empty_cache()
        stream.write(']\n')
    require(indices == sorted(set(indices)), 'final native output order changed')
    if run.meta['scope'] == 'full':
        require(indices == list(range(539128)), 'not all final output rows')
        require(counters['exact_old_candidate_rows'] == 185621, 'old candidate overlap incomplete')
    else:
        write_new(native_subset, baseline_subset)
    write_new(run.path / 'ROW_MAPPING.json', dict(source_prediction_sha256=source.manifest['predictions_sha256'],
                                                original_global_indices=indices))
    torch.save(delta_records, run.path / 'DELTA.pt')
    result = dict(counters)
    result.update(timing)
    result.update(export_wall_seconds=time.monotonic()-start, predictions_sha256=sha256(output),
                  delta_sha256=sha256(run.path/'DELTA.pt'), base_model_forward_calls=0, new_training_steps=0,
                  cost_scope='Cached P/c0/F adapter and native decoder; not end-to-end YOLO deployment speed')
    write_new(run.path / 'EXPORT.json', result)
    del source, model, old, delta_records
    gc.collect()
    torch.cuda.empty_cache()
    return output, result


def point(run, label, prediction_file, ids, baseline_check=False):
    from build_native_ledger import METRICS, extract_matches
    predictions = read_json(prediction_file)
    result = dict(metrics={}, matching={}, prediction_sha256=sha256(prediction_file), image_count=len(ids))
    old = {(r['image_id'], r['annotation_id']): r for r in read_jsonl(BASE / 'ledger/PER_GT_FAILURE.jsonl')} if baseline_check else None
    for task in ('bbox', 'segm'):
        resources(run)
        run.update(phase='point_evaluation', arm=label, task=task)
        t = time.monotonic()
        with (run.path / f'{label}_{task}_coco.log').open('x', encoding='utf8') as log, contextlib.redirect_stdout(log):
            ev = evaluate(GT, predictions, ids, task=task)
        result['metrics'][task] = dict(zip(METRICS, [100*float(x) if x >= 0 else None for x in ev.stats]))
        matches = extract_matches(ev)
        result['matching'][task] = [dict(image_id=k[0], annotation_id=k[1], **v) for k, v in sorted(matches.items())]
        if baseline_check:
            require(set(matches) == set(old), 'S0 ledger GT coverage changed')
            field = 'bbox75_match_output_id' if task == 'bbox' else 'segm75_match_output_id'
            for key, match in matches.items():
                wanted = old[key][field]
                want_dt = 0 if wanted is None else int(wanted.rsplit(':', 1)[1]) + 1
                require(match['ignore'] == old[key]['is_ignored'] and match['d75'] == want_dt, 'S0 GT matching changed')
            expected = read_json(BASE / 'ledger/SUMMARY.json')['metrics'][task]
            require(all(abs(result['metrics'][task][key]-value) < 1e-10 for key, value in expected.items()), 'S0 AP changed')
        if task == 'segm':
            cache = compact(ev)
            save_cache(run.path / f'{label}_MATCH_CACHE.pkl', cache)
            require(abs(ap(resampled_evaluator(cache, list(range(len(ids))))) - result['metrics'][task]['AP']) < 1e-12,
                    'compact-match COCO AP not identical')
            del cache
        print('S3_COCO', label, task, 'seconds', round(time.monotonic()-t, 2), flush=True)
        del ev, matches
        gc.collect()
    write_new(run.path / f'{label}_POINT.json', result)
    del predictions, old
    gc.collect()
    return result


def config_audit(run, expected, environment):
    from native_output_bridge import native_environment
    for rel, digest in expected['files'].items():
        require(sha256(ROOT / rel) == digest, 'input modified during evaluation: ' + rel)
    require(native_environment() == environment, 'runtime drift during evaluation')
    rss = psutil.Process().memory_info()
    write_new(run.path / 'CONFIG_AUDIT.json', dict(status='passed', input_files=len(expected['files']),
        actual_runtime_equal_original=True, native_profile_sha256=sha256(PROFILE), checkpoint_sha256=sha256(TRAIN/'last.pt'),
        base_weight_sha256=sha256(WEIGHT), original_area_threshold=9216, new_training_steps=0, base_model_forward_calls=0,
        frozen_inference_geometry_reused=True, evaluation_area_policy='complete bbox+segmentation JSON; loadRes bbox area',
        peak_process_rss_bytes=getattr(rss, 'peak_wset', rss.rss)))


def smoke(run, ids, expected, environment):
    subprocess.run([sys.executable, '-B', str(STUDY/'scripts/test_bootstrap.py')], check=True)
    output, exported = export(run, ids, reuse_old=False)
    native_pred = read_json(run.path / 'predictions_native_smoke.json')
    method_pred = read_json(output)
    gt = read_json(GT)
    checks = []
    for label, predictions in [('native', native_pred), ('mixed', method_pred)]:
        cache = compact(evaluate(GT, predictions, ids, ap_only=True, quiet=True))
        draws = [0, 1, 1, 2, 4, 4, 4, 7]
        checks.append(dict(arm=label, **verify_replication(cache, gt, predictions, draws)))
    write_new(run.path / 'DUPLICATE_IMAGE_PARITY.json', checks)
    del gt, native_pred, method_pred, cache
    gc.collect()
    # Baseline-only full-data work is reused by the formal comparison; it does
    # not inspect any new full-data method outcome or tune the method.
    base = point(run, 'BASE_FULL', NATIVE, expected['image_ids'], baseline_check=True)
    cache = load_cache(run.path / 'BASE_FULL_MATCH_CACHE.pkl')
    rng = np.random.default_rng(1729)
    durations = []
    for _ in range(3):
        resources(run)
        draws = rng.integers(0, 5000, size=5000)
        t = time.monotonic()
        ev = resampled_evaluator(cache, draws)
        require(np.isfinite(ap(ev)), 'invalid bootstrap AP')
        durations.append(time.monotonic()-t)
        del ev
    estimated = 2 * 2000 * max(durations)
    require(estimated < 18000, 'bootstrap cost does not fit frozen six-hour budget')
    timing = dict(single_arm_replicate_seconds=durations, conservative_2000_pair_seconds=estimated,
                  formal_wall_budget_seconds=21600, benchmarking_source='baseline only')
    write_new(run.path / 'COST_PREFLIGHT.json', timing)
    config_audit(run, expected, environment)
    run.finish(dict(export=exported, duplicate_image_parity='passed', full_baseline_metrics=base['metrics'], cost=timing),
               status='passed', scientific='engineering_only', next_action='Run one frozen S3 full5000 evaluation')


def full(run, smoke_path, expected, environment):
    require(read_json(smoke_path/'DECISION.json')['status'] == 'passed', 'smoke not passed')
    require(sha256(smoke_path/'PROTOCOL.md') == sha256(STUDY/'PROTOCOL.md'), 'smoke/full protocol changed')
    base = read_json(smoke_path/'BASE_FULL_POINT.json')
    base_cache = load_cache(smoke_path/'BASE_FULL_MATCH_CACHE.pkl')
    output, exported = export(run, expected['image_ids'], reuse_old=True)
    method = point(run, 'METHOD', output, expected['image_ids'])
    require(method['metrics']['bbox'] == base['metrics']['bbox'], 'Box AP changed')
    require(method['matching']['bbox'] == base['matching']['bbox'], 'Box matching changed')
    native_matches = {(r['image_id'], r['annotation_id']): r for r in base['matching']['segm']}
    modified_matches = {(r['image_id'], r['annotation_id']): r for r in method['matching']['segm']}
    require(set(native_matches) == set(modified_matches), 'method GT coverage changed')
    census = []
    for key, a in native_matches.items():
        b = modified_matches[key]
        require(a['ignore'] == b['ignore'], 'ordinary GT denominator changed')
        census.append(dict(image_id=key[0], annotation_id=key[1], ignored=a['ignore'],
                           baseline_success=bool(a['d75']) and not a['ignore'],
                           method_success=bool(b['d75']) and not b['ignore'],
                           native_dt_id=a['d75'], method_dt_id=b['d75']))
    jsonl(run.path/'GT_CENSUS.jsonl', census)
    successes = [r for r in census if r['baseline_success']]
    require(len(successes) == 22957, 'full success census denominator changed')
    damaged = [r for r in successes if not r['method_success']]
    recovered = [r for r in census if not r['ignored'] and not r['baseline_success'] and r['method_success']]
    jsonl(run.path/'DAMAGED_GT.jsonl', damaged)
    jsonl(run.path/'RECOVERED_GT.jsonl', recovered)
    damage = dict(damaged=len(damaged), population=len(successes), observed=len(successes),
                  exact_rate=len(damaged)/len(successes), upper95=len(damaged)/len(successes),
                  statistical_scope='exact finite COCO val2017 census, not future-scene guarantee',
                  recovered_from_baseline_failures=len(recovered))
    write_new(run.path/'DAMAGE_CENSUS.json', damage)
    method_cache = load_cache(run.path/'METHOD_MATCH_CACHE.pkl')
    require(base_cache['image_ids'] == method_cache['image_ids'] == expected['image_ids'], 'unpaired image universe')
    rng = np.random.default_rng(1729)
    draws_hash = hashlib.sha256()
    deltas = []
    begin = time.monotonic()
    with (run.path/'AP_BOOTSTRAP.jsonl').open('x', encoding='utf8') as stream:
        for i in range(2000):
            resources(run)
            draws = rng.integers(0, 5000, size=5000)
            draws_hash.update(draws.astype('<i8').tobytes())
            ev_a = resampled_evaluator(base_cache, draws)
            a = ap(ev_a)
            del ev_a
            ev_b = resampled_evaluator(method_cache, draws)
            b = ap(ev_b)
            del ev_b
            delta = b-a
            deltas.append(delta)
            stream.write(json.dumps(dict(replicate=i, baseline_AP=a, method_AP=b, delta_AP=delta))+'\n')
            if (i+1) % 25 == 0:
                stream.flush()
                run.update(phase='paired_AP_bootstrap', completed_resamples=i+1, total_resamples=2000)
                print('S3_BOOTSTRAP', i+1, '/ 2000', 'seconds', round(time.monotonic()-begin, 2), flush=True)
    interval = [float(v) for v in np.quantile(deltas, [.025, .975], method='linear')]
    effect = method['metrics']['segm']['AP'] - base['metrics']['segm']['AP']
    uncertainty = dict(resamples=2000, seed=1729, rng='numpy.default_rng PCG64', quantile='linear',
        ci95=interval, unit='AP_points_0_to_100', point_delta=effect,
        image_draws_sha256=draws_hash.hexdigest(), elapsed_seconds=time.monotonic()-begin,
        estimator='paired independent image copies; pinned COCOeval.accumulate for every replicate',
        repeated_images_receive_unique_image_annotation_detection_ids=True,
        scope='development-data bootstrap, not independent confirmation')
    write_new(run.path/'AP_UNCERTAINTY.json', uncertainty)
    gates = dict(mask_AP_point=effect >= .3, paired_AP_ci_lower=interval[0] > 0,
                 success_damage=damage['exact_rate'] <= .01, frozen_Box_outputs=True)
    config_audit(run, expected, environment)
    summary = dict(baseline=base['metrics'], method=method['metrics'], delta_mask_AP_points=effect,
                   uncertainty=uncertainty, damage=damage, gates=gates, export=exported,
                   baseline_evaluation_run=smoke_path.name, baseline_point_sha256=sha256(smoke_path/'BASE_FULL_POINT.json'),
                   published_baseline_parity='failed_separate_historical_status',
                   native_gt_ledger_validity='passed', independent_confirmation=False)
    passed = all(gates.values())
    run.finish(summary, status='passed' if passed else 'failed',
               scientific='S3_development_utility_supported' if passed else 'S3_preregistered_utility_failed',
               next_action=('Plan independent confirmation of this exact selective-repair candidate; no automatic training' if passed else
                            'Close this 9216-area candidate for deployment utility; synthesize coefficient-repair reach versus safe selection before any new experiment'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['smoke', 'full'], required=True)
    parser.add_argument('--smoke-run', type=Path)
    args = parser.parse_args()
    expected, environment = inspect_inputs()
    if args.scope == 'full':
        require(args.smoke_run is not None, 'full requires exact completed smoke')
        require(not list((STUDY/'runs').glob('RUN_S3_FULL_*')), 'formal run already exists; do not repeat automatically')
    parent = args.smoke_run if args.scope == 'full' else GATE
    with S3Run(args.scope, parent) as run:
        contract = dict(stage='S3', scope=args.scope, checkpoint_sha256=sha256(TRAIN/'last.pt'),
            source_baseline_run=BASE.name, native_profile_sha256=sha256(PROFILE), environment=environment,
            input_expectations_sha256=sha256(STUDY/'INPUT_EXPECTATIONS.json'),
            image_ids=expected['smoke_ids'] if args.scope == 'smoke' else expected['image_ids'],
            native_binary_original_image_area_min=9216, zero_training=True,
            immutable_output_fields='all except selected segmentation RLE',
            bootstrap=dict(resamples=2000, seed=1729, quantile='linear', paired=True,
                           new_ids_per_occurrence=True, recompute='official COCOeval.accumulate from cloned independent image matches'),
            gates=dict(delta_mask_AP_points_min=.3, delta_AP_CI_lower_strict_min=0,
                       exact_success_damage_max=.01, success_population=22957),
            budgets=dict(smoke_wall_seconds=3600, full_wall_seconds=21600, max_process_RSS_GiB=7,
                         min_available_RAM_MiB=768, min_disk_free_GiB=6, training_steps=0, base_model_forward_calls=0),
            full_baseline_reuse_from_smoke=str(args.smoke_run) if args.smoke_run else None)
        run.freeze(contract)
        resources(run)
        if args.scope == 'smoke':
            smoke(run, expected['smoke_ids'], expected, environment)
        else:
            full(run, args.smoke_run, expected, environment)


if __name__ == '__main__':
    main()
