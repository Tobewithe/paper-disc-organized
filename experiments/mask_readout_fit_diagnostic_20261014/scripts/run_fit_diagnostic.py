"""Zero-training census of the existing S2 B checkpoint on its original fit2000."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import contextlib
import gc
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'experiments/mask_readout_fit_diagnostic_20261014'
PARENT = ROOT / 'experiments/mask_readout_mechanism_v2_20261012'
sys.path.insert(0, str(PARENT / 'scripts'))
sys.path.insert(0, str(ROOT / 'shared/evaluation'))
sys.path.insert(0, str(ROOT / 'experiments/official_gt_failure_rebaseline_20261013/scripts'))
from posttrain_run import RegisteredRun, configure_native, tensor_state_hash, read_json, require, sha256, write_new
from posttrain_audit import read_jsonl, population, compare_outputs
from run_s2_evaluation import bootstrap, image_effects

B_RUN = PARENT / 'runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2'
CACHE_RUN = PARENT / 'runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec'
DEV_RUN = PARENT / 'runs/RUN_S2_EVAL_a24a564568c943eab362784abfbe2c54'
BASE_RUN = ROOT / 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
WEIGHT = ROOT / 'assets/models/coco_clean_20260911/yolo26m-seg.pt'
CHECKPOINT = B_RUN / 'last.pt'
GT = CACHE_RUN / 'dataset/training_annotations.json'
PROFILE = ROOT / 'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'
CHECKPOINT_SHA = '201ab6a7372c167e772217904eac8f4ba9946c26e5862c2882e3c01bdcd58e5e'
TRACE_SHA = 'fcae975a0fec3b03d0911785af2f9067b8bbd9ad4e6713aef7ba14911c96329d'
GT_SHA = '51477c03144651f01f941991661da85df0df17b6048a148feca0b7dad6bce26c'
CODE_FILES = [str(p.relative_to(ROOT)).replace('\\', '/') for p in (
    Path(__file__), STUDY/'scripts/test_fit_diagnostic.py', STUDY/'PROTOCOL.md',
    STUDY/'SOURCE_CONTEXT.json', STUDY/'SOURCE_DIRTY.patch',
    *[PARENT/'scripts'/n for n in ('posttrain_run.py', 'adapter_model.py', 'adapter_data.py',
        'adapter_primitives.py', 's1_geometry.py', 'run_s2_evaluation.py')],
    ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py',
    ROOT/'shared/evaluation/posttrain_audit.py', ROOT/'shared/evaluation/native_output_bridge.py',
    ROOT/'shared/evaluation/native_source_pins.json', PROFILE)]


class DiagnosticRun(RegisteredRun):
    """Reuse the checked lifecycle, with a separate study and current source provenance."""
    def __init__(self, scope):
        super().__init__('FIT_DIAG_' + scope.upper(), 'FIT_DIAGNOSTIC', B_RUN,
                         CODE_FILES, 900 if scope == 'smoke' else 3600)
        self.path = STUDY / 'runs' / self.run_id
        self.lock = STUDY / '.FIT_DIAGNOSTIC.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_FIT_DIAGNOSTIC_20261014',
                         scope=scope, optimization_started=False, optimizer_steps=0)

    def freeze(self, contract):
        context = read_json(STUDY/'SOURCE_CONTEXT.json')
        self.sources = {relative: sha256(ROOT/relative) for relative in self.code_files}
        self.meta.update(code_commit=context['code_commit'],
                         dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json', {'files': self.sources,
                  'code_commit': context['code_commit'], 'actual_sources_snapshotted': True})
        for i, relative in enumerate(self.sources):
            target = self.path/'source_snapshot'/(f'{i:02d}_'+Path(relative).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
        shutil.copyfile(STUDY/'PROTOCOL.md', self.path/'PROTOCOL.md')
        write_new(self.path/'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path/'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path/'PROTOCOL.md'),
                    contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'))


def external_sources():
    return {name: {'path': str(Path(importlib.import_module(name).__file__).resolve()),
                   'sha256': sha256(importlib.import_module(name).__file__)}
            for name in ('pycocotools.coco', 'pycocotools.cocoeval', 'pycocotools.mask', 'pycocotools._mask')}


def inspect_inputs():
    profile, environment = configure_native()
    training = read_json(B_RUN/'EVAL_CONTRACT.json')
    require(environment == training['environment'], 'runtime differs from original training environment')
    require(sha256(CHECKPOINT) == CHECKPOINT_SHA == read_json(B_RUN/'SUMMARY.json')['checkpoint_sha256'],
            'not the frozen final B checkpoint')
    require(sha256(GT) == GT_SHA, 'original training annotation subset changed')
    require(sha256(CACHE_RUN/'trace/TRACE_MANIFEST.json') == TRACE_SHA == training['source_trace_sha256'],
            'original native cache manifest changed')
    require(sha256(PROFILE) == training['profile_sha256'], 'baseline configuration identity changed')
    require(sha256(WEIGHT) == profile['model_sha256'], 'base weight changed')
    require(training['arm'] == 'B' and training['scope'] == 'full' and training['epochs'] == 3
            and training['expected_optimizer_steps'] == 1500, 'unexpected trained configuration')
    split = read_json(CACHE_RUN/'SPLIT_CONTRACT.json')
    require(sha256(CACHE_RUN/'SPLIT_CONTRACT.json') == training['fit_dev_split_sha256'], 'split changed')
    require(len(split['fit_ids']) == len(set(split['fit_ids'])) == 2000, 'wrong fit population')
    require(not set(split['fit_ids']) & set(split['dev_ids']), 'fit/dev overlap')
    for parent in (B_RUN, CACHE_RUN, DEV_RUN):
        require(read_json(parent/'run.json')['execution_status'] == 'completed', 'source run incomplete')
        require(read_json(parent/'CONFIG_AUDIT.json')['status'] == 'passed', 'source configuration audit failed')
    require(read_json(DEV_RUN/'DECISION.json')['status'] == 'failed', 'unexpected prior S2 decision')
    require(read_json(DEV_RUN/'EVAL_CONTRACT.json')['checkpoints']['B']['sha256'] == CHECKPOINT_SHA,
            'dev result came from another checkpoint')
    source_matches = {}
    for parent in (B_RUN, DEV_RUN):
        prior = read_json(parent/'SOURCE_MANIFEST.json')['files']
        for relative in CODE_FILES:
            if relative in prior:
                require(sha256(ROOT/relative) == prior[relative], 'source drift: '+relative)
                source_matches[relative] = prior[relative]
    native = read_json(CACHE_RUN/'trace/NATIVE_ARGS.json')
    original = read_json(BASE_RUN/'trace/NATIVE_ARGS.json')
    differences = {k: {'fit': native.get(k), 'S0': original.get(k)}
                   for k in set(native)|set(original) if native.get(k) != original.get(k)}
    require(set(differences) <= {'data', 'project'}, 'native argument drift')
    joins = read_json(B_RUN/'TRAINING_JOIN.json')
    require(set(map(int, joins['images'])) == set(split['fit_ids']), 'saved training joins incomplete')
    files = [CHECKPOINT, B_RUN/'initial.pt', B_RUN/'EVAL_CONTRACT.json', B_RUN/'TRAINING_JOIN.json',
             B_RUN/'IMAGE_ORDERS.json', B_RUN/'SUMMARY.json', CACHE_RUN/'SPLIT_CONTRACT.json',
             CACHE_RUN/'trace/TRACE_MANIFEST.json', CACHE_RUN/'trace/NATIVE_ARGS.json', GT, PROFILE, WEIGHT,
             DEV_RUN/'SUMMARY.json', DEV_RUN/'B_FIXED_PAIR_EFFECTS.jsonl', DEV_RUN/'EVAL_CONTRACT.json',
             DEV_RUN/'CONFIG_AUDIT.json']
    return {'profile': profile, 'environment': environment, 'training_contract': training,
            'split': split, 'source_matches': source_matches, 'native_argument_differences': differences,
            'inputs': {str(p): sha256(p) for p in files}, 'external_sources': external_sources()}


def damage_census(baseline, method):
    previous = {(r['image_id'], r['annotation_id']): r for r in baseline}
    actual = {(r['image_id'], r['annotation_id']): r for r in method}
    require(len(previous) == len(baseline) and len(actual) == len(method), 'duplicate GT ledger key')
    require(set(previous) == set(actual), 'GT coverage changed')
    require(all(r['is_ignored'] == actual[k]['is_ignored'] for k, r in previous.items()), 'ignore changed')
    success = [k for k, r in previous.items() if not r['is_ignored'] and r['task_mask75'] == 'success']
    require(success, 'no ordinary successful GT for census')
    damaged = [k for k in success if actual[k]['task_mask75'] == 'failure']
    return {'population_N': len(success), 'observed_n': len(success), 'damaged_k': len(damaged),
            'fraction': len(damaged)/len(success), 'finite_population_upper': len(damaged)/len(success),
            'unit': 'fraction_0_to_1', 'method': 'complete_finite_population_census',
            'damaged_GT_keys': [list(k) for k in damaged]}


def mapped_id(oid, global_indices, source_sha):
    return None if oid is None else source_sha+':'+str(global_indices[int(oid.rsplit(':', 1)[1])])


def baseline_to_global(rows, global_indices, source_sha):
    return [{**row, **{name: mapped_id(row[name], global_indices, source_sha)
            for name in ('bbox75_match_output_id', 'segm75_match_output_id')}} for row in rows]


def fixed_effects(baseline, method, global_indices, source_sha):
    actual = {(r['image_id'], r['annotation_id']): r for r in method}
    require(set(actual) == {(r['image_id'], r['annotation_id']) for r in baseline}, 'GT domain mismatch')
    result = []
    for before in baseline:
        after = actual[(before['image_id'], before['annotation_id'])]
        require(after['is_ignored'] == before['is_ignored'], 'ordinary GT domain changed')
        require(mapped_id(after['bbox75_match_output_id'], global_indices, source_sha)
                == before['bbox75_match_output_id'], 'bbox75 detection identity changed')
        if before['is_ignored'] or before['bbox75_match_output_id'] is None:
            continue
        require(after['box_iou_of_bbox_match'] == before['box_iou_of_bbox_match'], 'box IoU changed')
        a, b = before['mask_iou_of_bbox_match'], after['mask_iou_of_bbox_match']
        result.append({'image_id': before['image_id'], 'annotation_id': before['annotation_id'],
            'population': population(before), 'baseline_output_detection_id': before['bbox75_match_output_id'],
            'baseline_iou': a, 'method_iou': b, 'delta_iou': b-a, 'same_output_reaches_075': b >= .75})
    return result


def adjudicate(target, overall, damage, loss_delta):
    gates = {'target_effect': target['mean'] >= .005 and target['ci95'][0] > 0,
             'all_bbox75_guardrail': overall['mean'] >= -.001,
             'success_damage_census': damage['fraction'] <= .01}
    fit_pass = all(gates.values())
    lowered = loss_delta < -1e-6
    return {'fit_diagnostic_gates': gates, 'fit_repair_supported': fit_pass,
            'training_objective_lowered': lowered,
            'localization': 'fit_meets_requirements_dev_does_not' if fit_pass else
                            'repair_requirements_already_unmet_on_fit',
            'next_action': 'Audit the fixed-checkpoint train-to-dev effect gap; no new training' if fit_pass else
                           'Audit training-supervision-to-native-mask-metric correspondence; no new training',
            'prior_S2_negative_decision_retained': True}


def jsonl(path, rows):
    with path.open('x', encoding='utf8') as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False)+'\n')


def checkpoint_model():
    import torch
    from adapter_model import CoefficientAdapter
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    require(saved['arm'] == 'B' and saved['completed_epochs'] == 3, 'wrong checkpoint state')
    require(saved['contract_sha256'] == sha256(B_RUN/'EVAL_CONTRACT.json'), 'checkpoint contract mismatch')
    steps = {int(v['step']) for v in saved['optimizer_state_dict']['state'].values()}
    require(steps == {1500}, 'checkpoint did not have the fixed original step budget')
    model = CoefficientAdapter('B').cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'], strict=True)
    require(tensor_state_hash(model) == read_json(B_RUN/'SUMMARY.json')['final_state_sha256'], 'state identity mismatch')
    require(not any(p.requires_grad for p in model.parameters()), 'diagnostic model not frozen')
    return model


def export(run, source, coco, baseline):
    import torch
    from torch.nn import functional as F
    from adapter_data import training_join, projected_targets
    from adapter_model import loss_grids, vector_grouped_loss
    from native_output_bridge import decode_native, encode_rle
    from s1_geometry import full_rle_iou
    model = checkpoint_model()
    state_before = tensor_state_hash(model)
    stored = read_json(B_RUN/'TRAINING_JOIN.json')['images']
    baseline_by_gt = {(r['image_id'], r['annotation_id']): r for r in baseline}
    output = run.path/'B_predictions.json'
    losses, checks, global_indices = [], [], []
    direction_counts = Counter()
    valid_join_pairs = 0
    start = time.monotonic()
    with output.open('x', encoding='utf8') as predictions, \
         (run.path/'TRAINING_PAIR_EFFECTS.jsonl').open('x', encoding='utf8') as pairlog, \
         torch.inference_mode():
        predictions.write('[')
        comma = False
        for number, image_id in enumerate(source.records, 1):
            run.check_budget()
            cache, original = source.load(image_id)
            c0 = cache['c0'].cuda()
            decoded = decode_native(cache, c0)
            require(all(encode_rle(decoded[i]) == row['segmentation'] for i, row in enumerate(original)),
                    'exact zero-delta native RLE replay failed')
            del decoded
            join = training_join(original, coco.imgToAnns.get(image_id, []))
            expected = {int(k): v for k, v in stored[str(image_id)]['output_rank_to_annotation_id'].items()}
            require(join == expected, 'reconstructed supervision differs from saved training join')
            targets = projected_targets(coco, cache, join)
            delta = model(cache, 'eval:0')
            require(torch.isfinite(delta).all().item() and not delta.requires_grad, 'invalid frozen residual')
            coefficients = c0+delta
            logits, native, roi = loss_grids(cache, coefficients)
            before_loss, before = vector_grouped_loss(native, native, roi, targets)
            after_loss, after = vector_grouped_loss(logits, native, roi, targets)
            for key in ('matched_supervised_instances', 'preserved_instances', 'empty_roi_rows', 'no_output_image'):
                require(before[key] == after[key], 'loss denominator changed')
            losses.append({'image_id': image_id, 'zero': {'total_loss': float(before_loss), **before},
                'B': {'total_loss': float(after_loss), **after}, 'delta_total_loss': float(after_loss-before_loss),
                'delta_gt_loss': after['gt_loss']-before['gt_loss'],
                'delta_preservation_loss': after['preservation_loss']-before['preservation_loss']})
            pixels = roi.sum((1, 2))
            valid = [i for i in targets if int(pixels[i]) > 0]
            if valid:
                target_tensor = torch.stack([targets[i].to(logits) for i in valid])
                bce_a = (F.binary_cross_entropy_with_logits(native[valid], target_tensor, reduction='none')
                         * roi[valid]).sum((1, 2))/pixels[valid]
                bce_b = (F.binary_cross_entropy_with_logits(logits[valid], target_tensor, reduction='none')
                         * roi[valid]).sum((1, 2))/pixels[valid]
                require(abs(float(bce_a.mean())-before['gt_loss']) <= 1e-7, 'GT loss accounting mismatch')
                require(abs(float(bce_b.mean())-after['gt_loss']) <= 1e-7, 'GT loss accounting mismatch')
                bce_by_rank = {rank: (float(bce_a[k]), float(bce_b[k])) for k, rank in enumerate(valid)}
                del target_tensor, bce_a, bce_b
            else:
                bce_by_rank = {}
            del logits, native, roi, pixels, targets
            decoded = decode_native(cache, coefficients)
            changed = [{**row, 'segmentation': encode_rle(decoded[i])} for i, row in enumerate(original)]
            compare_outputs(original, changed)
            for rank in valid:
                aid = join[rank]
                gt_rle = coco.annToRLE(coco.anns[aid])
                iou_a = full_rle_iou(original[rank]['segmentation'], gt_rle)
                iou_b = full_rle_iou(changed[rank]['segmentation'], gt_rle)
                bce_a, bce_b = bce_by_rank[rank]
                oid = source.manifest['predictions_sha256']+':'+str(cache['prediction_indices'][rank])
                ledger_row = baseline_by_gt[(image_id, aid)]
                direction = ('loss_down' if bce_b-bce_a < -1e-6 else 'loss_up' if bce_b-bce_a > 1e-6 else 'loss_tie')
                direction += ('_iou_down' if iou_b-iou_a < -1e-12 else '_iou_up' if iou_b-iou_a > 1e-12 else '_iou_tie')
                direction_counts[direction] += 1
                valid_join_pairs += 1
                pairlog.write(json.dumps({'image_id': image_id, 'annotation_id': aid, 'training_output_id': oid,
                    'baseline_bbox75_output_id': ledger_row['bbox75_match_output_id'],
                    'same_as_bbox75_pair': oid == ledger_row['bbox75_match_output_id'],
                    'baseline_population': population(ledger_row), 'zero_gt_bce': bce_a, 'B_gt_bce': bce_b,
                    'delta_gt_bce': bce_b-bce_a, 'zero_full_iou': iou_a, 'B_full_iou': iou_b,
                    'delta_full_iou': iou_b-iou_a, 'direction': direction}, allow_nan=False)+'\n')
            for row in changed:
                if comma:
                    predictions.write(',')
                predictions.write(json.dumps(row, separators=(',', ':'), allow_nan=False))
                comma = True
            global_indices.extend(cache['prediction_indices'])
            checks.append({'image_id': image_id, 'outputs': len(original), 'exact_zero_delta_RLE': True,
                           'saved_training_join_equal': True, 'nonmask_fields_unchanged': True})
            if number % 50 == 0 or number == len(source.records):
                predictions.flush()
                pairlog.flush()
                run.update(phase='frozen_checkpoint_export', completed_images=number, total_images=len(source.records))
                print('FIT_EXPORT', number, '/', len(source.records), 'seconds', round(time.monotonic()-start, 1), flush=True)
                require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file()) <= 8*1024**3, 'artifact budget exceeded')
            del cache, original, delta, coefficients, decoded, changed, c0
    # Close the prediction array after the inference loop; the stream was opened once with mode x.
    with output.open('a', encoding='utf8') as predictions:
        predictions.write(']\n')
    require(global_indices == sorted(global_indices) and len(set(global_indices)) == len(global_indices), 'output row order changed')
    require(tensor_state_hash(model) == state_before, 'in-memory model changed')
    require(not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'gradient path present')
    jsonl(run.path/'LOSS_BY_IMAGE.jsonl', losses)
    jsonl(run.path/'ZERO_DELTA_AND_JOIN_CHECKS.jsonl', checks)
    write_new(run.path/'ROW_MAPPING.json', {'original_global_indices': global_indices,
               'source_prediction_sha256': source.manifest['predictions_sha256']})
    loss_summary = {arm: {key: sum(row[arm][key] for row in losses)/len(losses)
        for key in ('total_loss', 'gt_loss', 'preservation_loss')} for arm in ('zero', 'B')}
    loss_summary['delta'] = {key: loss_summary['B'][key]-loss_summary['zero'][key]
                            for key in ('total_loss', 'gt_loss', 'preservation_loss')}
    loss_summary.update(images=len(losses), matched_supervised_instances=valid_join_pairs,
        images_without_GT_supervision=sum(r['B']['matched_supervised_instances'] == 0 for r in losses),
        empty_roi_instances=sum(len(r['B']['empty_roi_rows']) for r in losses),
        per_instance_BCE_IoU_direction_counts=dict(direction_counts),
        reduction='separate ROI instance means then equal image mean; no-match images retained')
    require(sum(direction_counts.values()) == sum(r['B']['matched_supervised_instances'] for r in losses), 'loss-pair coverage mismatch')
    state_receipt = {'before': state_before, 'after': tensor_state_hash(model), 'parameters_require_grad': False,
                     'parameter_gradients_present': False, 'new_optimizer_steps': 0, 'model_eval': not model.training}
    write_new(run.path/'FROZEN_MODEL_STATE.json', state_receipt)
    del model
    torch.cuda.empty_cache()
    return output, global_indices, loss_summary, {'images': len(checks), 'outputs': len(global_indices),
        'seconds': time.monotonic()-start, 'zero_delta_exact_outputs': sum(x['outputs'] for x in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('smoke', 'full'))
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        data = inspect_inputs()
        print(json.dumps({'status': 'passed', 'checkpoint_sha256': CHECKPOINT_SHA, 'fit_images': len(data['split']['fit_ids']),
            'cache_manifest_sha256': TRACE_SHA, 'GT_sha256': GT_SHA, 'environment': data['environment'],
            'input_hashes': data['inputs'], 'source_matches': data['source_matches'],
            'native_argument_differences': data['native_argument_differences']}, ensure_ascii=False), flush=True)
        return
    require(args.scope is not None, 'scope required')
    require(not any((PARENT/p).exists() for p in ('.S1.lock', '.S2.lock')), 'another parent stage is active')
    with DiagnosticRun(args.scope) as run:
        data = inspect_inputs()
        ids = data['split']['fit_ids'][:8] if args.scope == 'smoke' else data['split']['fit_ids']
        contract = {'scope': args.scope, 'claim': 'fixed-checkpoint repair and damage census on original training images',
            'arm': 'B', 'checkpoint_sha256': CHECKPOINT_SHA, 'cache_run': CACHE_RUN.name,
            'training_run': B_RUN.name, 'development_result_run': DEV_RUN.name, 'image_ids': ids,
            'inputs_sha256': data['inputs'], 'environment': data['environment'],
            'external_evaluator_sources': data['external_sources'], 'native_argument_differences': data['native_argument_differences'],
            'new_training_steps': 0, 'dtype': 'float32', 'AMP': False, 'phase': 'eval:0',
            'preprocessing': 'retained original per-image native batch16 rect geometry; no image forward',
            'decoder': 'unchanged decode_native full coefficient matrix', 'primary_pairing': 'baseline COCO bbox75 detection',
            'training_pairing': 'original saved class_box50_score_order_one_to_one_ordinary_GT',
            'gates': {'T_delta_min': .005, 'T_CI_lower_strict_min': 0, 'all_bbox75_delta_min': -.001, 'damage_census_max': .01},
            'bootstrap': {'resamples': 2000, 'seed': 1729, 'CI': [.025, .975], 'method': 'percentile_linear'},
            'damage': 'all baseline successful ordinary GT within fit2000; exact finite-population ratio',
            'loss_reduction': 'unchanged per-ROI, separate group mean then image mean; weight=.1',
            'loss_direction_tolerance': 1e-6, 'iou_direction_tolerance': 1e-12,
            'maxDets': [1, 10, 100], 'useCats': 1, 'prediction_area': 'native full JSON bbox area',
            'wall_seconds': run.wall_seconds, 'max_artifact_bytes': 8*1024**3}
        run.freeze(contract)
        if args.scope == 'full':
            require(args.smoke_run is not None, 'completed engineering smoke required')
            require(read_json(args.smoke_run/'run.json')['execution_status'] == 'completed', 'smoke incomplete')
            require(read_json(args.smoke_run/'CONFIG_AUDIT.json')['status'] == 'passed', 'smoke configuration invalid')
            require(read_json(args.smoke_run/'DECISION.json')['status'] == 'passed', 'smoke failed')
            require(read_json(args.smoke_run/'SOURCE_MANIFEST.json')['files'] == run.sources, 'source changed after smoke')
            old = read_json(args.smoke_run/'EVAL_CONTRACT.json')
            require(old['inputs_sha256'] == contract['inputs_sha256'] and old['environment'] == contract['environment'], 'smoke inputs drift')
        write_new(run.path/'INPUT_AUDIT.json', {'status': 'passed', 'input_hashes': data['inputs'],
            'matched_original_sources': data['source_matches'], 'native_argument_differences': data['native_argument_differences']})
        from adapter_data import FrozenCache
        from pycocotools.coco import COCO
        from build_native_ledger import evaluate_ledger
        source = FrozenCache(CACHE_RUN/'trace', ids)
        require(len(source.records) == len(ids), 'image coverage mismatch')
        indices = sorted(source.predictions)
        coco = COCO(str(GT))
        run.update(phase='baseline_GT_ledger', total_images=len(ids))
        with (run.path/'baseline_ledger.log').open('x', encoding='utf8') as log, contextlib.redirect_stdout(log):
            raw_baseline, baseline_ap = evaluate_ledger(GT, [source.predictions[i] for i in indices], ids,
                                                      source.manifest['predictions_sha256'])
        baseline = baseline_to_global(raw_baseline, indices, source.manifest['predictions_sha256'])
        del raw_baseline
        counts = Counter(population(r) for r in baseline)
        write_new(run.path/'BASELINE_POPULATIONS.json', dict(counts))
        jsonl(run.path/'BASELINE_GT_LEDGER.jsonl', baseline)
        if args.scope == 'full':
            require(counts['T'] >= 200, 'insufficient predeclared target population')
        gc.collect()
        output, exported_indices, losses, export_receipt = export(run, source, coco, baseline)
        require(exported_indices == indices, 'method coverage or ordering differs')
        run.update(phase='method_GT_ledger')
        with (run.path/'B_ledger.log').open('x', encoding='utf8') as log, contextlib.redirect_stdout(log):
            method, method_ap = evaluate_ledger(GT, read_json(output), ids, sha256(output))
        jsonl(run.path/'B_GT_LEDGER.jsonl', method)
        effects = fixed_effects(baseline, method, indices, source.manifest['predictions_sha256'])
        require(sum(row['population'] == 'T' for row in effects) == counts['T'], 'target coverage differs')
        jsonl(run.path/'FIXED_PAIR_EFFECTS.jsonl', effects)
        damage = damage_census(baseline, method)
        write_new(run.path/'DAMAGE_CENSUS.json', damage)
        if args.scope == 'full':
            target = bootstrap(image_effects(effects, target_only=True))
            overall = bootstrap(image_effects(effects, target_only=False))
            decision = adjudicate(target, overall, damage, losses['delta']['total_loss'])
        else:
            target = overall = None
            decision = {'engineering_only': True, 'next_action': 'Execute authorized frozen full fit2000 diagnostic'}
        dev_effects = read_jsonl(DEV_RUN/'B_FIXED_PAIR_EFFECTS.jsonl')
        dev = read_json(DEV_RUN/'SUMMARY.json')['B']
        for name, target_only in [('target', True), ('all_bbox75', False)]:
            recalculated = bootstrap(image_effects(dev_effects, target_only=target_only))
            require(abs(recalculated['mean']-dev[name]['mean']) < 1e-14 and
                    all(abs(a-b) < 1e-14 for a, b in zip(recalculated['ci95'], dev[name]['ci95'])), 'prior dev result mismatch')
        run.update(phase='post_run_configuration_audit')
        source.verify()
        require(sha256(source.prediction_path) == source.manifest['predictions_sha256'], 'source prediction changed')
        for path, digest in data['inputs'].items():
            require(sha256(path) == digest, 'frozen input changed: '+path)
        from native_output_bridge import native_environment
        require(native_environment() == data['environment'], 'actual runtime drift')
        require(external_sources() == data['external_sources'], 'evaluator source drift')
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file()) <= 8*1024**3, 'artifact budget exceeded')
        run.check_budget()
        write_new(run.path/'CONFIG_AUDIT.json', {'status': 'passed', 'scope': 'frozen B fit census diagnostic',
            'checks': {'original_checkpoint_state_unchanged': True, 'zero_new_training_steps_and_no_gradients': True,
                'original_cache_P_c0_F_and_geometries_unchanged': True, 'all_zero_delta_RLE_exact': True,
                'all_nonmask_output_fields_and_order_unchanged': True, 'full_image_GT_target_success_coverage': True,
                'original_training_join_and_loss_denominators': True, 'bbox_and_segm_matching_separate': True,
                'bbox75_detection_identity_unchanged': True, 'runtime_and_pinned_sources_unchanged': True,
                'inputs_weights_checkpoints_predictions_SHA_unchanged': True, 'development_result_recomputed_equal': True},
            'contract_sha256': sha256(run.path/'EVAL_CONTRACT.json'), 'input_audit_sha256': sha256(run.path/'INPUT_AUDIT.json'),
            'zero_delta_outputs': export_receipt['zero_delta_exact_outputs'], 'new_optimizer_steps': 0})
        summary = {'scope': args.scope, 'fit_images': len(ids), 'baseline_populations': dict(counts),
            'target': target, 'all_bbox75': overall, 'damage_census': damage, 'fixed_checkpoint_loss': losses,
            'export': export_receipt, 'diagnostic_decision': decision,
            'retained_dev': {'target': dev['target'], 'all_bbox75': dev['all_bbox75'], 'damage': dev['damage']},
            'fit_minus_dev_mean': None if args.scope == 'smoke' else
                {'target': target['mean']-dev['target']['mean'], 'all_bbox75': overall['mean']-dev['all_bbox75']['mean']},
            'secondary_train_AP_only': {'zero': baseline_ap, 'B': method_ap},
            'claim_scope': 'conditional description of original training images; does not reopen failed S2 or prove root cause'}
        scientific = 'engineering_only' if args.scope == 'smoke' else decision['localization']
        run.finish(summary, status='passed', scientific=scientific, next_action=decision['next_action'])
        print('DIAGNOSTIC_COMPLETED', run.run_id, json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
