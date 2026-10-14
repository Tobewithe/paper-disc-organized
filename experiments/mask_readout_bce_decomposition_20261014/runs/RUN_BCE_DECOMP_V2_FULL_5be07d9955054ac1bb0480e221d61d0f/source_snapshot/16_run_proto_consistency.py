"""Frozen B checkpoint, original fit2000: add same-ROI proto binary IoU only."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT/'experiments/mask_readout_proto_consistency_20261014'
FIT_STUDY = ROOT/'experiments/mask_readout_fit_diagnostic_20261014'
FIT_RUN = FIT_STUDY/'runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926'
sys.path.insert(0, str(FIT_STUDY/'scripts'))
import run_fit_diagnostic as prior
from posttrain_run import RegisteredRun, read_json, require, sha256, write_new, tensor_state_hash
from posttrain_audit import read_jsonl
from proto_statistics import pair_key, direction, tensor_counts, summarize, adjudicate

CODE_FILES = list(dict.fromkeys([*prior.CODE_FILES, *[str(p.relative_to(ROOT)).replace('\\', '/') for p in (
    Path(__file__), STUDY/'scripts/proto_statistics.py', STUDY/'scripts/test_proto_consistency.py',
    STUDY/'scripts/launch_audit.ps1', STUDY/'PROTOCOL.md', STUDY/'INPUT_EXPECTATIONS.json',
    STUDY/'SOURCE_CONTEXT.json', STUDY/'SOURCE_DIRTY.patch')]]))


class AuditRun(prior.DiagnosticRun):
    def __init__(self, scope):
        RegisteredRun.__init__(self, 'PROTO_AUDIT_'+scope.upper(), 'PROTO_CONSISTENCY', FIT_RUN,
                               CODE_FILES, 900 if scope == 'smoke' else 1800)
        self.path = STUDY/'runs'/self.run_id
        self.lock = STUDY/'.PROTO_CONSISTENCY.lock'
        self.meta.update(study_id='STUDY_MASK_READOUT_PROTO_CONSISTENCY_20261014', scope=scope,
                         baseline_run_id=prior.BASE_RUN.name, optimization_started=False, optimizer_steps=0)

    def freeze(self, contract):
        context = read_json(STUDY/'SOURCE_CONTEXT.json')
        self.sources = {relative: sha256(ROOT/relative) for relative in self.code_files}
        self.meta.update(code_commit=context['code_commit'], dirty_patch_sha256=sha256(STUDY/'SOURCE_DIRTY.patch'))
        write_new(self.path/'SOURCE_MANIFEST.json', {'files': self.sources, 'code_commit': context['code_commit'],
                                                   'actual_sources_snapshotted': True})
        for i, relative in enumerate(self.sources):
            target = self.path/'source_snapshot'/(f'{i:02d}_'+Path(relative).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
        shutil.copyfile(STUDY/'PROTOCOL.md', self.path/'PROTOCOL.md')
        write_new(self.path/'EVAL_CONTRACT.json', contract)
        self.update(plan_status='frozen', protocol_path=str(self.path/'PROTOCOL.md'),
                    protocol_sha256=sha256(self.path/'PROTOCOL.md'), contract_sha256=sha256(self.path/'EVAL_CONTRACT.json'))


def inspect_inputs():
    data = prior.inspect_inputs()
    expected = read_json(STUDY/'INPUT_EXPECTATIONS.json')
    for relative, digest in expected['files'].items():
        require(sha256(ROOT/relative) == digest, 'prior evidence identity changed: '+relative)
        data['inputs'][str(ROOT/relative)] = digest
    require(read_json(FIT_RUN/'run.json')['execution_status'] == 'completed', 'fit diagnostic incomplete')
    require(read_json(FIT_RUN/'CONFIG_AUDIT.json')['status'] == 'passed', 'fit diagnostic invalid')
    require(read_json(FIT_RUN/'EVAL_CONTRACT.json')['environment'] == data['environment'], 'fit runtime differs')
    for relative, digest in read_json(FIT_RUN/'SOURCE_MANIFEST.json')['files'].items():
        if relative in CODE_FILES:
            require(sha256(ROOT/relative) == digest, 'fit source changed: '+relative)
    pairs = read_jsonl(FIT_RUN/'TRAINING_PAIR_EFFECTS.jsonl')
    require(len(pairs) == len({pair_key(r) for r in pairs}) == 14268, 'prior supervision pair coverage changed')
    require(sum(r['direction'] == 'loss_down_iou_down' for r in pairs) == 3950, 'prior discrepancy population changed')
    checks = read_jsonl(FIT_RUN/'ZERO_DELTA_AND_JOIN_CHECKS.jsonl')
    require(len(checks) == 2000 and sum(r['outputs'] for r in checks) == 218209, 'prior output coverage changed')
    require(all(r['exact_zero_delta_RLE'] and r['saved_training_join_equal'] and r['nonmask_fields_unchanged']
                for r in checks), 'prior replay or pairing failed')
    return data


def close(a, b, message):
    error = abs(float(a)-float(b))
    require(error <= 1e-7, message)
    return error


def measure(run, source, coco, old_pairs, old_losses, old_checks):
    import torch
    from torch.nn import functional as F
    from adapter_data import training_join, projected_targets
    from adapter_model import loss_grids, vector_grouped_loss
    model = prior.checkpoint_model()
    state_before = tensor_state_hash(model)
    stored = read_json(prior.B_RUN/'TRAINING_JOIN.json')['images']
    records, images, global_indices = [], [], []
    selected_pairs = {pair_key(r): r for r in old_pairs if r['image_id'] in source.records}
    losses_by_id = {r['image_id']: r for r in old_losses}
    expected_checks = {r['image_id']: r for r in old_checks}
    max_bce_error = max_loss_error = 0.
    start = time.monotonic()
    with (run.path/'PAIR_PROTO_AUDIT.jsonl').open('x', encoding='utf8') as out, torch.inference_mode():
        import json
        for number, image_id in enumerate(source.records, 1):
            run.check_budget()
            cache, original = source.load(image_id)
            require(len(original) == expected_checks[image_id]['outputs'], 'output coverage differs from endpoint Run')
            join = training_join(original, coco.imgToAnns.get(image_id, []))
            expected = {int(k): v for k, v in stored[str(image_id)]['output_rank_to_annotation_id'].items()}
            require(join == expected, 'saved supervision pairing changed')
            targets = projected_targets(coco, cache, join)
            delta = model(cache, 'eval:0')
            require(not delta.requires_grad and torch.isfinite(delta).all().item(), 'invalid frozen residual')
            logits, native, roi = loss_grids(cache, cache['c0'].cuda()+delta)
            zero_total, zero = vector_grouped_loss(native, native, roi, targets)
            B_total, B = vector_grouped_loss(logits, native, roi, targets)
            zero['total_loss'], B['total_loss'] = float(zero_total), float(B_total)
            previous = losses_by_id[image_id]
            for arm, actual in [('zero', zero), ('B', B)]:
                for key in ('gt_loss', 'preservation_loss', 'total_loss'):
                    max_loss_error = max(max_loss_error, close(actual[key], previous[arm][key], 'per-image loss changed'))
                for key in ('matched_supervised_instances', 'preserved_instances', 'empty_roi_rows', 'no_output_image'):
                    require(actual[key] == previous[arm][key], 'loss denominator changed')
            pixels = roi.sum((1, 2))
            valid = [i for i in targets if int(pixels[i]) > 0]
            if valid:
                target_tensor = torch.stack([targets[i].to(logits) for i in valid])
                bce_a = (F.binary_cross_entropy_with_logits(native[valid], target_tensor, reduction='none')
                         * roi[valid]).sum((1, 2))/pixels[valid]
                bce_b = (F.binary_cross_entropy_with_logits(logits[valid], target_tensor, reduction='none')
                         * roi[valid]).sum((1, 2))/pixels[valid]
                before_counts = tensor_counts(native[valid], target_tensor, roi[valid])
                after_counts = tensor_counts(logits[valid], target_tensor, roi[valid])
                for k, rank in enumerate(valid):
                    oid = source.manifest['predictions_sha256']+':'+str(cache['prediction_indices'][rank])
                    key = (image_id, join[rank], oid)
                    require(key in selected_pairs, 'new or changed supervision identity')
                    previous_pair = selected_pairs[key]
                    a, b = float(bce_a[k]), float(bce_b[k])
                    max_bce_error = max(max_bce_error, close(a, previous_pair['zero_gt_bce'], 'zero BCE changed'),
                                        close(b, previous_pair['B_gt_bce'], 'B BCE changed'))
                    expected_direction = 'loss_'+direction(b-a, 1e-6)+'_iou_'+direction(previous_pair['delta_full_iou'])
                    require(expected_direction == previous_pair['direction'], 'prior direction not reproduced')
                    require(before_counts[k]['gt_pixels'] == after_counts[k]['gt_pixels'] and
                            before_counts[k]['roi_pixels'] == after_counts[k]['roi_pixels'], 'ROI or GT changed between arms')
                    effect = after_counts[k]['iou']-before_counts[k]['iou']
                    row = {**previous_pair, 'output_rank': rank, 'recomputed_zero_gt_bce': a, 'recomputed_B_gt_bce': b,
                           'zero_proto': before_counts[k], 'B_proto': after_counts[k], 'delta_proto_iou': effect,
                           'proto_direction': direction(effect), 'proto_hw': list(cache['proto'].shape[-2:]),
                           'input_hw': cache['input_hw'], 'original_hw': cache['original_hw']}
                    out.write(json.dumps(row, allow_nan=False)+'\n')
                    records.append(row)
                del target_tensor, bce_a, bce_b, before_counts, after_counts
            images.append({'image_id': image_id, 'outputs': len(original), 'valid_supervision_pairs': len(valid),
                           'zero': zero, 'B': B, 'saved_training_join_equal': True,
                           'prior_endpoint_output_coverage_equal': True})
            global_indices.extend(cache['prediction_indices'])
            if number % 50 == 0 or number == len(source.records):
                out.flush()
                run.update(phase='proto_same_ROI_measurement', completed_images=number, total_images=len(source.records))
                print('PROTO_AUDIT', number, '/', len(source.records), 'seconds', round(time.monotonic()-start, 1), flush=True)
            del cache, original, targets, delta, logits, native, roi, pixels, zero_total, B_total
    require(len(records) == len({pair_key(r) for r in records}) == len(selected_pairs), 'supervision coverage incomplete')
    require({pair_key(r) for r in records} == set(selected_pairs), 'supervision identity differs')
    expected_indices = set(source.predictions)
    require(global_indices == sorted(expected_indices), 'native output order changed')
    require(tensor_state_hash(model) == state_before, 'model state changed')
    require(not model.training and not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'gradient path active')
    prior.jsonl(run.path/'IMAGE_CHECKS.jsonl', images)
    write_new(run.path/'FROZEN_MODEL_STATE.json', {'before': state_before, 'after': tensor_state_hash(model),
        'parameters_require_grad': False, 'parameter_gradients_present': False, 'new_optimizer_steps': 0, 'model_eval': True})
    del model
    torch.cuda.empty_cache()
    return records, {'images': len(images), 'outputs': len(global_indices), 'max_pair_BCE_abs_error': max_bce_error,
                     'max_image_loss_abs_error': max_loss_error, 'seconds': time.monotonic()-start,
                     'empty_roi_instances': sum(len(r['B']['empty_roi_rows']) for r in images),
                     'images_without_GT_supervision': sum(r['valid_supervision_pairs'] == 0 for r in images)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('smoke', 'full'))
    parser.add_argument('--smoke-run', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        data = inspect_inputs()
        print('PREFLIGHT_PASSED', 'fit_images', len(data['split']['fit_ids']), 'verified_inputs', len(data['inputs']), flush=True)
        return
    require(args.scope, 'scope required')
    require(not any((prior.PARENT/p).exists() for p in ('.S1.lock', '.S2.lock')) and
            not (FIT_STUDY/'.FIT_DIAGNOSTIC.lock').exists(), 'another parent diagnostic or stage active')
    with AuditRun(args.scope) as run:
        data = inspect_inputs()
        ids = data['split']['fit_ids'][:8] if args.scope == 'smoke' else data['split']['fit_ids']
        contract = {'scope': args.scope, 'claim': 'locate observed BCE/binary-IoU discordance using same-proto-ROI binary masks',
            'source_fit_run': FIT_RUN.name, 'arm': 'B', 'checkpoint_sha256': prior.CHECKPOINT_SHA,
            'cache_run': prior.CACHE_RUN.name, 'image_ids': ids, 'inputs_sha256': data['inputs'],
            'environment': data['environment'], 'external_evaluator_sources': data['external_sources'],
            'native_argument_differences': data['native_argument_differences'],
            'new_training_steps': 0, 'dtype': 'float32', 'AMP': False, 'phase': 'eval:0',
            'preprocessing': 'retained original per-image native batch16 rect geometry; no base-model image forward',
            'matrix_shape_policy': 'all original final outputs before selecting supervision rows',
            'training_pairing': 'unchanged saved class_box50_score_order_one_to_one_ordinary_GT',
            'ROI': 'unchanged loss_grids left/top inclusive right/bottom exclusive proto pixel coordinates',
            'GT_projection': 'unchanged projected_targets/project_gt from frozen original annotations',
            'binary_prediction': 'strict logit>0', 'empty_union_iou': 1, 'empty_roi': 'excluded_and_counted_as_in_training',
            'endpoint_policy': 'reuse hash-verified prior full-image IoU and exact native replay; no new COCO evaluation',
            'loss_direction_tolerance': 1e-6, 'iou_direction_tolerance': 1e-12, 'BCE_replay_abs_tolerance': 1e-7,
            'expected_full_pairs': 14268, 'expected_full_discrepant_pairs': 3950, 'early_majority_strict_threshold': .5,
            'wall_seconds': run.wall_seconds, 'max_artifact_bytes': 1024**3}
        run.freeze(contract)
        if args.scope == 'full':
            require(args.smoke_run is not None, 'completed engineering smoke required')
            for name, key, value in [('run.json', 'execution_status', 'completed'), ('CONFIG_AUDIT.json', 'status', 'passed'),
                                     ('DECISION.json', 'status', 'passed')]:
                require(read_json(args.smoke_run/name)[key] == value, 'smoke not passed')
            require(read_json(args.smoke_run/'SOURCE_MANIFEST.json')['files'] == run.sources, 'source changed after smoke')
            old = read_json(args.smoke_run/'EVAL_CONTRACT.json')
            require(old['inputs_sha256'] == contract['inputs_sha256'] and old['environment'] == contract['environment'], 'smoke drift')
        write_new(run.path/'INPUT_AUDIT.json', {'status': 'passed', 'input_hashes': data['inputs'],
            'matched_original_sources': data['source_matches'], 'native_argument_differences': data['native_argument_differences']})
        from adapter_data import FrozenCache
        from pycocotools.coco import COCO
        source = FrozenCache(prior.CACHE_RUN/'trace', ids)
        mapping = read_json(FIT_RUN/'ROW_MAPPING.json')
        require(mapping['source_prediction_sha256'] == source.manifest['predictions_sha256'], 'endpoint prediction identity differs')
        if args.scope == 'full':
            require(mapping['original_global_indices'] == sorted(source.predictions), 'endpoint output order differs')
        records, measurement = measure(run, source, COCO(str(prior.GT)), read_jsonl(FIT_RUN/'TRAINING_PAIR_EFFECTS.jsonl'),
                                      read_jsonl(FIT_RUN/'LOSS_BY_IMAGE.jsonl'), read_jsonl(FIT_RUN/'ZERO_DELTA_AND_JOIN_CHECKS.jsonl'))
        summary = summarize(records)
        if args.scope == 'full':
            require(summary['supervised_pairs'] == 14268 and summary['known_discrepant_pairs'] == 3950, 'formal domain changed')
            decision = adjudicate(summary)
        else:
            decision = {'engineering_only': True, 'next_action': 'Execute authorized frozen full fit2000 proto consistency audit'}
        run.update(phase='post_run_configuration_audit')
        source.verify()
        require(sha256(source.prediction_path) == source.manifest['predictions_sha256'], 'source prediction changed')
        for path, digest in data['inputs'].items():
            require(sha256(path) == digest, 'input changed: '+path)
        from native_output_bridge import native_environment
        require(native_environment() == data['environment'], 'runtime drift')
        require(prior.external_sources() == data['external_sources'], 'external source drift')
        run.check_budget()
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file()) <= 1024**3, 'artifact budget exceeded')
        write_new(run.path/'CONFIG_AUDIT.json', {'status': 'passed', 'new_optimizer_steps': 0,
            'checks': {'checkpoint_and_inmemory_state_unchanged': True, 'all_cache_P_c0_F_and_geometry_hashes_unchanged': True,
                'runtime_packages_and_source_hashes_unchanged': True, 'saved_training_join_exact': True,
                'all_prior_supervision_pairs_covered': True, 'prior_BCE_and_all_image_losses_reproduced': True,
                'original_GT_projection_ROI_and_loss_reduction': True, 'full_output_matrix_and_order_retained': True,
                'all_prior_endpoint_evidence_and_prediction_hashes_unchanged': True, 'zero_new_training_steps_and_no_gradients': True},
            'max_pair_BCE_abs_error': measurement['max_pair_BCE_abs_error'],
            'max_image_loss_abs_error': measurement['max_image_loss_abs_error'],
            'endpoint_replay_scope': 'prior 218209 exact zero-delta RLE replay reused by SHA; no new RLE replay',
            'contract_sha256': sha256(run.path/'EVAL_CONTRACT.json'), 'input_audit_sha256': sha256(run.path/'INPUT_AUDIT.json')})
        run.finish({'scope': args.scope, **summary, 'measurement': measurement, 'diagnostic_decision': decision,
                    'claim_scope': 'fixed fit population stage localization; current S2 remains rejected'},
                   scientific='engineering_only' if args.scope == 'smoke' else decision['localization'], next_action=decision['next_action'])
        print('PROTO_AUDIT_COMPLETED', run.run_id, summary['discrepant_proto_directions'], flush=True)


if __name__ == '__main__':
    main()
