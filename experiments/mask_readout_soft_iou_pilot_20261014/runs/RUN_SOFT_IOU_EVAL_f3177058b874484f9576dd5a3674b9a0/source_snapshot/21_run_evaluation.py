"""One frozen soft-IoU B checkpoint, unchanged S2 endpoints and damage design."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import subprocess
import sys
from pilot_common import *
from posttrain_audit import read_jsonl, population
from soft_iou_loss import LOSS_SPEC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--retry-of', type=Path)
    args = parser.parse_args()
    training_run = args.training_run.resolve()
    require(training_run.is_relative_to((STUDY/'runs').resolve()), 'training from another study')
    for name in ('DECISION.json', 'CONFIG_AUDIT.json'):
        require(read_json(training_run/name)['status'] == 'passed', 'training unverified')
    with PilotRun('EVAL', training_run, 3600) as run:
        data = inspect_inputs()
        from audit_s2_completed import audit
        audit(training_run)
        train_contract = read_json(training_run/'EVAL_CONTRACT.json')
        require(train_contract['gt_loss'] == LOSS_SPEC and train_contract['scope'] == 'full', 'wrong loss or budget')
        for key, value in data['original_contract'].items():
            require(train_contract[key] == value, 'training contract differs: '+key)
        for name in ('initial.pt', 'IMAGE_ORDERS.json', 'TRAINING_JOIN.json'):
            require(sha256(training_run/name) == sha256(B_RUN/name), 'original training input differs: '+name)
        checkpoints = {'B': {'run_id': training_run.name, 'path': str(training_run/'last.pt'),
            'sha256': sha256(training_run/'last.pt'), 'training_contract_sha256': sha256(training_run/'EVAL_CONTRACT.json')}}
        require(checkpoints['B']['sha256'] == read_json(training_run/'SUMMARY.json')['checkpoint_sha256'], 'wrong checkpoint')
        old = read_json(DEV_RUN/'EVAL_CONTRACT.json')
        evaluation = copy.deepcopy(old)
        evaluation.update(checkpoints=checkpoints, control_arm=None,
            inputs_sha256=data['inputs_sha256'], external_evaluator_sources=data['external_evaluator_sources'],
            gt_loss=LOSS_SPEC, comparison_BCE_run=DEV_RUN.name,
            comparison_scope='same_dev_fixed_pairs_descriptive; gates_against_native_S0',
            source_training_independent_audit_sha256=sha256(training_run/'INDEPENDENT_CONFIG_AUDIT.json'))
        unchanged = {k: evaluation[k] == v for k, v in old.items() if k not in ('checkpoints', 'control_arm')}
        require(all(unchanged.values()), 'original evaluation configuration drift')
        run.freeze(evaluation)
        require(read_json(training_run/'SOURCE_MANIFEST.json')['files'] == run.sources, 'source changed after training')
        write_new(run.path/'ORIGINAL_EVALUATION_CONFIG_COMPARE.json', {'status': 'passed',
            'original_field_equality': unchanged, 'changes': ['new B checkpoint', 'no new C candidate'],
            'same_damage_design_new_single_draw_after_checkpoint_freeze': True})
        with (run.path/'evaluation_tests.log').open('x', encoding='utf8') as stream:
            tested = subprocess.run([sys.executable, '-B', str(PARENT/'scripts/test_s2_evaluation.py')], stdout=stream, stderr=subprocess.STDOUT)
        require(tested.returncode == 0, 'original evaluation tests failed')
        baseline = read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')
        from damage_acceptance import select_damage_panel, score_damage
        prior_panels = [p for p in (STUDY/'runs').glob('RUN_SOFT_IOU_EVAL_*/DAMAGE_PANEL.json')
                        if p.parent != run.path and read_json(p)['checkpoint_sha256'] == checkpoints['B']['sha256']]
        if args.retry_of is not None:
            require(read_json(args.retry_of/'run.json')['execution_status'] == 'error', 'retry only retained engineering failure')
            require(sha256(args.retry_of/'EVAL_CONTRACT.json') == sha256(run.path/'EVAL_CONTRACT.json'), 'retry contract differs')
            panel = read_json(args.retry_of/'DAMAGE_PANEL.json')
            require(all(read_json(p) == panel for p in prior_panels), 'inconsistent prior sample')
            run.update(retry_of=args.retry_of.name, damage_panel_reused=True)
        else:
            require(not prior_panels, 'checkpoint already has a damage draw; reuse it in engineering recovery')
            panel = select_damage_panel(baseline, evaluation['ledger_sha256'], checkpoints['B']['sha256'], sha256(run.path/'EVAL_CONTRACT.json'), 1024)
        write_new(run.path/'DAMAGE_PANEL.json', panel)
        run.update(phase='damage_panel_frozen', damage_panel_sha256=sha256(run.path/'DAMAGE_PANEL.json'))
        dev_ids = set(data['split']['dev_ids'])
        ids = sorted(dev_ids | set(panel['image_ids']))
        write_new(run.path/'EVALUATION_IMAGES.json', ids)
        from adapter_data import FrozenCache
        from pycocotools.coco import COCO
        source = FrozenCache(BASE/'trace', ids)
        coco = COCO(str(GT))
        expected_T = {(r['image_id'], r['annotation_id']) for r in baseline if r['image_id'] in dev_ids and population(r) == 'T'}
        require(len(expected_T) == 655, 'dev target identity changed')
        prediction_path, effects, global_indices, export_stats = original_evaluation.export_and_measure(
            run, 'B', training_run/'last.pt', source, baseline, dev_ids, coco)
        require({(r['image_id'], r['annotation_id']) for r in effects if r['population'] == 'T'} == expected_T, 'target coverage differs')
        target = original_evaluation.bootstrap(original_evaluation.image_effects(effects))
        overall = original_evaluation.bootstrap(original_evaluation.image_effects(effects, target_only=False))
        run.update(phase='COCO_task_ledger', arm='B')
        sys.path.insert(0, str(ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts'))
        from build_native_ledger import evaluate_ledger
        with (run.path/'B_ledger.log').open('x', encoding='utf8') as stream, contextlib.redirect_stdout(stream):
            method, ap = evaluate_ledger(GT, read_json(prediction_path), ids, sha256(prediction_path))
        wanted = {(r['image_id'], r['annotation_id']): r for r in baseline if r['image_id'] in set(ids)}
        require({(r['image_id'], r['annotation_id']) for r in method} == set(wanted), 'method GT coverage changed')
        for row in method:
            previous = wanted[(row['image_id'], row['annotation_id'])]
            require(row['is_ignored'] == previous['is_ignored'], 'ignore policy changed')
            oid = row['bbox75_match_output_id']
            mapped = None if oid is None else source.manifest['predictions_sha256']+':'+str(global_indices[int(oid.rsplit(':', 1)[1])])
            require(mapped == previous['bbox75_match_output_id'], 'bbox detection identity changed')
        damage = score_damage(panel, baseline, [r for r in method if r['image_id'] in set(panel['image_ids'])])
        jsonl(run.path/'B_GT_LEDGER.jsonl', method)
        jsonl(run.path/'BASELINE_SUBSET_GT_LEDGER.jsonl', wanted.values())
        main = {'target': target, 'all_bbox75': overall, 'damage': damage, 'export': export_stats,
                'secondary_AP_on_union_dev_and_damage_images': ap, 'prediction_sha256': sha256(prediction_path)}
        write_new(run.path/'B_RESULTS.json', main)
        old_effects = read_jsonl(DEV_RUN/'B_FIXED_PAIR_EFFECTS.jsonl')
        keyed = lambda rows: {(r['image_id'], r['annotation_id'], r['baseline_output_detection_id']): r for r in rows}
        old_keys, new_keys = keyed(old_effects), keyed(effects)
        require(set(old_keys) == set(new_keys), 'BCE/soft-IoU dev fixed pair identity differs')
        require(all(r['baseline_iou'] == old_keys[k]['baseline_iou'] and r['population'] == old_keys[k]['population'] for k, r in new_keys.items()), 'comparison baseline changed')
        old_image = original_evaluation.image_effects(old_effects)
        new_image = original_evaluation.image_effects(effects)
        contrast = original_evaluation.bootstrap({i: new_image[i]-old_image[i] for i in new_image})
        gates = {'target_effect': target['mean'] >= .005 and target['ci95'][0] > 0,
                 'all_bbox75_guardrail': overall['mean'] >= -.001,
                 'success_damage': damage['gate'] == 'damage_guardrail_passed'}
        status = 'passed' if all(gates.values()) else ('blocked' if gates['target_effect'] and gates['all_bbox75_guardrail'] and damage['gate'] == 'blocked_insufficient_precision' else 'failed')
        source.verify()
        verify_after(data)
        require(sha256(training_run/'last.pt') == checkpoints['B']['sha256'], 'checkpoint changed')
        require(panel['checkpoint_sha256'] == checkpoints['B']['sha256'] and panel['inference_contract_sha256'] == sha256(run.path/'EVAL_CONTRACT.json'), 'damage binding changed')
        write_new(run.path/'CONFIG_AUDIT.json', {'status': 'passed', 'scope': 'soft-IoU checkpoint at original S2 gates',
            'checks': {'training_checkpoint_log_independent_audit': True, 'all_original_evaluation_fields_equal': True,
                'same_dev_fixed_pair_identity': True, 'same_native_P_c0_F_geometry_and_decode': True,
                'boxes_classes_scores_order_unchanged': True, 'bbox_COCO_match_identity_unchanged': True,
                'complete_same_image_GT_ignore_policy': True, 'fixed_checkpoint_single_damage_draw': True,
                'frozen_inputs_environment_code_unchanged': True},
            'checkpoint_sha256': checkpoints['B']['sha256'], 'contract_sha256': sha256(run.path/'EVAL_CONTRACT.json'),
            'damage_panel_sha256': sha256(run.path/'DAMAGE_PANEL.json')})
        run.check_budget()
        run.finish({'gates': gates, 'B': main, 'soft_IoU_minus_BCE_target': contrast, 'target_GT': len(expected_T),
                    'scientific_scope': 'fixed native configuration COCO val2017 development; not independent confirmation'},
                   status=status, scientific='S2_'+status,
                   next_action='S3 full 5000 with exactly this checkpoint; do not auto-start in this task' if status == 'passed' else
                               'STOP this training configuration; next same-checkpoint same-ROI-to-original paired diagnostic; no new training')
        print('SOFT_IOU_S2_DECISION', status, json.dumps({'gates': gates, 'target': target, 'overall': overall, 'damage': damage}), flush=True)


if __name__ == '__main__':
    main()
