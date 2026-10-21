"""Frozen five-method paired COCO bootstrap and complete-study adjudication."""
from __future__ import annotations
import argparse
import contextlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

from joint_data import ROOT, read, write, sha, require
from joint_runner import Run, compare_ledgers
sys.path.insert(0, str(ROOT / 'experiments/mask_gate_s3r_20261015/scripts'))
from coco_bootstrap import evaluate, compact, save_cache, resampled_evaluator, ap, verify_replication

METHODS = ('Ns', 'G_old_s', 'G_budget', 'C_native', 'PC_native')
CACHES = None
DRAWS = None

def replicate(index):
    return index, [ap(resampled_evaluator(c, DRAWS[index])) for c in CACHES]

def ledger(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def main():
    global CACHES, DRAWS
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    require(1 <= args.workers <= 4, 'frozen statistics worker maximum is four')
    with Run(args.config, 'statistics') as run:
        require(np.__version__ == '2.4.3', 'NumPy configuration changed')
        prepared, native = run.pointer('PREPARE'), run.pointer('NATIVE')
        paths = {'Ns': native, 'G_old_s': native}
        paths.update({m: run.pointer('EVAL_' + m) for m in METHODS[2:]})
        summaries = {m: read(paths[m] / (m + '_METRICS.json')) for m in METHODS}
        ids = read(prepared / 'DATA_CONTRACT.json')['val_ids']
        require(len(ids) == 5000, 'full 5000 image population required')
        dataset = read(run.config['val_annotations'])
        original_ledger = ledger(native / 'Ns_PER_GT.jsonl')
        success = {r['annotation_id'] for r in original_ledger if r['task_mask75'] == 'success'}
        historical = ledger(run.config['historical_native_ledger'])
        historical_success = {r['annotation_id'] for r in historical if r['task_mask75'] == 'success'}
        require(len(historical_success) == 22957, 'historical anchor ledger changed')
        write(run.path / 'NATIVE_LEDGER_MIGRATION.json', {
            'historical_success': len(historical_success), 'server_success': len(success),
            'lost_success_ids': sorted(historical_success-success),
            'gained_success_ids': sorted(success-historical_success),
            'current_damage_denominator': len(success), 'complete_current_census': True})
        current_ledger = {}
        damage = {}
        original_predictions = read(native / 'Ns_predictions.json')
        identity = [{k: v for k, v in p.items() if k != 'segmentation'} for p in original_predictions]
        del original_predictions
        id_to_pos = {iid: i for i, iid in enumerate(ids)}
        crowd_image = next(a['image_id'] for a in dataset['annotations'] if a.get('iscrowd') and a['image_id'] in id_to_pos)
        verify_draws = [0, 0, id_to_pos[crowd_image], 123, 999, 4999]
        CACHES, checks = [], {}
        for name in METHODS:
            run.check()
            predictions_path = paths[name] / (name + '_predictions.json')
            require(sha(predictions_path) == summaries[name]['predictions_sha256'], 'prediction SHA drift: ' + name)
            predictions = read(predictions_path)
            require([{k: v for k, v in p.items() if k != 'segmentation'} for p in predictions] == identity,
                    'full output identity drift: ' + name)
            require(summaries[name]['metrics']['bbox']['AP'] == summaries['Ns']['metrics']['bbox']['AP'], 'bbox AP changed')
            with (run.path / (name + '_bootstrap_cache.log')).open('x', encoding='utf-8') as log, contextlib.redirect_stdout(log):
                ev = evaluate(run.config['val_annotations'], predictions, ids, ap_only=True)
                require(abs(ap(ev)-summaries[name]['metrics']['segm']['AP']) < 1e-10, 'full/compact COCO AP mismatch')
                cache = compact(ev)
                checks[name] = verify_replication(cache, dataset, predictions, verify_draws)
            save_cache(run.path / (name + '_COCO_CACHE.pkl'), cache)
            CACHES.append(cache)
            current_ledger[name] = ledger(paths[name] / (name + '_PER_GT.jsonl'))
            damaged, repaired = compare_ledgers(original_ledger, current_ledger[name])
            damage[name] = {'damaged_annotation_ids': damaged, 'repaired_annotation_ids': repaired,
                            'native_success_denominator': len(success), 'damaged': len(damaged),
                            'repaired': len(repaired), 'damage_fraction': len(damaged)/len(success)}
            del predictions, ev
            run.update(phase='bootstrap_cache_and_literal_duplicate_verification', method=name)
        write(run.path / 'BOOTSTRAP_LITERAL_VERIFICATION.json', checks)
        write(run.path / 'DAMAGE_CENSUS.json', damage)
        config = run.config['bootstrap']
        require(config['replicates'] == 2000 and config['seed'] == 20261018, 'bootstrap changed')
        DRAWS = np.random.default_rng(config['seed']).integers(0, len(ids), size=(2000,len(ids)), dtype=np.int32)
        np.save(run.path / 'PAIRED_IMAGE_DRAWS.npy', DRAWS, allow_pickle=False)
        values = np.full((2000, 5), np.nan, dtype=np.float64)
        completed = 0
        # Linux fork shares immutable cached match arrays. Each child independently
        # reruns original COCO accumulation on the same image draw for every method.
        with (run.path / 'BOOTSTRAP_PROGRESS.jsonl').open('x', encoding='utf-8') as progress:
            with mp.get_context('fork').Pool(args.workers) as pool:
                for index, result in pool.imap_unordered(replicate, range(2000), chunksize=1):
                    run.check()
                    values[index] = result
                    progress.write(json.dumps({'replicate': index, 'AP_points': result}) + '\n')
                    completed += 1
                    if completed % 25 == 0:
                        progress.flush()
                        run.update(phase='paired_bootstrap', completed_replicates=completed, total_replicates=2000)
                        print('BOOTSTRAP', completed, 2000, flush=True)
        require(np.isfinite(values).all(), 'incomplete paired AP bootstrap')
        np.save(run.path / 'BOOTSTRAP_AP_POINTS.npy', values, allow_pickle=False)
        point = {m: summaries[m]['metrics']['segm']['AP'] for m in METHODS}
        increments = {}
        for name in METHODS[1:]:
            j = METHODS.index(name)
            for reference in ('Ns','G_old_s'):
                if name == reference: continue
                r = METHODS.index(reference)
                increments[name + '_minus_' + reference] = {
                    'AP_points': point[name] - point[reference],
                    'ci95': np.percentile(values[:,j]-values[:,r], [2.5,97.5]).tolist()}
        tests = {}
        for reference, threshold in [('Ns',.5),('G_old_s',.3),('G_budget',.2),('C_native',.2)]:
            r = METHODS.index(reference)
            interval = np.percentile(values[:,4]-values[:,r], [2.5,97.5]).tolist()
            delta = point['PC_native']-point[reference]
            tests['PC_native_minus_' + reference] = {'AP_points': delta, 'ci95': interval,
                'minimum_AP_points': threshold, 'passed': delta >= threshold and interval[0] > 0}
        pc_eval = read(paths['PC_native'] / 'SUMMARY.json')
        guards = {'native_success_damage': damage['PC_native']['damage_fraction'] <= .01,
                  'all_output_identity': True,
                  'inference_cost': bool(pc_eval['inference_cost']['cost_guard_passed'])}
        passed = all(t['passed'] for t in tests.values()) and all(guards.values())
        if passed:
            decision = 'joint_adaptation_supported_as_practical_base_pending_external_confirmation_and_capacity_controls'
            next_action = 'design_separate_external_confirmation_and_capacity_controls_no_automatic_new_training'
        elif (all(guards.values()) and tests['PC_native_minus_Ns']['passed']
              and tests['PC_native_minus_G_old_s']['passed']
              and not all(tests[k]['passed'] for k in ('PC_native_minus_G_budget','PC_native_minus_C_native'))):
            decision = 'native_and_old_method_gain_retained_joint_extra_value_not_established'
            next_action = 'retain_actual_best_valid_baseline_and_close_extra_Proto_investment_in_this_configuration'
        else:
            decision = 'primary_joint_configuration_does_not_pass_frozen_investment_gate'
            next_action = 'stop_additional_training_for_this_configuration_and_consolidate_best_valid_method_evidence'
        additional_lost, additional_saved = compare_ledgers(current_ledger['G_old_s'], current_ledger['PC_native'])
        summary = {'methods': METHODS, 'images':5000, 'ordinary_gt':summaries['Ns']['ordinary_gt'],
            'native_success_denominator':len(success), 'Mask_AP_points':point,
            'relative_to_Ns_and_existing_G':increments, 'primary_comparisons':tests,
            'PC_vs_G_old_task_success_change': {'lost_annotation_ids':additional_lost, 'gained_annotation_ids':additional_saved,
                'difference_in_native_damage_count':damage['PC_native']['damaged']-damage['G_old_s']['damaged']},
            'guards': guards, 'failed_guards':[key for key,value in guards.items() if not value],
            'failed_comparisons':[key for key,value in tests.items() if not value['passed']],
            'primary_gate_passed':passed, 'decision':decision, 'unique_next_action':next_action,
            'bootstrap':{'replicates':2000, 'seed':20261018, 'common_draws_sha256':sha(run.path/'PAIRED_IMAGE_DRAWS.npy'),
                         'interval':'pointwise paired percentile 95%, not simultaneous familywise confidence'},
            'additional_training_steps':0, 'scope':'same-server development COCO, one seed; not independent external confirmation',
            'no_automatic_followup_training':True}
        write(run.path / 'DECISION.json', summary)
        run.finish(summary, decision)

if __name__ == '__main__':
    main()
