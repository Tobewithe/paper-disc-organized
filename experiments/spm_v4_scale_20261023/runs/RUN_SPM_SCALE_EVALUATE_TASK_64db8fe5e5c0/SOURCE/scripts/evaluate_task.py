import argparse
from collections import defaultdict
import contextlib
import gc
import io
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from common import dump, read, sha
from data import Store
from evaluate import evaluate, predict, score
from task_context import task_context
from task_model import load, memory_module

MODES = ['baseline', 'continued', 'adapter', 'random_memory', 'task_spm']
INTERVENTIONS = ['off', 'permute_channels', 'shuffle_spatial', 'random_dictionary']


def folder(root, stage):
    return root/'runs'/read(root/(stage.upper()+'_POINTER.json'))['run_id']


def source_runs(root):
    sources = {}
    for mode in MODES:
        run = folder(root, mode); meta = read(run/'run.json'); summary = read(run/'SUMMARY.json')
        assert meta['execution_status'] == 'completed'
        assert meta['optimizer_steps'] == (12720 if mode == 'baseline' else 8000)
        assert sha(run/'best.pt') == summary['best_checkpoint_sha256']
        sources[mode] = {'run_id': run.name, 'checkpoint_sha256': sha(run/'best.pt'),
                         'selected_step': summary['selected']['step'], 'summary': summary}
    schedules = [sources[m]['summary']['sample_schedule_sha256'] for m in MODES[1:]]
    assert len(set(schedules)) == 1, 'Different domain draw/augmentation schedules'
    assert len({sources[m]['summary']['frozen_body_before'] for m in MODES[1:]}) == 1
    assert (sources['random_memory']['summary']['full_head_trainable_parameters'] ==
            sources['task_spm']['summary']['full_head_trainable_parameters'])
    return sources


@torch.no_grad()
def cost(model, store):
    rows = [r for r in store.rows if r['domain'] == 'real' and r['split'] == 'dev'][:24]
    images = [store.batch([r], 0, augment=False)['img'] for r in rows]
    for _ in range(3):
        predict(model, images[0])
    torch.cuda.synchronize(); timings = []
    for image in images:
        start = time.perf_counter(); predict(model, image); torch.cuda.synchronize()
        timings.append((time.perf_counter()-start)*1000)
    return {'median_inference_and_mask_decode_ms': float(np.median(timings)),
            'p90_ms': float(np.percentile(timings, 90)), 'images': len(images), 'warmup': 3,
            'total_parameters': sum(p.numel() for p in model.parameters()), 'precision': 'fp32',
            'device': torch.cuda.get_device_name(), 'disk_and_COCOeval_excluded': True}


@torch.no_grad()
def identity_and_contribution(model, store):
    module = memory_module(model); module.capture = False; measurements = []
    rows = [r for r in store.rows if r['domain'] == 'real' and r['split'] == 'dev'][:24]
    for row in rows:
        image = store.batch([row], 0, augment=False)['img']
        module.intervention = 'none'; result = model.eval()(image)
        det, proto = result[0]; proto = proto[0] if isinstance(proto, tuple) else proto
        chosen = det[0, :, 4] >= .25
        coefficients = det[0, chosen, 6:]; bases = proto[0].flatten(1)
        native = coefficients[:, :32]@bases[:32]; extra = coefficients[:, 32:]@bases[32:]
        native_rms = float(native.square().mean().sqrt()) if native.numel() else 0.
        extra_rms = float(extra.square().mean().sqrt()) if extra.numel() else 0.
        raw = det[..., :6].clone(); del result, proto, native, extra
        maximum = 0.
        for intervention in INTERVENTIONS:
            module.intervention = intervention
            altered = model(image)[0][0][..., :6]
            maximum = max(maximum, float((raw-altered).abs().max()))
        assert maximum == 0., 'Intervention changed raw Box/Class/score or row order'
        measurements.append({'source_id': row['source_id'], 'display_detections': int(chosen.sum()),
                             'native_mask_logit_rms': native_rms, 'added_mask_logit_rms': extra_rms,
                             'raw_detection_max_error_all_interventions': maximum})
    module.capture = False; module.maps = {}; module.intervention = 'none'
    return {'rows': measurements, 'score_threshold_for_logit_description': .25,
            'scope': 'uncropped_160x160_logit_rms_not_mask_AP',
            'raw_rows_box_class_score_unchanged': True,
            'post_mask_empty_filter_can_remove_different_rows': True}


def family_scores(root, run, store, tags):
    groups = defaultdict(list)
    for row in store.rows:
        if row['domain'] == 'real' and row['split'] == 'dev':
            groups[row['group']].append(row['image_id'])
    source = read(root/'DEV_COCO.json'); values = {}
    for index, (group, ids) in enumerate(sorted(groups.items())):
        gt_file = run/f'family_{index}_COCO.json'; keep = set(ids)
        gt = {**source, 'images': [r for r in source['images'] if r['id'] in keep],
              'annotations': [r for r in source['annotations'] if r['image_id'] in keep]}
        dump(gt_file, gt); values[group] = {'images': len(ids), 'instances': len(gt['annotations']), 'metrics': {}}
        for tag in tags:
            predictions = [r for r in read(run/'evaluation'/(tag+'_predictions.json')) if r['image_id'] in keep]
            stats = score(gt_file, predictions, 'segm')
            values[group]['metrics'][tag] = {'mask_ap': stats[0], 'mask_ap75': stats[2]}
    return {'group_identity': 'inferred_filename_family_not_verified_physical_pen', 'families': values}


def decisions(metrics, cfg):
    primary = metrics['task_spm']['mask_ap']-metrics['continued']['mask_ap']
    deltas = {m: metrics['task_spm']['mask_ap']-metrics[m]['mask_ap'] for m in MODES[:-1]}
    drops = {m: metrics['task_spm']['mask_ap']-metrics['task_spm_'+m]['mask_ap'] for m in INTERVENTIONS}
    used = drops['off'] > cfg['prototype_intervention_meaningful_delta_pp']
    specific = used and deltas['adapter'] > 0 and deltas['random_memory'] > 0
    strongest_native = max(metrics['baseline']['mask_ap'], metrics['continued']['mask_ap'])
    strongest_delta = metrics['task_spm']['mask_ap']-strongest_native
    matched_target = primary >= cfg['target_mask_ap_absolute_delta_pp']
    target = strongest_delta >= cfg['target_mask_ap_absolute_delta_pp']
    if target and specific:
        status = 'five_pp_development_target_and_operational_prototype_use_supported'
        direction = 'deliver_task_supervised_high_resolution_prototype_mask_basis'
        next_action = 'independent_real_test_with_frozen_checkpoint_and_no_tuning'
    elif specific and primary > 0:
        status = 'positive_prototype_increment_below_five_pp_target'
        direction = 'prioritize_observed_prototype_helped_mask_failure_types'
        next_action = 'reuse_saved_predictions_and_logit_contributions_to_choose_one_evidence_based_revision'
    else:
        status = 'prototype_specific_utility_not_supported_in_this_configuration'
        best = max(MODES, key=lambda m: metrics[m]['mask_ap'])
        direction = 'retain_best_actual_checkpoint_'+best+'_and_stop_scaling_this_dictionary'
        next_action = 'use_existing_same_checkpoint_interventions_and_G_R_controls_to_identify_the_supported_bottleneck'
    return {'status': status, 'target_delta_pp': 5., 'measured_F_minus_continued_pp': primary,
            'F_minus_comparators_pp': deltas, 'normal_minus_intervention_pp': drops,
            'operational_prototype_use_drop_exceeds_0_1_pp': used, 'F_exceeds_G_and_R_and_off_drops': specific,
            'matched_primary_five_pp_met': matched_target,
            'strongest_retained_native_mask_ap': strongest_native, 'F_minus_strongest_native_pp': strongest_delta,
            'goal_acceptance_addendum': 'GOAL_ACCEPTANCE_ADDENDUM.md',
            'five_pp_metric_target_met': target, 'goal_achieved_pending_artifact_acceptance': target and specific,
            'chosen_research_direction': direction, 'unique_next_action': next_action,
            'evidence_scope': 'single_seed_432_real_development_images_selected_on_same_development_set',
            'not_independent_test_or_unconfounded_causal_proof': True}


def main(root):
    cfg = read(root/'TASK_CONFIG_V3.json'); sources = source_runs(root)
    with task_context(root, 'evaluate_task', freeze='EVALUATION_FREEZE.json') as (run, meta):
        store = Store(read(root/'ALL_DATASET.json')['rows'], max_cache=64)
        metrics, costs = {}, {}
        for mode in MODES:
            model = load(folder(root, mode)/'best.pt').eval(); metrics[mode] = evaluate(model, root, store, run/'evaluation', mode)
            expected = sources[mode]['summary']['selected']['mask_ap']
            assert abs(metrics[mode]['mask_ap']-expected) < 1e-7, (mode, expected, metrics[mode]['mask_ap'])
            costs[mode] = cost(model, store)
            if mode == 'task_spm':
                contributions = identity_and_contribution(model, store); dump(run/'CONTRIBUTIONS.json', contributions)
                module = memory_module(model); before = module.frozen_hash()
                for intervention in INTERVENTIONS:
                    module.intervention = intervention; tag = 'task_spm_'+intervention
                    metrics[tag] = evaluate(model, root, store, run/'evaluation', tag)
                module.intervention = 'none'; assert module.frozen_hash() == before
            del model; gc.collect(); torch.cuda.empty_cache()
        family = family_scores(root, run, store, list(metrics)); decision = decisions(metrics, cfg)
        dump(run/'SOURCE_RUNS.json', sources); dump(run/'METRICS.json', metrics); dump(run/'COST.json', costs)
        dump(run/'FAMILY_RESULTS.json', family); dump(run/'DECISION.json', decision)
        meta.update(source_run_ids={k: v['run_id'] for k, v in sources.items()}, optimizer_steps=0,
                    evaluation_images=432, prototype_steps=20000, method_steps_per_arm=8000)
        from task_visualize import figures
        figures(root, run, store, sources, metrics, costs, family, decision)
        dump(run/'SUMMARY.json', {'metrics': metrics, 'cost': costs, 'decision': decision,
                                  'five_pp_target_met': decision['five_pp_metric_target_met'],
                                  'training_steps_in_evaluation_run': 0,
                                  'figure_files': [p.name for p in sorted((run/'figures').glob('*.png'))]})
        from task_report import report
        report(root, run, sources, metrics, costs, family, decision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    args = parser.parse_args(); torch.set_num_threads(4); cv2.setNumThreads(1)
    main(Path(args.root))
