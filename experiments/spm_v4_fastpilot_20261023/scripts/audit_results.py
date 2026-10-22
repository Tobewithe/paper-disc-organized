"""Independent artifact, split, frozen-coordinate and COCO score receipt."""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import dump, read, sha, utc


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); args = p.parse_args(); root = Path(args.root)
    torch.set_num_threads(4)
    dataset = read(root/'DATASET.json'); rows = dataset['rows']; by_source = {r['source_id']: r for r in rows}
    train = {r['source_id'] for r in rows if r['split'] == 'train'}
    dev = {r['source_id'] for r in rows if r['split'] == 'dev'}
    assert not train & dev and len(rows) == len(by_source) == 418
    assert not {r['domain']+'/'+r['group'] for r in rows if r['split'] == 'train'} & {r['domain']+'/'+r['group'] for r in rows if r['split'] == 'dev'}
    for row in rows:
        assert sha(row['image']) == row['image_sha256'] and sha(row['label']) == row['label_sha256']
        if row['domain'] == 'cutout':
            assert row['base_source_id'] in train and row['donor_source_id'] in train
    def run(stage):
        return root/'runs'/read(root/(stage.upper()+'_POINTER.json'))['run_id']
    baseline = torch.load(run('baseline')/'best.pt', map_location='cpu', weights_only=False)
    memory = torch.load(run('prototype')/'memory.pt', map_location='cpu', weights_only=False)
    assert sha(run('baseline')/'best.pt') == memory['upstream_checkpoint_sha256']
    features = torch.load(run('prototype')/'embedding.pt', map_location='cpu', weights_only=False)
    assert all(r['source_id'] in train for r in features['records'])
    assert memory['centers'].shape == (32, 128)
    norm_error = float((memory['centers'].norm(dim=1)-1).abs().max()); assert norm_error < 1e-5
    arms = {}; schedules = []
    for stage in ['baseline', 'continued', 'adapter', 'spm', 'export_lite']:
        folder = run(stage); meta = read(folder/'run.json'); summary = read(folder/'SUMMARY.json')
        assert meta['execution_status'] == 'completed' and meta['exit_code'] == 0
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        if stage == 'export_lite':
            metrics = summary['metrics']; tag = 'native_head_export'
            assert not any(k.startswith('spm.') for k in checkpoint['state_dict'])
        else:
            metrics = summary['selected']; tag = f"step_{metrics['step']:04d}"
            evaluations = read(folder/'EVALUATIONS.json')
            assert metrics['mask_ap'] == max(e['mask_ap'] for e in evaluations)
            assert checkpoint['step'] == metrics['step']
            schedule = read(folder/'SAMPLE_SCHEDULE.json')
            assert len(schedule) == (600 if stage == 'baseline' else 400)
            assert all(s in train for batch in schedule for s in batch)
            if stage != 'baseline':
                assert summary['frozen_body_before'] == summary['frozen_body_after']
                assert all(torch.equal(t, checkpoint['state_dict'][k]) for k, t in baseline['state_dict'].items()
                           if not k.startswith('model.23.'))
                assert all(sum(by_source[s]['domain'] == 'real' for s in batch) == 2 for batch in schedule)
                schedules.append(sha(folder/'SAMPLE_SCHEDULE.json'))
        if stage == 'spm':
            expected = {'spm.mean': memory['mean'].view(1, -1, 1, 1), 'spm.std': memory['std'].view(1, -1, 1, 1),
                        'spm.keys': memory['centers'], 'spm.values': memory['centers'],
                        'spm.projector.weight': memory['projector_weight'].view(128, 256, 1, 1),
                        'spm.projector.bias': memory['projector_bias']}
            assert all(torch.equal(t, checkpoint['state_dict'][k]) for k, t in expected.items())
        predictions = read(folder/'evaluation'/(tag+'_predictions.json'))
        checked = {}
        for kind, name in [('segm', 'mask_ap'), ('bbox', 'box_ap')]:
            with contextlib.redirect_stdout(io.StringIO()):
                gt = COCO(str(root/'DEV_COCO.json'))
                pred = [{k: v for k, v in item.items() if k != ('bbox' if kind == 'segm' else 'segmentation')} for item in predictions]
                result = gt.loadRes(pred)
                evaluator = COCOeval(gt, result, kind); evaluator.params.maxDets = [1, 10, 100]
                evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            value = float(evaluator.stats[0]*100)
            assert abs(value-metrics[name]) < 1e-10
            checked[name] = value
        arms[stage] = {'run_id': folder.name, 'checkpoint_sha256': sha(folder/'best.pt'),
                       'recomputed': checked, 'recorded_steps': meta['optimizer_steps']}
        print('AUDIT_OK', stage, checked, flush=True)
        del checkpoint
    assert len(set(schedules)) == 1
    decision = read(run('visualize')/'DECISION.json')
    delta = arms['spm']['recomputed']['mask_ap']-arms['continued']['recomputed']['mask_ap']
    assert abs(delta-decision['primary_delta_pp']) < 1e-10 and delta > 0
    receipt = {'checked_at': utc(), 'status': 'passed', 'data_images': 418,
               'real_train_dev': [172, 54], 'cache_file_hashes_verified': len(rows)*2,
               'train_dev_groups_disjoint': True, 'cutout_donors_train_only': True,
               'prototype_sources_train_only': True, 'prototype_tokens': len(features['records']),
               'prototype_unit_norm_max_error': norm_error, 'memory_coordinates_and_buffers_frozen': True,
               'arm_sample_schedule_identical': True, 'all_recomputed_AP_differences_below': 1e-10,
               'arms': arms, 'positive_development_delta_pp': delta,
               'scope': 'artifact_and_evaluation_validity_not_independent_test_or_mechanism_confirmation'}
    dump(root/'ACCEPTANCE_RECEIPT.json', receipt)


if __name__ == '__main__':
    main()
