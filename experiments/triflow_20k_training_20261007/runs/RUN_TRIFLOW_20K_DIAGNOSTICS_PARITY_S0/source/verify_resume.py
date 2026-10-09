"""Real 32-image uninterrupted versus interrupted/resumed training equivalence."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from frozen_io import dump_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--images-list', type=Path, required=True)
    p.add_argument('--revision', default='')
    p.add_argument('--reuse-engineering-runs', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    run = root / 'runs' / a.run_id
    weights = 'D:/coco_wire/models/yolo26m-seg.pt'
    vendor = 'D:/coco_wire/vendor_8.4.100'
    annotations = 'D:/coco_wire/bgcr_native_20261002/data/annotations/instances_train2017.json'
    names = ['RUN_TRIFLOW_20K_ENGINEERING32_S0', 'RUN_TRIFLOW_20K_ENGINEERING_INTERRUPTED_S0',
             'RUN_TRIFLOW_20K_ENGINEERING_RESUMED_S0']
    if a.revision:
        if not a.revision.isalnum():
            raise ValueError('Revision must be alphanumeric')
        names = [name + '_' + a.revision for name in names]
    common = ['--weights', weights, '--vendor', vendor, '--images-list', str(a.images_list),
              '--annotations', annotations, '--epochs', '1', '--seed', '0', '--instance-chunk', '4',
              '--device', 'cuda', '--max-images', '32', '--checkpoint-seconds', '300', '--trace-every', '250']
    for i, name in enumerate(names):
        if a.reuse_engineering_runs:
            metadata = json.loads((root / 'runs' / name / 'run.json').read_text(encoding='utf-8-sig'))
            expected_state = 'failed' if i == 1 else 'completed'
            expected_exit = 2 if i == 1 else 0
            if metadata.get('execution_status') != expected_state or metadata.get('exit_code') != expected_exit:
                raise ValueError('Existing engineering Run has not reached its required terminal state: ' + name)
            continue
        additions = []
        if i == 1:
            additions = ['--engineering-stop-after-steps', '10']
        if i == 2:
            additions = ['--resume-from', str(root / 'runs' / names[1] / 'checkpoint_latest.pt')]
        cmd = [sys.executable, '-X', 'utf8', str(root / 'scripts/run_stage.py'), '--root', str(root),
               '--run-id', name, '--purpose', 'engineering_checkpoint_resume_equivalence',
               '--script', 'train_20k.py', '--', '--root', str(root), '--run-id', name] + common + additions
        code = subprocess.run(cmd, cwd=root).returncode
        expected = 2 if i == 1 else 0
        if code != expected:
            raise RuntimeError(f'{name} exit {code}, expected {expected}; preserve all partial runs')
    import numpy as np
    import torch
    files = [root / 'runs' / name / 'checkpoint_latest.pt' for name in names]
    uninterrupted = torch.load(files[0], map_location='cpu', weights_only=False)
    interrupted = torch.load(files[1], map_location='cpu', weights_only=False)
    resumed = torch.load(files[2], map_location='cpu', weights_only=False)
    for snapshot, name in zip((uninterrupted, interrupted, resumed), names):
        if snapshot['run_id'] != name:
            raise ValueError('Snapshot Run identity differs')
        for file, digest in snapshot['contract']['sources'].items():
            if sha256(root / 'runs' / name / 'source' / file) != digest or sha256(root / 'scripts' / file) != digest:
                raise ValueError('Existing engineering snapshot tested another training implementation: ' + file)
    counts = {'tensors': 0, 'tensor_elements': 0, 'scalars': 0}

    def exact(left, right, label):
        if type(left) is not type(right):
            raise AssertionError('Type mismatch: ' + label)
        if isinstance(left, torch.Tensor):
            if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
                raise AssertionError('Tensor values/dtype/shape differ: ' + label)
            if left.contiguous().numpy().tobytes() != right.contiguous().numpy().tobytes():
                raise AssertionError('Tensor bytes differ: ' + label)
            counts['tensors'] += 1
            counts['tensor_elements'] += left.numel()
        elif isinstance(left, np.ndarray):
            if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
                raise AssertionError('NumPy state differs: ' + label)
        elif isinstance(left, dict):
            if set(left) != set(right):
                raise AssertionError('Dict keys differ: ' + label)
            for key in left:
                exact(left[key], right[key], label + '/' + str(key))
        elif isinstance(left, (tuple, list)):
            if len(left) != len(right):
                raise AssertionError('Sequence length differs: ' + label)
            for i, (one, two) in enumerate(zip(left, right)):
                exact(one, two, label + '/' + str(i))
        else:
            if left != right:
                raise AssertionError('Value differs: ' + label)
            counts['scalars'] += 1

    for key in ('contract', 'contract_sha256', 'head_state_dict', 'optimizer_state_dict',
                'initial_head_state_dict', 'rng', 'head_state_sha256'):
        exact(uninterrupted[key], resumed[key], key)
    # Extraction calls belong to this process, whereas the learned state and
    # all frozen tensor integrity facts must be identical across both runs.
    for key in uninterrupted['frozen_integrity']:
        if key != 'extracted_images':
            exact(uninterrupted['frozen_integrity'][key], resumed['frozen_integrity'][key], 'frozen_integrity/' + key)
    for key in ('attempts', 'applied', 'instances', 'order', 'cursor', 'task_evidence', 'task_probed_epoch',
                'finite_groups', 'gradient_steps', 'frozen_input_checks', 'initial_head_state_sha256'):
        exact(uninterrupted['state'][key], resumed['state'][key], 'state/' + key)
    if len(uninterrupted['state']['epochs']) != 1 or len(resumed['state']['epochs']) != 1:
        raise AssertionError('Engineering epoch did not complete')
    for key in uninterrupted['state']['epochs'][0]:
        if key != 'seconds':
            exact(uninterrupted['state']['epochs'][0][key], resumed['state']['epochs'][0][key], 'epoch/' + key)
    for key in ('data_hashes', 'sources', 'base_weights_sha256', 'dimensions', 'observed_inputs',
                'unique_images', 'frozen_initial_state_sha256'):
        exact(uninterrupted['provider_state'][key], resumed['provider_state'][key], 'provider/' + key)
    if interrupted['state']['applied'] != 10 or uninterrupted['state']['applied'] <= 10:
        raise AssertionError('Actual 10-step interruption evidence missing')
    stop = json.loads((root / 'runs' / names[1] / 'INTENTIONAL_ENGINEERING_STOP.json').read_text())
    if stop['smoke_only'] is not True or (root / 'runs' / names[1] / 'head_final.pt').exists():
        raise AssertionError('Interrupted engineering run fabricated a final deployable head')
    audits = [json.loads((root / 'runs' / name / 'TRAINING_AUDIT.json').read_text()) for name in (names[0], names[2])]
    if not all(r.get('passed') is True and r.get('smoke_only') is True for r in audits):
        raise AssertionError('Actual engineering audits did not pass')
    result = {'passed': True, 'scope': 'Engineering checkpoint/resume equality; no method capacity measurement',
        'engineering_images': 32, 'epochs': 1, 'interrupted_applied_steps': 10,
        'reused_completed_engineering_runs': a.reuse_engineering_runs,
        'final_applied_steps': uninterrupted['state']['applied'], 'runs': names,
        'snapshot_sha256': {name: sha256(path) for name, path in zip(names, files)},
        'head_optimizer_all_rng_and_order_bitwise_exact': True,
        'final_head_state_sha256': uninterrupted['head_state_sha256'],
        'training_sources': uninterrupted['contract']['sources'],
        'training_contract_sha256': uninterrupted['contract_sha256'],
        'all_epoch_scientific_counts_and_losses_exact': True, 'counts': counts,
        'effective_training_seconds': audits[0]['effective_training_seconds'],
        'end_to_end_engineering_seconds': audits[0]['this_execution_seconds'],
        'projected_20000_images_8epoch_training_seconds': audits[0]['effective_training_seconds'] / 32 * 20000 * 8,
        'projection_is_estimate': True, 'projection_limit': '32 declared engineering images; full split instance complexity may differ',
        'provider_repeated_extraction_visits': {'uninterrupted': uninterrupted['provider_state']['visits']['images'],
                                             'resumed': resumed['provider_state']['visits']['images']},
        'frozen_extraction_calls_in_final_process': {'uninterrupted': uninterrupted['frozen_integrity']['extracted_images'],
                                                  'resumed': resumed['frozen_integrity']['extracted_images']},
        'provider_visit_difference_reason': 'Resuming a partial image regenerates frozen input; it does not duplicate committed optimizer updates',
        'checkpoint_semantics': 'Only the durable prefix is committed; later crash work may roll back and replay'}
    dump_json(run / 'RESUME_VERIFICATION.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
