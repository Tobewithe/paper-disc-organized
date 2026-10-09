"""One execution: full-data readiness, training, complete evaluation, readout."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from frozen_io import dump_json, sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    run = root / 'runs' / a.run_id
    archive = Path(__file__).parent
    locked = {file.name: sha256(file) for file in archive.glob('*.py')}
    protocol_sha = sha256(archive / 'PROTOCOL.md')
    dump_json(run / 'PIPELINE_INPUTS.json', {'source_sha256': locked, 'protocol_sha256': protocol_sha,
        'full_train_images_required': 118287, 'epochs': 8, 'seed': 0, 'eval_images': 5000,
        'environment': 'laptop_28358lan', 'data_wait_limit_hours': 48,
        'waiting_is_not_training': True, 'started_at': now()})

    def guard():
        for file, digest in locked.items():
            if sha256(root / 'scripts' / file) != digest:
                raise ValueError('Queued pipeline source changed; preserve this Run and launch a new version: ' + file)
        if sha256(root / 'PROTOCOL.md') != protocol_sha:
            raise ValueError('Queued protocol changed')

    def status(stage, scientific_training_status):
        dump_json(run / 'PIPELINE_PROGRESS.json', {'stage': stage, 'observed_at': now(),
                 'scientific_training_status': scientific_training_status,
                 'training_run': 'RUN_TRIFLOW_FULL_TRAIN_S0', 'evaluation_run': 'RUN_TRIFLOW_FULL_COCO5000_EVAL_S0'})

    def stage(identifier, purpose, script, arguments):
        guard()
        cmd = [sys.executable, '-X', 'utf8', str(archive / 'run_stage.py'), '--root', str(root),
               '--run-id', identifier, '--purpose', purpose, '--script', script, '--'] + arguments
        code = subprocess.run(cmd, cwd=root).returncode
        if code:
            raise RuntimeError(f'{identifier} failed with exit {code}; preserve partial outputs')

    try:
        status('waiting_for_complete_official_train2017_materialization', 'not_started')
        started = time.monotonic()
        ready = root / 'data/FULL_DATA_READY.json'
        materialization = root / 'runs/RUN_TRIFLOW_FULL_DATA_MATERIALIZE_S0/run.json'
        while True:
            if ready.exists() and materialization.exists():
                metadata = json.loads(materialization.read_text(encoding='utf-8-sig'))
                if metadata.get('execution_status') == 'completed':
                    break
                if metadata.get('execution_status') == 'failed':
                    raise RuntimeError('Full dataset materialization failed; training remains unstarted')
            if time.monotonic()-started > 48*3600:
                raise TimeoutError('Complete data was not available within 48 hours; no subset substitution')
            time.sleep(30)
        status('complete_data_preflight', 'not_started')
        stage('RUN_TRIFLOW_FULL_PREFLIGHT_S0', 'complete_original_train_split_and_actual_execution_contract',
              'preflight_fullscale.py', ['--root', str(root), '--run-id', 'RUN_TRIFLOW_FULL_PREFLIGHT_S0'])
        status('full_original_coco_training_8epochs', 'running')
        stage('RUN_TRIFLOW_FULL_TRAIN_S0', 'full_original_train2017_seed0_fixed_8epoch_online_training',
              'train_fullscale.py', ['--root', str(root), '--run-id', 'RUN_TRIFLOW_FULL_TRAIN_S0',
                '--weights', 'D:/coco_wire/models/yolo26m-seg.pt', '--vendor', 'D:/coco_wire/vendor_8.4.100',
                '--images-list', str(root / 'data/train_full.txt'),
                '--annotations', 'D:/coco_wire/bgcr_native_20261002/data/annotations/instances_train2017.json',
                '--epochs', '8', '--seed', '0', '--instance-chunk', '4', '--device', 'cuda',
                '--checkpoint-seconds', '300', '--trace-every', '250'])
        status('full_val2017_native_evaluation', 'completed')
        stage('RUN_TRIFLOW_FULL_COCO5000_EVAL_S0', 'full_val2017_native_baseline_vs_fullscale_triflow',
              'evaluate_fullscale.py', ['--root', str(root), '--run-id', 'RUN_TRIFLOW_FULL_COCO5000_EVAL_S0',
                '--head', str(root / 'runs/RUN_TRIFLOW_FULL_TRAIN_S0/head_final.pt'),
                '--images-list', str(root / 'data/val_full.txt'),
                '--annotations', 'D:/coco_wire/data/annotations/instances_val2017.json',
                '--weights', 'D:/coco_wire/models/yolo26m-seg.pt', '--vendor', 'D:/coco_wire/vendor_8.4.100',
                '--baseline-cache', 'D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/official',
                '--device', 'cuda'])
        status('independent_readout', 'completed')
        stage('RUN_TRIFLOW_FULL_READOUT_VERIFY_S0', 'independent_fullscale_coco_and_paired_aggregate_verification',
              'verify_fullscale_readout.py', ['--run', str(root / 'runs/RUN_TRIFLOW_FULL_COCO5000_EVAL_S0'),
                '--out', str(root / 'runs/RUN_TRIFLOW_FULL_READOUT_VERIFY_S0/READOUT_VERIFICATION.json')])
        status('report', 'completed')
        stage('RUN_TRIFLOW_FULL_REPORT_S0', 'fullscale_coverage_execution_and_scientific_readout',
              'build_fullscale_report.py', ['--root', str(root), '--run-id', 'RUN_TRIFLOW_FULL_REPORT_S0',
                '--eval-run', 'RUN_TRIFLOW_FULL_COCO5000_EVAL_S0', '--train-run', 'RUN_TRIFLOW_FULL_TRAIN_S0',
                '--verification', str(root / 'runs/RUN_TRIFLOW_FULL_READOUT_VERIFY_S0/READOUT_VERIFICATION.json')])
        # Return bundles are prepared on the laptop after all writers settle.
        status('preparing_return_bundles', 'completed')
        stages = ['RUN_TRIFLOW_FULL_PREFLIGHT_S0', 'RUN_TRIFLOW_FULL_TRAIN_S0',
                  'RUN_TRIFLOW_FULL_COCO5000_EVAL_S0', 'RUN_TRIFLOW_FULL_READOUT_VERIFY_S0', 'RUN_TRIFLOW_FULL_REPORT_S0']
        for identifier in stages:
            subprocess.run([sys.executable, str(archive / 'collect_run.py'), '--root', str(root),
                            '--run', identifier], cwd=root, check=True)
        status('complete_remote_outputs_and_verified_report', 'completed')
        dump_json(run / 'PIPELINE_COMPLETE.json', {'passed_execution': True, 'completed_at': now(),
                  'remote_return_bundles_prepared': stages, 'local_return_status': 'pending',
                  'report': str(root / 'REPORT.md'), 'scientific_outcome': 'See actual final report; completion is not a positive-AP claim'})
    except BaseException as e:
        dump_json(run / 'PIPELINE_FAILURE.json', {'error': repr(e), 'failed_at': now(),
                  'scope': 'Preserve failed and partial Runs; fullscale scientific outcome remains unknown until verified'})
        raise


if __name__ == '__main__':
    main()
