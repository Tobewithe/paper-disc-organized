"""Resume the fixed budget with explicitly verified minimal diagnostics."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from frozen_io import dump_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--parity-receipt', type=Path, required=True)
    args = parser.parse_args()
    root, source = args.root.resolve(), Path(__file__).parent.resolve()
    run = root / 'runs' / args.run_id
    training_id = 'RUN_TRIFLOW_20K_TRAIN_S0_R3'
    addendum = root / 'EPOCH_EVALUATION_ADDENDUM.md'
    diagnostics_addendum = root / 'DIAGNOSTICS_EXECUTION_ADDENDUM.md'
    runtime = source / 'diagnostics_runtime.py'
    parity = args.parity_receipt.resolve()
    receipt = json.loads(parity.read_text(encoding='utf-8-sig'))
    if receipt.get('passed') is not True or receipt.get('smoke_only') is not True:
        raise ValueError('Actual diagnostic-equivalence engineering receipt required')
    engineering_file = root / 'runs/RUN_TRIFLOW_20K_DIAGNOSTICS_OBSERVER_ENGINEERING_VERIFY_S0/DIAGNOSTICS_OBSERVER_ENGINEERING_VERIFICATION.json'
    engineering = json.loads(engineering_file.read_text(encoding='utf-8-sig'))
    if engineering.get('passed') is not True or engineering.get('observer_source_sha256') != sha256(source / 'train_epoch_observer_r3.py'):
        raise ValueError('Actual R3 observer trajectory proof does not bind executed source')
    locked = {file.name: sha256(file) for file in source.glob('*.py')}
    dump_json(run / 'EPOCH_PIPELINE_INPUTS.json', {
        'resume_from': str(args.resume_from.resolve()),
        'resume_snapshot_sha256': sha256(args.resume_from),
        'scientific_budget_epochs': 8, 'training_images': 20000,
        'evaluation_images_each_epoch': 5000, 'paired_diagnostics_epochs': [8],
        'original_training_source_bytes_preserved': True,
        'runtime_diagnostic_calculation_changed': True,
        'diagnostics_runtime_sha256': sha256(runtime),
        'diagnostics_parity_receipt_sha256': sha256(parity),
        'source_sha256': locked, 'addendum_sha256': sha256(addendum),
        'diagnostics_addendum_sha256': sha256(diagnostics_addendum)})

    def stage(identifier, entry, arguments):
        for name, digest in locked.items():
            if sha256(root / 'scripts' / name) != digest:
                raise ValueError('Locked continuation source changed: ' + name)
        command = [sys.executable, '-X', 'utf8', str(source / 'run_stage.py'),
                   '--root', str(root), '--run-id', identifier,
                   '--purpose', 'verified_minimal_diagnostics_continuation',
                   '--script', entry, '--'] + arguments
        if subprocess.run(command, cwd=root).returncode:
            raise RuntimeError(identifier + ' failed; partial artifacts preserved')

    try:
        stage(training_id, 'train_epoch_observer_r3.py', [
            '--epoch-eval-script', str(source / 'evaluate_epoch_r3.py'),
            '--epoch-verifier-script', str(source / 'verify_epoch_readout_r3.py'),
            '--addendum', str(addendum),
            '--diagnostics-runtime', str(runtime),
            '--diagnostics-parity-receipt', str(parity),
            '--diagnostics-addendum', str(diagnostics_addendum),
            '--callback-images-list', str(root / 'data/val_full.txt'),
            '--callback-annotations', 'D:/coco_wire/data/annotations/instances_val2017.json',
            '--callback-baseline-cache', 'D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/official',
            '--callback-device', 'cuda', '--', '--root', str(root),
            '--run-id', training_id, '--weights', 'D:/coco_wire/models/yolo26m-seg.pt',
            '--vendor', 'D:/coco_wire/vendor_8.4.100',
            '--images-list', str(root / 'data/train20k.txt'),
            '--annotations', 'D:/coco_wire/bgcr_native_20261002/data/annotations/instances_train2017.json',
            '--epochs', '8', '--seed', '0', '--instance-chunk', '4', '--device', 'cuda',
            '--resume-from', str(args.resume_from.resolve()),
            '--checkpoint-seconds', '300', '--trace-every', '250'])
        report_id = 'RUN_TRIFLOW_20K_EPOCH_CURVE_REPORT_S0'
        stage(report_id, 'build_epoch_curve_report.py', [
            '--root', str(root), '--run-id', report_id, '--training-run', training_id])
        identifiers = [training_id, report_id]
        for epoch in range(1, 9):
            identifiers.extend([f'RUN_TRIFLOW_20K_EPOCH{epoch:02d}_EVAL_S0',
                                f'RUN_TRIFLOW_20K_EPOCH{epoch:02d}_READOUT_VERIFY_S0'])
        for identifier in identifiers:
            subprocess.run([sys.executable, str(source / 'collect_run.py'),
                            '--root', str(root), '--run', identifier], check=True, cwd=root)
        dump_json(run / 'PIPELINE_COMPLETE.json', {
            'status': 'completed', 'actual_epoch_evaluations': 8,
            'primary_epoch': 8, 'result': 'See actual verified report',
            'return_bundles_prepared': identifiers, 'local_return_status': 'pending',
            'completed_at': datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        dump_json(run / 'PIPELINE_FAILURE.json', {'error': repr(error),
            'time': datetime.now(timezone.utc).isoformat(), 'scientific_outcome': None})
        raise


if __name__ == '__main__':
    main()
