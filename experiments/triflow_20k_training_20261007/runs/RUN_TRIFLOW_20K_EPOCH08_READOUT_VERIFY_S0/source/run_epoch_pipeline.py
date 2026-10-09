"""Continue the same eight-epoch training with synchronous real epoch evaluations."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from frozen_io import dump_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--resume-from', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    source = Path(__file__).parent
    run = root / 'runs' / a.run_id
    training_id = 'RUN_TRIFLOW_20K_TRAIN_S0_R1'
    addendum = root / 'EPOCH_EVALUATION_ADDENDUM.md'
    locked = {file.name: sha256(file) for file in source.glob('*.py')}
    dump_json(run / 'EPOCH_PIPELINE_INPUTS.json', {'resume_from': str(a.resume_from),
        'resume_snapshot_sha256': sha256(a.resume_from), 'scientific_budget_epochs': 8,
        'training_images': 20000, 'evaluation_images_each_epoch': 5000,
        'paired_diagnostics_epochs': [8], 'training_source_preserved': True,
        'source_sha256': locked, 'addendum_sha256': sha256(addendum)})

    def stage(identifier, entry, arguments):
        for name, digest in locked.items():
            if sha256(root / 'scripts' / name) != digest:
                raise ValueError('Locked observer execution source changed: ' + name)
        command = [sys.executable, '-X', 'utf8', str(source / 'run_stage.py'), '--root', str(root),
                   '--run-id', identifier, '--purpose', 'user_requested_evaluation_at_every_epoch',
                   '--script', entry, '--'] + arguments
        code = subprocess.run(command, cwd=root).returncode
        if code:
            raise RuntimeError(identifier + ' failed; partial artifacts preserved')

    try:
        stage(training_id, 'train_epoch_observer.py', [
            '--epoch-eval-script', str(source / 'evaluate_epoch.py'),
            '--epoch-verifier-script', str(source / 'verify_epoch_readout.py'),
            '--addendum', str(addendum),
            '--callback-images-list', str(root / 'data/val_full.txt'),
            '--callback-annotations', 'D:/coco_wire/data/annotations/instances_val2017.json',
            '--callback-baseline-cache', 'D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/official',
            '--callback-device', 'cuda', '--', '--root', str(root), '--run-id', training_id,
            '--weights', 'D:/coco_wire/models/yolo26m-seg.pt', '--vendor', 'D:/coco_wire/vendor_8.4.100',
            '--images-list', str(root / 'data/train20k.txt'),
            '--annotations', 'D:/coco_wire/bgcr_native_20261002/data/annotations/instances_train2017.json',
            '--epochs', '8', '--seed', '0', '--instance-chunk', '4', '--device', 'cuda',
            '--resume-from', str(a.resume_from), '--checkpoint-seconds', '300', '--trace-every', '250'])
        stage('RUN_TRIFLOW_20K_EPOCH_CURVE_REPORT_S0', 'build_epoch_curve_report.py', [
            '--root', str(root), '--run-id', 'RUN_TRIFLOW_20K_EPOCH_CURVE_REPORT_S0', '--training-run', training_id])
        identifiers = [training_id, 'RUN_TRIFLOW_20K_EPOCH_CURVE_REPORT_S0']
        for epoch in range(1, 9):
            identifiers += ['RUN_TRIFLOW_20K_EPOCH%02d_EVAL_S0' % epoch,
                            'RUN_TRIFLOW_20K_EPOCH%02d_READOUT_VERIFY_S0' % epoch]
        for identifier in identifiers:
            subprocess.run([sys.executable, str(source / 'collect_run.py'), '--root', str(root),
                            '--run', identifier], check=True, cwd=root)
        dump_json(run / 'PIPELINE_COMPLETE.json', {'status': 'completed', 'actual_epoch_evaluations': 8,
            'primary_epoch': 8, 'result': 'See actual verified report', 'return_bundles_prepared': identifiers,
            'local_return_status': 'pending', 'completed_at': datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        dump_json(run / 'PIPELINE_FAILURE.json', {'error': repr(error),
                 'time': datetime.now(timezone.utc).isoformat(), 'scientific_outcome': None})
        raise


if __name__ == '__main__':
    main()
