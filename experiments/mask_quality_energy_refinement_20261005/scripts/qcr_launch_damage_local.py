"""Launch the authorized local fixed diagnostic with an observed Run record."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from runner import execute

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--images', type=int, choices=(2, 64), required=True)
    args = ap.parse_args()
    if not args.run_id.startswith('RUN_damage_') or any(c in args.run_id for c in '/\\:'):
        raise ValueError('Require an independent diagnostic Run ID')
    scripts = Path(__file__).resolve().parent
    study = scripts.parent
    workspace = study.parents[1]
    out = study/'runs'/args.run_id
    config = study/'LOCAL_DAMAGE_CONFIG_20261006.json'
    command = [sys.executable, '-u', str(scripts/'qcr_damage_gradient.py'),
               '--config', str(config), '--out', str(out), '--images', str(args.images)]
    snapshots = [str(scripts/name) for name in (
        'qcr_launch_damage_local.py', 'qcr_damage_gradient.py', 'qcr_local_streaming_data.py',
        'qcr_streaming_data.py', 'qcr_evaluate_complete.py', 'qcr_train_fixed.py',
        'online_runtime.py', 'prepare_qcr_cache.py', 'legacy_prepare_cache.py', 'qcr_metrics.py', 'runner.py')]
    snapshots += [str(config), str(study/'DAMAGE_DIAGNOSIS_PROTOCOL.md')]
    inputs = [str(study/'IMAGE_SPLIT.json'),
              str(study/'runs/RUN_stage1_quality_seed0/final.pt'),
              str(study/'runs/RUN_stage1_dev_original_metrics_seed0/PER_CANDIDATE.jsonl'),
              str(workspace/'assets/models/coco_clean_20260911/yolo26m-seg.pt'),
              str(workspace/'assets/datasets/coco/annotations/instances_train2017.json'),
              str(workspace/'assets/datasets/coco/annotations/instances_val2017.json')]
    sys.exit(execute(SimpleNamespace(command=command, output=str(out), run_id=args.run_id,
        study='STUDY_6f0c44af49e846f0a14f44a4b43d38e4', cwd=str(workspace), input=inputs,
        snapshot=snapshots, expect=[str(out/'COMPLETE.json'), str(out/'SUMMARY.json'), str(out/'PER_CANDIDATE.jsonl')],
        scope='{"purpose":"fixed DEV mechanism diagnosis","execution_environment":"user-authorized local CUDA","parameter_updates":false,"tuning":false}',
        metrics=str(out/'SUMMARY.json'))))
