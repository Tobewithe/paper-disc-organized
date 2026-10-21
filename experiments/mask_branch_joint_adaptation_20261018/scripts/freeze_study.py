"""Build the pre-execution contract and minimal transferable source/input bundle."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
import argparse

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
REMOTE = '/srv/gpu-lab/users/a25023104/projects/paper-disc-organized'
DATA = '/srv/gpu-lab/users/a25023104/datasets/coco2017'
NAME = STUDY.name

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    inputs = {
        'weights': 'assets/models/coco_clean_20260911/yolo26m-seg.pt',
        'profile': 'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json',
        'original_split': 'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec/SPLIT_CONTRACT.json',
        'adapter_initial': 'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2/initial.pt',
        'existing_G': 'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb/last.pt',
        'historical_native_summary': 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/SUMMARY.json',
        'historical_native_trace': 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/trace/TRACE_MANIFEST.json',
        'historical_native_ledger': 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/ledger/PER_GT_FAILURE.jsonl',
        'historical_image_ids': 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/image_ids.json',
    }
    config = {k: REMOTE + '/' + v for k, v in inputs.items()}
    config.update({
        'study_id': 'STUDY_MASK_BRANCH_JOINT_ADAPTATION_20261018',
        'study_root': f'{REMOTE}/experiments/{NAME}',
        'train_annotations': DATA + '/annotations/instances_train2017.json',
        'val_annotations': DATA + '/annotations/instances_val2017.json',
        'train_images': DATA + '/train2017', 'val_images': DATA + '/val2017',
        'expected_python': '3.12.3',
        'expected_packages': {'torch': '2.5.1+cu124', 'torchvision': '0.20.1+cu124',
            'ultralytics': '8.4.100', 'numpy': '2.4.3', 'pycocotools': '2.0.11',
            'faster-coco-eval': '1.8.0', 'opencv-python': '4.13.0.92', 'Pillow': '11.1.0', 'PyYAML': '6.0.2'},
        'expected_annotations': {'train': '610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d',
                                 'val': 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'},
        'seed': 20261018, 'training_seed': 0, 'fit_images': 20000, 'epochs': 3,
        'source_batch': 16, 'effective_batch': 4, 'steps_per_arm': 15000,
        'training_hours': 8, 'statistics_hours': 12, 'gpu_memory_limit_gib': 44, 'area_gate': 9216,
        'optimizer': {'name': 'AdamW', 'lr': 0.0001, 'betas': [0.9, 0.999], 'eps': 1e-8,
                      'weight_decay': 0.0001, 'foreach': False, 'fused': False},
        'training_policy': {'amp': False, 'ema': False, 'scheduler': None, 'gradient_clip': None,
                            'augmentation': False, 'early_stopping': False, 'checkpoint_selection': 'last'},
        'runtime_policy': json.loads((ROOT / inputs['profile']).read_text(encoding='utf-8'))['runtime_policy'],
        'smoke_max_images': 32, 'smoke_steps_per_arm': 2,
        'migration_guard': {'historical_bbox_ap': 52.35143015406392,
            'historical_segm_ap': 44.01417484774351, 'historical_G_segm_ap': 44.1557477428,
            'max_absolute_ap_drift': 0.02, 'max_G_gain_drift': 0.02},
        'bootstrap': {'replicates': 2000, 'seed': 20261018, 'percentiles': [2.5, 97.5],
                      'unit': 'paired_whole_images', 'shared_draws': True},
        'gates': {'PC_minus_N': 0.5, 'PC_minus_G_old': 0.3, 'PC_minus_G_budget': 0.2,
                  'PC_minus_C_native': 0.2, 'all_ci_lower_gt': 0, 'native_success_damage_max': 0.01,
                  'PC_native_inference_median_ratio_max': 4.0, 'unchanged_detection_identity': True},
        'protocol': f'{REMOTE}/experiments/{NAME}/PROTOCOL.md',
        'frozen_manifest': f'{REMOTE}/experiments/{NAME}/FROZEN_MANIFEST.json',
    })
    write(STUDY / 'CONFIG.json', config)
    write(STUDY / 'INPUT_REFERENCES.json', {k: {'relative_path': v, 'sha256': sha(ROOT / v),
         'size_bytes': (ROOT / v).stat().st_size} for k, v in inputs.items()})
    if args.config_only:
        print('CONFIG and INPUT_REFERENCES prepared; source freeze not created')
        return
    roots = ['shared/evaluation', 'shared/vendor/ultralytics_8_4_100',
             'experiments/official_gt_failure_rebaseline_20261013/scripts',
             'experiments/mask_readout_mechanism_v2_20261012/scripts',
             'experiments/mask_readout_soft_iou_pilot_20261014/scripts',
             'experiments/mask_gate_s3r_20261015/scripts', f'experiments/{NAME}/scripts']
    files = {ROOT / p for p in inputs.values()}
    for folder in roots:
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')
                     and not p.name.startswith(('data_environment_', 'prepare_server')))
    files.update(STUDY / f for f in ('PROTOCOL.md', 'CONFIG.json', 'INPUT_REFERENCES.json', 'AUTHORIZATION.json'))
    rows = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'size_bytes': p.stat().st_size}
            for p in sorted(files)]
    manifest = {'schema_version': 1, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_state': 'explicit_exported_file_hashes_include_uncommitted_current_study_sources',
        'protocol_sha256': sha(STUDY / 'PROTOCOL.md'), 'config_sha256': sha(STUDY / 'CONFIG.json'),
        'files': rows}
    freeze = STUDY / 'FROZEN_MANIFEST.json'
    if freeze.exists():
        raise RuntimeError('Existing freeze must not be silently replaced; preserve as a versioned engineering recovery first')
    write(freeze, manifest)
    bundle = STUDY / 'transfer' / 'joint_adaptation_v1.tar.gz'
    bundle.parent.mkdir(exist_ok=True)
    with tarfile.open(bundle, 'w:gz') as archive:
        for p in sorted(files | {freeze}):
            archive.add(p, arcname=p.relative_to(ROOT).as_posix())
    write(STUDY / 'BUNDLE_RECEIPT.json', {'bundle': str(bundle), 'sha256': sha(bundle),
          'size_bytes': bundle.stat().st_size, 'files': len(rows) + 1, 'remote_root': REMOTE})
    print(json.dumps({'bundle': str(bundle), 'size_bytes': bundle.stat().st_size, 'files': len(rows) + 1}))

if __name__ == '__main__':
    main()
