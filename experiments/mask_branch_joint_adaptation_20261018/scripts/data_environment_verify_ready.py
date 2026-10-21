#!/usr/bin/env python3
"""Single finite independent readiness verification after jobs 99 and 100 end."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid

BASE = Path('/srv/gpu-lab/users/a25023104')
PROJECT = BASE / 'projects/paper-disc-organized'
STUDY = PROJECT / 'experiments/mask_branch_joint_adaptation_20261018'
DATA = BASE / 'datasets/coco2017'
RUN = STUDY / 'runs' / ('RUN_DATA_ENV_VERIFY_' + uuid.uuid4().hex)
EXPECTED_VERSIONS = {'ultralytics': '8.4.100', 'torch': '2.5.1+cu124', 'torchvision': '0.20.1+cu124', 'numpy': '2.4.3', 'pycocotools': '2.0.11', 'faster-coco-eval': '1.8.0', 'opencv-python': '4.13.0.92', 'Pillow': '11.1.0', 'PyYAML': '6.0.2'}
EXPECTED_GT = {'train2017': ('610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d', 118287, 860001), 'val2017': ('e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f', 5000, 36781)}

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)

def main():
    RUN.mkdir(parents=True, exist_ok=False)
    meta = {'run_id': RUN.name, 'job_id': os.environ.get('SLURM_JOB_ID'), 'execution_status': 'running', 'scientific_status': 'not_applicable_independent_input_verification', 'started_at': now(), 'training_steps': 0, 'model_forwards': 0, 'dependencies': [99, 100], 'finite_single_execution': True}
    write(RUN / 'run.json', meta)
    write(STUDY / 'data_environment_verification_status.json', meta | {'run_path': str(RUN)})
    try:
        account = subprocess.check_output(['sacct', '-X', '-j', '99,100', '--parsable2', '--noheader', '--format=JobIDRaw,State,ExitCode'], text=True)
        (RUN / 'PREDECESSOR_JOBS.txt').write_text(account, encoding='utf-8')
        states = {line.split('|')[0]: line.split('|')[1].split()[0] for line in account.splitlines() if line}
        for job in ['99', '100']:
            assert states.get(job) in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}, states
        status_path = STUDY / 'data_environment_status.json'
        status = json.loads(status_path.read_text())
        assert status.get('all_required_inputs_ready') is True and status['execution_status'] == 'completed', status
        runtime_receipts = []
        for name in ['RUN_DATA_ENV_PYTHON_RECOVERY_9b71ff58e43f4eda93210e49d1c7d9af', 'RUN_DATA_ENV_WHEEL_RECOVERY_de188f6c52d645eda5490c27e4d2cc2c']:
            path = STUDY / 'runs' / name / 'ENVIRONMENT.json'
            if path.exists():
                item = json.loads(path.read_text())
                assert item['status'] == 'passed'
                assert {key: item['versions'][key] for key in EXPECTED_VERSIONS} == EXPECTED_VERSIONS
                assert item['versions']['python'].startswith('3.12.3 ')
                runtime_receipts.append({'path': str(path), 'sha256': sha(path)})
        assert runtime_receipts, 'No independently checkable exact environment receipt'
        py = str(PROJECT / '.venv/bin/python')
        source = "import sys,json,importlib.metadata as m; import torch,torchvision,numpy,ultralytics,pycocotools.mask,faster_coco_eval,cv2,PIL,yaml; names=" + repr(list(EXPECTED_VERSIONS)) + "; print(json.dumps({'python':sys.version,'versions':{n:m.version(n) for n in names},'cuda_build':torch.version.cuda,'imports':'passed'}))"
        env = dict(os.environ, YOLO_CONFIG_DIR=str(BASE / '.cache/ultralytics'))
        check = subprocess.run([py, '-c', source], env=env, text=True, capture_output=True)
        (RUN / 'IMPORTS.log').write_text(check.stdout + check.stderr, encoding='utf-8')
        assert check.returncode == 0, 'Full library imports failed; see IMPORTS.log'
        actual = json.loads(check.stdout.strip().splitlines()[-1])
        assert actual['versions'] == EXPECTED_VERSIONS and actual['python'].startswith('3.12.3 ') and actual['cuda_build'] == '12.4', actual
        components = {}
        for name in ['train2017.zip', 'val2017.zip', 'annotations_trainval2017.zip']:
            receipt = status['results'][name]
            assert receipt['status'] == 'passed' and receipt['member_crc_verified'] is True
            archive = Path(receipt['archive']).resolve()
            assert archive.is_relative_to(DATA.resolve())
            assert archive.stat().st_size == receipt['bytes']
            checksum = sha(archive)
            assert checksum == receipt['sha256'], (name, checksum, receipt['sha256'])
            components[name] = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': checksum, 'matches_verified_component': True}
        split_checks = {}
        for split, (expected_sha, images, annotations) in EXPECTED_GT.items():
            path = DATA / 'annotations' / f'instances_{split}.json'
            checksum = sha(path)
            assert checksum == expected_sha, (split, checksum)
            gt = json.loads(path.read_text())
            assert len(gt['images']) == images and len(gt['annotations']) == annotations and len(gt['categories']) == 80
            expected_names = {im['file_name'] for im in gt['images']}
            actual_files = list((DATA / split).glob('*.jpg'))
            assert len(actual_files) == images and {p.name for p in actual_files} == expected_names
            assert all(p.stat().st_size > 0 for p in actual_files)
            split_checks[split] = {'images': images, 'annotations': annotations, 'categories': 80, 'annotation_sha256': checksum, 'directory': str(DATA / split), 'all_filenames_match_gt': True, 'all_files_nonempty': True}
            del gt, expected_names, actual_files
        meta.update(execution_status='completed', completed_at=now(), all_required_inputs_ready=True, environment=actual, runtime_receipts=runtime_receipts, components=components, splits=split_checks, aggregate_status_path=str(status_path), aggregate_status_sha256=sha(status_path), source_sha256=sha(Path(__file__)), source_runtime='python3 system standard library plus fresh imports from private Python only')
        write(RUN / 'run.json', meta)
        write(RUN / 'READY.json', meta)
        write(STUDY / 'data_environment_READY.json', meta | {'run_path': str(RUN)})
        write(STUDY / 'data_environment_verification_status.json', meta | {'run_path': str(RUN)})
        print('INDEPENDENT_INPUT_READY', RUN, flush=True)
    except Exception:
        meta.update(execution_status='failed', completed_at=now(), error=traceback.format_exc(), all_required_inputs_ready=False)
        write(RUN / 'run.json', meta)
        write(STUDY / 'data_environment_verification_status.json', meta | {'run_path': str(RUN)})
        raise

if __name__ == '__main__':
    main()
