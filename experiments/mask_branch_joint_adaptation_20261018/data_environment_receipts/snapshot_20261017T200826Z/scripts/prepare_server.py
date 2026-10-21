#!/usr/bin/env python3
"""Prepare private pinned runtime and original COCO data; no model execution."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

BASE = Path('/srv/gpu-lab/users/a25023104')
PROJECT = BASE / 'projects/paper-disc-organized'
STUDY = PROJECT / 'experiments/mask_branch_joint_adaptation_20261018'
DATA = BASE / 'datasets/coco2017'
RUN = STUDY / 'runs' / ('RUN_DATA_ENV_' + uuid.uuid4().hex)
ENV = dict(os.environ, UV_CACHE_DIR=str(BASE / '.cache/uv'), UV_PYTHON_INSTALL_DIR=str(BASE / '.cache/uv-python'), UV_HTTP_TIMEOUT='120', UV_HTTP_RETRIES='5')
ARCHIVES = (
    ('train2017.zip', 'http://images.cocodataset.org/zips/train2017.zip', 'train2017/', 118287),
    ('val2017.zip', 'http://images.cocodataset.org/zips/val2017.zip', 'val2017/', 5000),
    ('annotations_trainval2017.zip', 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip', 'annotations/', None),
)

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def command(args, name):
    print(now(), name, args, flush=True)
    with (RUN / (name + '.log')).open('ab') as log:
        result = subprocess.run(args, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} exited {result.returncode}; see {RUN / (name + ".log")}')

def prepare_environment():
    command(['uv', 'python', 'install', '3.12.3', '--no-bin'], 'python-install')
    command(['uv', 'venv', '--python', '3.12.3', str(PROJECT / '.venv')], 'venv-create')
    py = str(PROJECT / '.venv/bin/python')
    command(['uv', 'pip', 'install', '--python', py, '--index-url', 'https://download.pytorch.org/whl/cu124', 'torch==2.5.1+cu124', 'torchvision==0.20.1+cu124'], 'torch-install')
    command(['uv', 'pip', 'install', '--python', py, '--index-url', 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple', 'ultralytics==8.4.100', 'numpy==2.4.3', 'pycocotools==2.0.11', 'faster-coco-eval==1.8.0', 'opencv-python==4.13.0.92', 'Pillow==11.1.0', 'PyYAML==6.0.2', 'torch==2.5.1+cu124', 'torchvision==0.20.1+cu124'], 'packages-install')
    source = '''import sys,json,importlib.metadata as m,torch\nnames=['ultralytics','torch','torchvision','numpy','pycocotools','faster-coco-eval','opencv-python','Pillow','PyYAML']\nx={n:m.version(n) for n in names}; x['python']=sys.version; x['cuda_build']=torch.version.cuda; print(json.dumps(x))\nassert sys.version_info[:3]==(3,12,3)\nexpected=['8.4.100','2.5.1+cu124','0.20.1+cu124','2.4.3','2.0.11','1.8.0','4.13.0.92','11.1.0','6.0.2']\nassert [x[n] for n in names]==expected, x\n'''
    result = subprocess.run([py, '-c', source], env=ENV, text=True, capture_output=True)
    (RUN / 'environment-check.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Private runtime version/import validation failed')
    versions = json.loads(result.stdout.strip().splitlines()[-1])
    freeze = subprocess.check_output(['uv', 'pip', 'freeze', '--python', py], env=ENV, text=True)
    (RUN / 'requirements-resolved.txt').write_text(freeze, encoding='utf-8')
    write(RUN / 'ENVIRONMENT.json', {'status': 'passed', 'checked_at': now(), 'python': py, 'versions': versions, 'shared_environment_modified': False, 'gpu_access_attempted': False})
    return {'status': 'passed', 'versions': versions}

def prepare_archive(item):
    filename, url, prefix, expected_count = item
    archive = DATA / 'archives' / filename
    archive.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = RUN / (filename + '.json')
    command(['curl', '--fail', '--location', '--retry', '5', '--retry-delay', '5', '--connect-timeout', '30', '--speed-time', '180', '--speed-limit', '1024', '--continue-at', '-', '--output', str(archive), url], filename + '-download')
    result = {'source_url': url, 'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive), 'source': 'COCO original official host; HTTP because HTTPS endpoint TLS handshake failed', 'started_extract': now()}
    with zipfile.ZipFile(archive) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        if expected_count is not None:
            assert len(files) == expected_count, (filename, len(files), expected_count)
        for i in files:
            assert i.filename.startswith(prefix), i.filename
            target = (DATA / i.filename).resolve()
            assert target.is_relative_to(DATA.resolve()), i.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            # ZipExtFile verifies each decompressed member CRC. Existing files are
            # left untouched only after comparing their bytes against the archive.
            if target.exists():
                import zlib
                crc = 0
                with target.open('rb') as old:
                    for b in iter(lambda: old.read(1024 * 1024), b''):
                        crc = zlib.crc32(b, crc)
                if target.stat().st_size == i.file_size and (crc & 0xffffffff) == i.CRC:
                    continue
                raise RuntimeError(f'Existing member differs from official archive: {target}')
            tmp = target.with_suffix(target.suffix + '.extracting')
            with z.open(i) as src, tmp.open('wb') as dst:
                for b in iter(lambda: src.read(1024 * 1024), b''):
                    dst.write(b)
            tmp.replace(target)
        result.update(status='passed', file_count=len(files), member_crc_verified=True, completed_at=now())
    if expected_count is None:
        result['instances'] = {}
        for split, count in [('train2017', 118287), ('val2017', 5000)]:
            p = DATA / 'annotations' / ('instances_' + split + '.json')
            obj = json.loads(p.read_text(encoding='utf-8'))
            assert len(obj['images']) == count
            assert len(obj['categories']) == 80
            result['instances'][split] = {'path': str(p), 'sha256': sha(p), 'images': len(obj['images']), 'annotations': len(obj['annotations']), 'categories': len(obj['categories'])}
    write(receipt_path, result)
    return result

def main():
    RUN.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    run = {'run_id': RUN.name, 'study': 'mask_branch_joint_adaptation_20261018', 'execution_status': 'running', 'scientific_status': 'not_applicable_data_environment_preparation', 'created_at': now(), 'host': platform.node(), 'job_id': os.environ.get('SLURM_JOB_ID'), 'training_steps': 0, 'model_forwards': 0, 'commands_logged': True}
    write(RUN / 'run.json', run)
    write(STUDY / 'data_environment_status.json', run | {'run_path': str(RUN)})
    results, failures = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(prepare_environment): 'environment'}
        futures.update({pool.submit(prepare_archive, item): item[0] for item in ARCHIVES})
        for f in concurrent.futures.as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()
                print(now(), 'COMPLETE', name, flush=True)
            except Exception:
                failures[name] = traceback.format_exc()
                print(now(), 'ERROR', name, failures[name], flush=True)
            write(RUN / 'PROGRESS.json', {'completed': results, 'failures': failures, 'updated_at': now()})
    run.update(execution_status='failed' if failures else 'completed', completed_at=now(), elapsed_seconds=time.monotonic() - start, results=results, failures=failures)
    write(RUN / 'run.json', run)
    write(STUDY / 'data_environment_status.json', run | {'run_path': str(RUN)})
    raise SystemExit(1 if failures else 0)

if __name__ == '__main__':
    main()
