#!/usr/bin/env python3
"""Fetch exact official CUDA wheels; recover transport without version changes."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import shutil
import time
import traceback
import urllib.request
import uuid

import prepare_server as base

PARENT = base.STUDY / 'runs/RUN_DATA_ENV_8394c214fcc14e9cb8a4698e4b2598ee'
RUNTIME_PARENT = base.STUDY / 'runs/RUN_DATA_ENV_PYTHON_RECOVERY_9b71ff58e43f4eda93210e49d1c7d9af'
TRAIN_RECOVERY = base.STUDY / 'runs/RUN_DATA_ENV_RECOVERY_10b640b4f66e444ab46ae244a38487ec'
base.RUN = base.STUDY / 'runs' / ('RUN_DATA_ENV_WHEEL_RECOVERY_' + uuid.uuid4().hex)
RUN = base.RUN
WHEEL_DIR = base.BASE / '.cache/pinned-pytorch-cu124-wheels'
WHEELS = [
    ('torch-2.5.1+cu124-cp312-cp312-linux_x86_64.whl', 'bf6484bfe5bc4f92a4a1a1bf553041505e19a911f717065330eb061afe0e14d7'),
    ('torchvision-0.20.1+cu124-cp312-cp312-linux_x86_64.whl', 'd1053ec5054549e7dac2613b151bffe323f3c924939d296df4d7d34925aaf3ad'),
]

def download(item):
    name, expected = item
    url = 'https://download.pytorch.org/whl/cu124/' + name.replace('+', '%2B')
    target = WHEEL_DIR / name
    if target.exists():
        assert base.sha(target) == expected
        return target
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        total, etag = int(r.headers['Content-Length']), r.headers['ETag']
        assert r.headers['Accept-Ranges'] == 'bytes'
    n = 4 if name.startswith('torch-') else 1
    def part(index):
        start, end = total * index // n, total * (index + 1) // n - 1
        path = WHEEL_DIR / f'{name}.part-{index}'
        for attempt in range(6):
            have = path.stat().st_size if path.exists() else 0
            assert have <= end - start + 1
            if have == end - start + 1:
                return path
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={start + have}-{end}'})
                with urllib.request.urlopen(req, timeout=120) as r:
                    assert r.status == 206
                    assert r.headers['Content-Range'] == f'bytes {start + have}-{end}/{total}'
                    assert r.headers['ETag'] == etag
                    with path.open('ab') as out:
                        for block in iter(lambda: r.read(1024 * 1024), b''):
                            out.write(block)
                assert path.stat().st_size == end - start + 1
                return path
            except Exception as e:
                with (RUN / 'download-retries.log').open('a') as f:
                    f.write(f'{base.now()} {name} part={index} attempt={attempt}: {e!r}\n')
                if attempt == 5:
                    raise
                time.sleep(5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        pieces = list(pool.map(part, range(n)))
    with target.with_suffix('.assembling').open('wb') as dst:
        for piece in pieces:
            with piece.open('rb') as src:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    target.with_suffix('.assembling').replace(target)
    actual = base.sha(target)
    assert actual == expected, (name, actual, expected)
    base.write(RUN / (name + '.json'), {'status': 'passed', 'source_url': url, 'official_index': 'https://download.pytorch.org/whl/cu124/' + ('torch/' if name.startswith('torch-') else 'torchvision/'), 'sha256': actual, 'official_index_sha256': expected, 'bytes': total, 'etag': etag, 'ranges': n, 'all_ranges_206_content_range_etag_verified': True})
    return target

def main():
    RUN.mkdir(parents=True, exist_ok=False)
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = {'run_id': RUN.name, 'retry_of': RUNTIME_PARENT.name, 'job_id': os.environ.get('SLURM_JOB_ID'), 'execution_status': 'running', 'scientific_status': 'not_applicable_data_environment_preparation', 'created_at': base.now(), 'training_steps': 0, 'model_forwards': 0, 'reason': 'Exact official SHA-verified wheels replace prolonged official-index metadata transport; dependencies from Tsinghua mirror'}
    base.write(RUN / 'run.json', meta)
    base.write(base.STUDY / 'data_environment_wheel_recovery_status.json', meta | {'run_path': str(RUN)})
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            wheel_paths = list(pool.map(download, WHEELS))
        ready = RUNTIME_PARENT / 'RUNTIME_READY.json'
        original = Path('/proc/3499663/cmdline')
        if not ready.exists() and original.exists():
            args = original.read_bytes().split(b'\0')
            assert args[1:3] == [b'pip', b'install']
            assert b'https://download.pytorch.org/whl/cu124' in args
            assert str(base.PROJECT / '.venv/bin/python').encode() in args
            os.kill(3499663, signal.SIGTERM)
            for _ in range(100):
                if not Path('/proc/3499663').exists():
                    break
                time.sleep(.2)
            assert not Path('/proc/3499663').exists()
            meta['interrupted_original_index_install_after_exact_wheel_verification'] = True
        # If the original process advanced to package installation, let it finish.
        deadline = time.monotonic() + 3600
        while not ready.exists() and json.loads((RUNTIME_PARENT / 'run.json').read_text())['execution_status'] == 'running':
            if time.monotonic() > deadline:
                raise RuntimeError('Original package installation did not finish')
            time.sleep(2)
        if ready.exists():
            meta.update(execution_status='completed', completed_at=base.now(), recovery_used=False, reason='Original installation completed before direct-wheel recovery was needed', original_runtime_ready=str(ready))
            base.write(RUN / 'run.json', meta)
            base.write(base.STUDY / 'data_environment_wheel_recovery_status.json', meta | {'run_path': str(RUN)})
            return
        assert (base.PROJECT / '.venv/bin/python').exists()
        original_command = base.command
        def command(args, name):
            if name in ['python-install', 'venv-create']:
                return
            if name == 'torch-install':
                args = ['uv', 'pip', 'install', '--python', str(base.PROJECT / '.venv/bin/python'), '--index-url', 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple', 'numpy==2.4.3', 'Pillow==11.1.0', *map(str, wheel_paths)]
            return original_command(args, name)
        base.command = command
        runtime = base.prepare_environment()
        base.write(RUN / 'RUNTIME_READY.json', {'status': 'passed', 'runtime_receipt': str(RUN / 'ENVIRONMENT.json'), 'runtime': runtime, 'checked_at': base.now()})
        expected = {'train2017.zip': TRAIN_RECOVERY / 'train2017.zip.json', 'val2017.zip': PARENT / 'val2017.zip.json', 'annotations_trainval2017.zip': PARENT / 'annotations_trainval2017.zip.json'}
        deadline = time.monotonic() + 8 * 3600
        while not all(p.exists() for p in expected.values()):
            if time.monotonic() > deadline:
                raise RuntimeError('Dataset component readiness timeout')
            time.sleep(15)
        results = {name: json.loads(p.read_text()) for name, p in expected.items()}
        assert all(item['status'] == 'passed' for item in results.values())
        results['environment'] = runtime
        meta.update(execution_status='completed', completed_at=base.now(), recovery_used=True, results=results, component_receipts={name: str(p) for name, p in expected.items()} | {'environment': str(RUN / 'ENVIRONMENT.json')}, all_required_inputs_ready=True)
        base.write(RUN / 'run.json', meta)
        base.write(base.STUDY / 'data_environment_status.json', meta | {'run_path': str(RUN)})
        base.write(base.STUDY / 'data_environment_wheel_recovery_status.json', meta | {'run_path': str(RUN)})
        print('DATA_ENV_READY', RUN, flush=True)
    except Exception:
        meta.update(execution_status='failed', completed_at=base.now(), error=traceback.format_exc())
        base.write(RUN / 'run.json', meta)
        base.write(base.STUDY / 'data_environment_wheel_recovery_status.json', meta | {'run_path': str(RUN)})
        raise

if __name__ == '__main__':
    main()
