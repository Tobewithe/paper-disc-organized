#!/usr/bin/env python3
"""Same-version source build fallback, private paths only, zero GPU/model work."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import traceback
import uuid

import prepare_server as base

PARENT = base.STUDY / 'runs/RUN_DATA_ENV_8394c214fcc14e9cb8a4698e4b2598ee'
TRAIN_RECOVERY = base.STUDY / 'runs/RUN_DATA_ENV_RECOVERY_10b640b4f66e444ab46ae244a38487ec'
base.RUN = base.STUDY / 'runs' / ('RUN_DATA_ENV_PYTHON_RECOVERY_' + uuid.uuid4().hex)
RUN = base.RUN
BUILD = base.BASE / '.cache/python3123-source-build'
PREFIX = base.BASE / '.cache/python3123-source-runtime'
SOURCE_URL = 'https://mirrors.huaweicloud.com/python/3.12.3/Python-3.12.3.tgz'
OFFICIAL_MD5 = '3c5498a34d5226c9b746b1199f0bf2d9'

def main():
    RUN.mkdir(parents=True, exist_ok=False)
    meta = {'run_id': RUN.name, 'retry_of': PARENT.name, 'job_id': os.environ.get('SLURM_JOB_ID'), 'execution_status': 'running', 'scientific_status': 'not_applicable_data_environment_preparation', 'created_at': base.now(), 'training_steps': 0, 'model_forwards': 0, 'reason': 'Exact Python 3.12.3 source-build fallback for slow official standalone delivery; no system package changes'}
    base.write(RUN / 'run.json', meta)
    base.write(base.STUDY / 'data_environment_runtime_recovery_status.json', meta | {'run_path': str(RUN)})
    try:
        BUILD.mkdir(parents=True, exist_ok=True)
        archive = BUILD / 'Python-3.12.3.tgz'
        base.command(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '20', '--max-time', '300', '--output', str(archive), SOURCE_URL], 'python-source-download')
        assert hashlib.md5(archive.read_bytes()).hexdigest() == OFFICIAL_MD5
        base.write(RUN / 'PYTHON_SOURCE.json', {'mirror_url': SOURCE_URL, 'official_release_page': 'https://www.python.org/downloads/release/python-3123/', 'official_source_url': 'https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz', 'md5_expected': OFFICIAL_MD5, 'md5_passed': True, 'sha256': base.sha(archive), 'bytes': archive.stat().st_size, 'install_prefix': str(PREFIX), 'shared_environment_modified': False})
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                assert (BUILD / member.name).resolve().is_relative_to(BUILD.resolve())
            tar.extractall(BUILD, filter='data')
        source = BUILD / 'Python-3.12.3'
        original_cwd = os.getcwd()
        os.chdir(source)
        try:
            base.command(['./configure', '--prefix=' + str(PREFIX), '--with-ensurepip=no'], 'python-configure')
            base.command(['make', '-j4'], 'python-build')
            base.command(['make', 'install'], 'python-install-private')
        finally:
            os.chdir(original_cwd)
        py = str(PREFIX / 'bin/python3.12')
        check = 'import sys,ssl,zlib,bz2,lzma,ctypes,multiprocessing,hashlib,json; assert sys.version_info[:3]==(3,12,3); print(json.dumps({"python":sys.version,"openssl":ssl.OPENSSL_VERSION,"required_standard_library_imports":"passed"}))'
        base.command([py, '-c', check], 'source-python-check')
        env_receipt = PARENT / 'ENVIRONMENT.json'
        original_uv = Path('/proc/3481560/cmdline')
        if original_uv.exists() and original_uv.read_bytes().split(b'\0')[1:4] == [b'python', b'install', b'3.12.3']:
            # Stop only the still-downloading same-version standalone process.
            os.kill(3481560, signal.SIGTERM)
            for _ in range(100):
                if not Path('/proc/3481560').exists():
                    break
                time.sleep(.2)
            assert not Path('/proc/3481560').exists()
            assert not (base.PROJECT / '.venv').exists(), 'A runtime already exists; refuse competing writers'
            original_command = base.command
            def command(args, name):
                if name == 'python-install':
                    return
                if name == 'venv-create':
                    args = ['uv', 'venv', '--python', py, str(base.PROJECT / '.venv')]
                return original_command(args, name)
            base.command = command
            runtime = base.prepare_environment()
            runtime['python_build_origin'] = 'official Python 3.12.3 source; mirror bytes matched official release checksum'
            meta['standalone_download_interrupted_after_verified_source_build'] = True
            runtime_path = RUN / 'ENVIRONMENT.json'
        else:
            # Original installation already advanced; do not create a competing env.
            deadline = time.monotonic() + 4 * 3600
            while not env_receipt.exists():
                parent = json.loads((PARENT / 'run.json').read_text())
                if parent['execution_status'] != 'running':
                    raise RuntimeError('Original environment failed before readiness')
                if time.monotonic() > deadline:
                    raise RuntimeError('Original environment readiness timeout')
                time.sleep(15)
            runtime = json.loads(env_receipt.read_text())
            runtime_path = env_receipt
            meta['source_build_unused_because_original_advanced'] = True
        base.write(RUN / 'RUNTIME_READY.json', {'status': 'passed', 'runtime_receipt': str(runtime_path), 'runtime': runtime, 'checked_at': base.now()})
        # Aggregate independently preserved successful components. A transfer/runtime
        # handoff can make parent/recovery aggregate Run exit failed; no old receipt
        # is changed. The actual verified component receipts determine readiness.
        deadline = time.monotonic() + 8 * 3600
        expected = {'train2017.zip': TRAIN_RECOVERY / 'train2017.zip.json', 'val2017.zip': PARENT / 'val2017.zip.json', 'annotations_trainval2017.zip': PARENT / 'annotations_trainval2017.zip.json'}
        while not all(p.exists() for p in expected.values()):
            if time.monotonic() > deadline:
                raise RuntimeError('Dataset component readiness timeout')
            time.sleep(15)
        results = {name: json.loads(p.read_text()) for name, p in expected.items()}
        assert all(item['status'] == 'passed' for item in results.values())
        results['environment'] = runtime
        meta.update(execution_status='completed', completed_at=base.now(), results=results, component_receipts={name: str(p) for name, p in expected.items()} | {'environment': str(runtime_path)}, all_required_inputs_ready=True)
        base.write(RUN / 'run.json', meta)
        base.write(base.STUDY / 'data_environment_status.json', meta | {'run_path': str(RUN)})
        base.write(base.STUDY / 'data_environment_runtime_recovery_status.json', meta | {'run_path': str(RUN)})
        print('DATA_ENV_READY', RUN, flush=True)
    except Exception:
        meta.update(execution_status='failed', completed_at=base.now(), error=traceback.format_exc())
        base.write(RUN / 'run.json', meta)
        base.write(base.STUDY / 'data_environment_runtime_recovery_status.json', meta | {'run_path': str(RUN)})
        raise

if __name__ == '__main__':
    main()
