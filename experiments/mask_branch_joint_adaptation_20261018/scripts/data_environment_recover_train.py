#!/usr/bin/env python3
"""Only recover the slow train ZIP transfer; do not restart environment/val/GT."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import urllib.request
import uuid

import prepare_server as base

PARENT = base.STUDY / 'runs/RUN_DATA_ENV_8394c214fcc14e9cb8a4698e4b2598ee'
base.RUN = base.STUDY / 'runs' / ('RUN_DATA_ENV_RECOVERY_' + uuid.uuid4().hex)
RUN = base.RUN
URL = base.ARCHIVES[0][1]
ARCHIVE = base.DATA / 'archives/train2017.zip'
PARTS = base.DATA / 'archives/train2017.zip.range-parts'

def request_headers(start=None, end=None):
    headers = {'User-Agent': 'COCO-research-data-preparation/1.0'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    return urllib.request.Request(URL, headers=headers, method='HEAD' if start is None else 'GET')

def main():
    RUN.mkdir(parents=True, exist_ok=False)
    meta = {'run_id': RUN.name, 'retry_of': PARENT.name, 'job_id': os.environ.get('SLURM_JOB_ID'), 'execution_status': 'running', 'scientific_status': 'not_applicable_data_environment_preparation', 'created_at': base.now(), 'training_steps': 0, 'model_forwards': 0, 'reason': 'Same official train archive using 8 Range streams; private environment, val and annotations continue in parent Job 97'}
    base.write(RUN / 'run.json', meta)
    base.write(base.STUDY / 'data_environment_recovery_status.json', meta | {'run_path': str(RUN)})
    try:
        # Terminate exactly the original train downloader after identity check.
        pid = 3481563
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'curl ' in cmdline and str(ARCHIVE) in cmdline and URL in cmdline, cmdline
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not Path(f'/proc/{pid}').exists():
                break
            time.sleep(.2)
        assert not Path(f'/proc/{pid}').exists(), 'Original writer did not exit'
        prefix = ARCHIVE.with_suffix('.zip.original-prefix')
        assert not prefix.exists()
        ARCHIVE.rename(prefix)
        prefix_size, prefix_sha = prefix.stat().st_size, base.sha(prefix)
        with urllib.request.urlopen(request_headers(), timeout=60) as r:
            size, etag = int(r.headers['Content-Length']), r.headers['ETag']
            assert r.headers.get('Accept-Ranges') == 'bytes'
        base.write(RUN / 'RECOVERY_INPUT.json', {'url': URL, 'content_length': size, 'etag': etag, 'original_downloader_pid': pid, 'original_downloader_command': cmdline, 'preserved_prefix': str(prefix), 'prefix_size': prefix_size, 'prefix_sha256': prefix_sha})
        PARTS.mkdir(parents=True, exist_ok=True)
        def download(index):
            begin, finish = size * index // 8, size * (index + 1) // 8 - 1
            target = PARTS / f'part-{index:02d}'
            for attempt in range(8):
                have = target.stat().st_size if target.exists() else 0
                assert have <= finish - begin + 1
                if have == finish - begin + 1:
                    return target
                try:
                    with urllib.request.urlopen(request_headers(begin + have, finish), timeout=120) as r:
                        assert r.status == 206, r.status
                        assert r.headers['Content-Range'] == f'bytes {begin + have}-{finish}/{size}', dict(r.headers)
                        assert r.headers['ETag'] == etag, dict(r.headers)
                        with target.open('ab') as out:
                            for block in iter(lambda: r.read(1024 * 1024), b''):
                                out.write(block)
                    assert target.stat().st_size == finish - begin + 1
                    return target
                except Exception as e:
                    with (RUN / 'range-errors.log').open('a') as f:
                        f.write(f'{base.now()} part={index} attempt={attempt} {e!r}\n')
                    if attempt == 7:
                        raise
                    time.sleep(5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(download, range(8)))
        combined = ARCHIVE.with_suffix('.zip.assembling')
        with combined.open('wb') as dst:
            for p in paths:
                with p.open('rb') as src:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        assert combined.stat().st_size == size
        import hashlib
        h = hashlib.sha256()
        with combined.open('rb') as f:
            remaining = prefix_size
            while remaining:
                block = f.read(min(8 * 1024 * 1024, remaining))
                assert block
                h.update(block)
                remaining -= len(block)
        assert h.hexdigest() == prefix_sha, 'Recovered bytes mismatch preserved original prefix'
        combined.replace(ARCHIVE)
        # Reuse original ZIP SHA/member CRC/count/extraction checker without downloading again.
        original_command = base.command
        base.command = lambda args, name: None if name == 'train2017.zip-download' else original_command(args, name)
        train = base.prepare_archive(base.ARCHIVES[0])
        train['range_transfer'] = {'streams': 8, 'etag': etag, 'status_206_content_range_etag_all_checked': True, 'original_prefix_exact': True}
        base.write(RUN / 'train2017.zip.json', train)
        # Parent keeps its authentic interrupted-transfer failure; acceptance combines
        # successful parent environment/val/GT and this separately identified recovery.
        deadline = time.monotonic() + 8 * 3600
        while True:
            parent = json.loads((PARENT / 'run.json').read_text())
            if parent['execution_status'] != 'running':
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Parent environment/val/GT still running after eight hours')
            time.sleep(15)
        results = parent.get('results', {})
        for name in ['environment', 'val2017.zip', 'annotations_trainval2017.zip']:
            assert results.get(name, {}).get('status') == 'passed', (name, parent)
        assert set(parent.get('failures', {})) == {'train2017.zip'}, parent
        results['train2017.zip'] = train
        meta.update(execution_status='completed', completed_at=base.now(), results=results, original_transfer_failure_retained=True)
        base.write(RUN / 'run.json', meta)
        base.write(base.STUDY / 'data_environment_status.json', meta | {'run_path': str(RUN), 'all_required_inputs_ready': True})
        base.write(base.STUDY / 'data_environment_recovery_status.json', meta | {'run_path': str(RUN)})
        print('DATA_ENV_READY', RUN, flush=True)
    except Exception:
        meta.update(execution_status='failed', completed_at=base.now(), error=traceback.format_exc())
        base.write(RUN / 'run.json', meta)
        base.write(base.STUDY / 'data_environment_recovery_status.json', meta | {'run_path': str(RUN)})
        raise

if __name__ == '__main__':
    main()
