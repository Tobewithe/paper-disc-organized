"""Register and run CPU synthetic checks only. No real-data execution mode exists."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

from rfc_paths import ROOT, STUDY, ORIGINAL, SOFT
from rfc_contract import sha256, read_json, check_source_references, execution_readiness


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-of', default=None)
    args = parser.parse_args()
    run_id = 'RUN_RFC_ENGINEERING_'+uuid.uuid4().hex
    path = STUDY/'runs'/run_id
    lock = STUDY/'.engineering.lock'
    started = time.monotonic()
    meta = {'study_id': 'STUDY_MASK_REGION_F_CORRESPONDENCE_20261017', 'run_id': run_id,
            'stage_id': 'cpu_synthetic_component_verification', 'kind': 'engineering_only',
            'execution_status': 'running', 'scientific_status': 'not_evaluated',
            'artifact_status': 'partial', 'transfer_status': 'local_only',
            'started_at': now(), 'ended_at': None, 'exit_code': None, 'retry_of': args.retry_of,
            'executor': {'host': socket.gethostname(), 'pid': os.getpid(), 'device': 'cpu',
                         'interpreter': sys.executable, 'workspace': str(ROOT)},
            'command': [sys.executable, '-B', *sys.argv], 'wall_budget_seconds': 600,
            'base_model_forward_calls': 0, 'real_data_reads': 0,
            'optimizer_steps': 0, 'new_F_cache_exports': 0}
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(meta, stream)
    path.mkdir(parents=True, exist_ok=False)
    write(path/'run.json', meta)
    print('RUN_DIRECTORY', str(path), flush=True)
    sources = {}
    success = False
    try:
        check_source_references()
        files = sorted((STUDY/'scripts').glob('*.py')) + [
            STUDY/'ENGINEERING_PROTOCOL.md', STUDY/'PROTOCOL_DRAFT.md', STUDY/'DESIGN_SPEC.json',
            STUDY/'INPUT_REFERENCES.json', ORIGINAL/'adapter_model.py',
            ORIGINAL/'adapter_primitives.py', SOFT/'soft_iou_loss.py']
        sources = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
        meta['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        patch = subprocess.check_output(['git', 'diff', '--no-ext-diff', '--',
                                         STUDY.relative_to(ROOT).as_posix(),
                                         'project/ACTIVE_PLAN.md', 'project/PLAN_CATALOG.json'], cwd=ROOT)
        (path/'dirty.patch').write_bytes(patch)
        meta['dirty_patch_sha256'] = sha256(path/'dirty.patch')
        snapshots = []
        (path/'source_snapshot').mkdir()
        for i, (relative, digest) in enumerate(sources.items()):
            destination = path/'source_snapshot'/f'{i:02d}_{Path(relative).name}'
            shutil.copyfile(ROOT/relative, destination)
            snapshots.append({'original': relative, 'snapshot': destination.relative_to(path).as_posix(),
                              'sha256': digest})
        write(path/'SOURCE_MANIFEST.json', {'files': sources, 'snapshots': snapshots})
        shutil.copyfile(STUDY/'ENGINEERING_PROTOCOL.md', path/'PROTOCOL.md')
        meta.update(protocol_path='PROTOCOL.md', protocol_sha256=sha256(path/'PROTOCOL.md'))
        expected = read_json(STUDY/'DESIGN_SPEC.json')['training']['numeric_environment_exact_reference']
        actual = {'python': platform.python_version(), 'torch': importlib.metadata.version('torch'),
                  'numpy': importlib.metadata.version('numpy'), 'device': 'cpu', 'threads': 1}
        write(path/'EVAL_CONTRACT.json', {
            'kind': 'synthetic_engineering_not_scientific_evaluation', 'test_input': 'generated synthetic tensors',
            'actual_environment': actual, 'original_scientific_environment': expected,
            'allowed_environment_difference': 'CPU software behavior checks only; does not authorize native data replay or training',
            'native_environment_parity': False, 'wall_seconds': 600, 'artifact_bytes_max': 64*1024**2,
            'real_training_updates': 0, 'base_model_calls': 0, 'COCO_reads': 0, 'new_cache_exports': 0,
            'checkpoint_use': 'read existing initial/G state_dict on CPU; synthetic forward only, no updates'})
        meta['contract_sha256'] = sha256(path/'EVAL_CONTRACT.json')
        write(path/'run.json', meta)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
        child = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-v',
                                '-s', str(STUDY/'scripts'), '-p', 'test_*.py'],
                               cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace',
                               timeout=max(.001, 600-(time.monotonic()-started)))
        (path/'tests.log').write_text(child.stdout, encoding='utf-8')
        print(child.stdout, flush=True)
        unchanged = all(sha256(ROOT/relative) == digest for relative, digest in sources.items())
        referenced = check_source_references()
        size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        audit = {'status': 'passed' if unchanged and size <= 64*1024**2 else 'failed',
                 'source_hashes_unchanged': unchanged, 'reference_check': referenced,
                 'artifact_bytes': size, 'scope': 'CPU synthetic engineering only',
                 'native_environment_parity': False, 'real_execution_readiness': execution_readiness()}
        write(path/'CONFIG_AUDIT.json', audit)
        found = re.search(r'Ran (\d+) tests?', child.stdout)
        success = child.returncode == 0 and unchanged and size <= 64*1024**2 and found is not None
        write(path/'SUMMARY.json', {'run_id': run_id, 'status': 'passed' if success else 'failed',
                                   'tests_run': int(found.group(1)) if found else None,
                                   'child_exit_code': child.returncode,
                                   'actual_environment': actual, 'base_model_forward_calls': 0,
                                   'optimizer_steps': 0, 'new_F_cache_exports': 0,
                                   'scientific_results': {'A_minus_N': None, 'A_minus_G': None,
                                                          'mechanism_support': 'not_evaluated'}})
        write(path/'DECISION.json', {'status': 'engineering_passed' if success else 'engineering_failed',
                                    'scientific_status': 'not_evaluated',
                                    'summary_sha256': sha256(path/'SUMMARY.json'),
                                    'next_action': 'Review retained failure' if not success else
                                      'Prepare the pinned-laptop native replay/real smoke stage; no automatic export or training'})
    except Exception:
        error = traceback.format_exc()
        (path/'ERROR.txt').write_text(error, encoding='utf-8')
        print(error, file=sys.stderr, flush=True)
        write(path/'DECISION.json', {'status': 'engineering_failed', 'reason': error,
                                    'scientific_status': 'not_evaluated',
                                    'next_action': 'Repair only the demonstrated engineering blocker'})
    finally:
        meta.update(execution_status='completed' if success else 'failed',
                    artifact_status='complete' if success else 'retained_failure',
                    exit_code=0 if success else 1, ended_at=now(),
                    elapsed_seconds=time.monotonic()-started)
        write(path/'run.json', meta)
        files = [{'path': p.relative_to(path).as_posix(), 'sha256': sha256(p),
                  'bytes': p.stat().st_size} for p in sorted(path.rglob('*')) if p.is_file()]
        write(path/'ARTIFACT_MANIFEST.json', {'artifacts': files})
        lock.unlink()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
