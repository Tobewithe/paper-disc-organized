"""Execute one immutable stage with factual Run metadata and preserved failure."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--purpose', required=True)
    parser.add_argument('--script', required=True)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    root = a.root.resolve()
    run = root / 'runs' / a.run_id
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'run.json').exists():
        prior = json.loads((run / 'run.json').read_text(encoding='utf-8-sig'))
        if prior.get('execution_status') not in (None, 'planned'):
            raise RuntimeError('Run already executed; use a distinct retry Run ID')
    source = run / 'source'
    source.mkdir(exist_ok=True)
    files = list((root / 'scripts').glob('*.py')) + [root / 'PROTOCOL.md', root / 'USER_PROPOSAL.txt']
    hashes = {}
    for file in files:
        destination = source / file.name
        shutil.copyfile(file, destination)
        hashes[file.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    if Path(a.script).name != a.script or not (source / a.script).is_file():
        raise ValueError('Unknown stage entry')
    command = [sys.executable, '-X', 'utf8', str(source / a.script)] + (a.args[1:] if a.args[:1] == ['--'] else a.args)
    meta = {'run_id': a.run_id, 'purpose': a.purpose, 'status': 'running',
            'execution_status': 'running', 'artifact_status': 'generating',
            'transfer_status': 'not_started', 'environment': 'laptop_28358lan',
            'ssh_alias': '28358lan', 'remote_directory': str(run),
            'interpreter': sys.executable, 'cwd': str(root), 'started_at': now(),
            'ended_at': None, 'exit_code': None, 'unverified': ['stage completion and scientific result pending']}
    dump(run / 'run.json', meta)
    dump(run / 'SOURCE.json', {'executed_script': str(source / a.script),
                              'sha256': hashes, 'archive_time_utc': now()})
    env = os.environ.copy()
    env.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', PYTHONUTF8='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', KMP_DUPLICATE_LIB_OK='TRUE')
    with (run / 'stdout.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        dump(run / 'PROCESS.json', {'pid': process.pid, 'command': command, 'started_at_utc': now()})
        meta['pid'] = process.pid
        dump(run / 'run.json', meta)
        code = process.wait()
    meta.update(status='completed' if code == 0 else 'failed', execution_status='completed' if code == 0 else 'failed',
                artifact_status='generated' if code == 0 else 'partial', ended_at=now(), exit_code=code,
                unverified=[] if code == 0 else ['stage failed; partial artifacts preserved'])
    dump(run / 'run.json', meta)
    dump(run / ('COMPLETE.json' if code == 0 else 'FAILED.json'),
         {'exit_code': code, 'completed': code == 0, 'observed_at_utc': now()})
    print(json.dumps({'run_id': a.run_id, 'exit_code': code, 'directory': str(run)}), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
