"""Read existing results and package an audit snapshot; never run a model or job."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess
import tarfile

ROOT = Path('/srv/gpu-lab/users/a25023104/projects/paper-disc-organized')
STUDY = ROOT / 'experiments/mask_branch_joint_adaptation_20261018'
OUT = STUDY / 'audits/status_20261018'
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

commands = {
    'queue': ['squeue', '-u', 'a25023104', '-o', '%i %j %T %R'],
    'accounting': ['sacct', '-j', '97,98,99,100,101,102,103,104,105,106,107',
                   '--format=JobID,JobName,State,ExitCode,Elapsed,Start,End', '-X', '-P'],
    'processes': ['ps', '-u', 'a25023104', '-o', 'pid,ppid,etime,args'],
}
status = {'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
          'timezone': dt.datetime.now().astimezone().tzname(), 'commands': {},
          'stage_lock_exists': (STUDY / '.JOINT_STAGE.lock').exists()}
for name, command in commands.items():
    r = subprocess.run(command, text=True, capture_output=True, check=False)
    status['commands'][name] = {'command': command, 'returncode': r.returncode,
                                'stdout': r.stdout, 'stderr': r.stderr}
write('SERVER_STATUS.json', status)
frozen = json.loads((STUDY / 'FROZEN_MANIFEST.json').read_text())
checks = []
for item in frozen['files']:
    p = ROOT / item['path']
    digest = sha(p) if p.is_file() else None
    checks.append({'path': item['path'], 'expected_sha256': item['sha256'],
                   'actual_sha256': digest, 'passed': digest == item['sha256']})
write('FROZEN_RECHECK.json', {'checked_at_utc': status['checked_at_utc'],
      'passed': all(x['passed'] for x in checks), 'count': len(checks),
      'manifest_sha256': sha(STUDY / 'FROZEN_MANIFEST.json'), 'checks': checks})
files = []
for p in sorted(STUDY.rglob('*')):
    rel = p.relative_to(STUDY)
    if not p.is_file() or 'audits' in rel.parts or '__pycache__' in rel.parts:
        continue
    if rel.parts[0] == 'runs' or (len(rel.parts) == 1 and p.suffix in {'.json', '.md'}):
        files.append((p, 'study/' + rel.as_posix()))
log_root = Path('/srv/gpu-lab/users/a25023104/logs')
for p in sorted(log_root.glob('*.log')):
    if p.stem.rsplit('-', 1)[-1] in {str(i) for i in range(97, 108)}:
        files.append((p, 'logs/' + p.name))
for name in ('SERVER_STATUS.json', 'FROZEN_RECHECK.json'):
    files.append((OUT / name, 'audit/' + name))
manifest = [{'path': rel, 'size_bytes': p.stat().st_size, 'sha256': sha(p)}
            for p, rel in files]
write('SNAPSHOT_MANIFEST.json', {'files': manifest})
archive = OUT / 'remote_snapshot.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for p, rel in files:
        tf.add(p, arcname=rel, recursive=False)
    tf.add(OUT / 'SNAPSHOT_MANIFEST.json', arcname='SNAPSHOT_MANIFEST.json')
receipt = {'archive_sha256': sha(archive), 'archive_size_bytes': archive.stat().st_size,
           'files': len(files), 'checked_at_utc': status['checked_at_utc'],
           'frozen_files_passed': all(x['passed'] for x in checks)}
write('TRANSFER_SOURCE.json', receipt)
print(json.dumps(receipt))
