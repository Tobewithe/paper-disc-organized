"""Small file-only execution recorder; no workbench service dependency."""
from __future__ import annotations
import datetime, importlib.metadata, json, os, platform, shutil, socket, subprocess, sys
from pathlib import Path

def now():
    return datetime.datetime.now().astimezone().isoformat()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def record_run(study, run, script, arguments, inputs):
    """Execute a frozen source copy; preserve failures, prohibit overwrite."""
    protocol = json.loads((study/'protocol.json').read_text())
    out = study/'runs'/run['run_id']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'run.json').exists():
        raise FileExistsError(f'Existing execution must not be overwritten: {out}')
    snapshots = out/'snapshots'
    snapshots.mkdir()
    for source in (study/'scripts').glob('*.py'):
        shutil.copy2(source, snapshots/source.name)
    shutil.copy2(study/'protocol.json', snapshots/'protocol.json')
    command = [sys.executable, '-u', str(snapshots/script), *map(str, arguments)]
    metadata = dict(schema_version=2, run_id=run['run_id'], study_id=protocol['study_id'],
                    source_kind='subprocess_observed', kind=run['kind'], arm=run['arm'],
                    checkpoint=run.get('checkpoint'), status='running', started_at=now(),
                    retry_of=run.get('retry_of'),
                    finished_at=None, command=command, working_directory=str(snapshots),
                    host=socket.gethostname(), inputs=[{'path':str(p), 'exists':Path(p).exists()} for p in inputs],
                    snapshots_directory=str(snapshots), locations=[{'kind':'remote','path':str(out)}],
                    environment={'python':sys.executable,'python_version':platform.python_version(),
                                 'packages':{p:importlib.metadata.version(p) for p in ['ultralytics','torch','numpy']}},
                    scientific_outcome='not_assessed')
    atomic_json(out/'run.json', metadata)
    try:
        with (out/'stdout.log').open('w', buffering=1) as log, (out/'stderr.log').open('w', buffering=1) as err:
            child = subprocess.Popen(command, cwd=snapshots, stdout=log, stderr=err)
            metadata['pid'] = child.pid
            atomic_json(out/'run.json', metadata)
            code = child.wait()
        metadata.update(status='completed' if code == 0 else 'failed', exit_code=code, finished_at=now())
        atomic_json(out/'run.json', metadata)
        if code:
            raise RuntimeError(f'{run["arm"]}/{run["kind"]} exited {code}: {out}')
    except BaseException as error:
        if metadata['status'] == 'running':
            metadata.update(status='failed', finished_at=now(), error=repr(error))
            atomic_json(out/'run.json', metadata)
        raise
    return out
