"""Run records, stage locks and byte-level source contracts for object v0."""
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
import uuid


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify(root):
    freeze = read(root/'FREEZE_V0.json')
    for rel, digest in freeze['files'].items():
        assert sha(root/rel) == digest, ('source_changed', rel)
    return freeze


@contextmanager
def context(root, stage, authorization=False):
    root = Path(root).resolve()
    cfg = read(root/'CONFIG_V0.json')
    if authorization:
        approval = read(root/'EXECUTION_AUTHORIZATION.json')
        assert approval['formal_execution_authorized'] is True
        assert approval['config_sha256'] == sha(root/'CONFIG_V0.json')
        assert approval['protocol_sha256'] == sha(root/'PROTOCOL_V0.md')
    freeze = verify(root)
    run_id = 'RUN_SPM_OBJECT_'+stage.upper()+'_'+uuid.uuid4().hex[:12]
    lock = root/('LOCK_'+stage.upper())
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({'run_id': run_id, 'host': platform.node(), 'pid': os.getpid()}).encode())
    os.close(fd)
    run = root/'runs'/run_id
    run.mkdir(parents=True)
    meta = dict(study_id=cfg['study_id'], run_id=run_id, stage=stage, started_at=utc(),
                execution_status='running', executor=platform.node(), pid=os.getpid(),
                slurm_job_id=os.getenv('SLURM_JOB_ID'), command=sys.argv, optimizer_steps=0,
                rendered_images=0, protocol_sha256=sha(root/'PROTOCOL_V0.md'),
                config_sha256=sha(root/'CONFIG_V0.json'), code_commit=freeze['code_commit'])
    dump(run/'run.json', meta)
    dump(run/'EVAL_CONTRACT.json', cfg)
    for rel in freeze['files']:
        dst = run/'SOURCE'/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/rel, dst)
    shutil.copyfile(root/'FREEZE_V0.json', run/'SOURCE/FREEZE_V0.json')
    print('RUN', run_id, flush=True)
    try:
        yield run, meta, cfg
        meta.update(execution_status='completed', exit_code=0)
        dump(root/(stage.upper()+'_POINTER.json'), {'run_id': run_id, 'slurm_job_id': meta['slurm_job_id']})
    except BaseException:
        meta.update(execution_status='failed', exit_code=1)
        (run/'ERROR.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        meta['ended_at'] = utc()
        dump(run/'run.json', meta)
        dump(run/'ARTIFACT_MANIFEST.json', {
            p.relative_to(run).as_posix(): {'sha256': sha(p), 'size_bytes': p.stat().st_size}
            for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'})
        lock.unlink()
