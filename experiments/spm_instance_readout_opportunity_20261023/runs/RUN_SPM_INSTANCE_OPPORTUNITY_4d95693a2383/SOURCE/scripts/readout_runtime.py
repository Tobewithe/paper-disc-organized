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
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


@contextmanager
def context(root):
    lock = root/'LOCK_OPPORTUNITY'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    run = root/'runs'/('RUN_SPM_INSTANCE_OPPORTUNITY_'+uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    meta = dict(study_id='STUDY_SPM_INSTANCE_READOUT_OPPORTUNITY_20261023', run_id=run.name,
                started_at=utc(), execution_status='running', slurm_job_id=os.getenv('SLURM_JOB_ID'),
                executor=platform.node(), pid=os.getpid(), command=sys.argv, optimizer_steps=0,
                rendered_images=0, protocol_sha256=sha(root/'PROTOCOL.md'))
    dump(run/'run.json', meta)
    print('RUN', run.name, flush=True)
    try:
        freeze = read(root/'FREEZE.json')
        for name, digest in freeze['files'].items():
            source = root/name
            assert sha(source) == digest, name
            target = run/'SOURCE'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for name, digest in freeze['external_files'].items():
            assert sha(name) == digest, name
        shutil.copy2(root/'FREEZE.json', run/'SOURCE/FREEZE.json')
        cfg = read(root/'CONFIG.json')
        dump(run/'EVAL_CONTRACT.json', cfg)
        yield run, meta, cfg
        meta.update(execution_status='completed', ended_at=utc(), exit_code=0)
        dump(run/'run.json', meta)
        dump(root/'OPPORTUNITY_POINTER.json', dict(run_id=run.name, slurm_job_id=meta['slurm_job_id']))
    except BaseException:
        (run/'ERROR.log').write_text(traceback.format_exc(), encoding='utf-8')
        meta.update(execution_status='failed', ended_at=utc(), exit_code=1)
        dump(run/'run.json', meta)
        raise
    finally:
        dump(run/'ARTIFACT_MANIFEST.json', {
            p.relative_to(run).as_posix(): dict(sha256=sha(p), size_bytes=p.stat().st_size)
            for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'})
        lock.unlink()
