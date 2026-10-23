"""Frozen sources, atomic run ownership, immutable failed runs and receipts."""
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


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def context(root, stage):
    lock = root/'LOCK_TRANSFER'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({'pid': os.getpid(), 'stage': stage, 'host': platform.node()}).encode()); os.close(fd)
    run = root/'runs'/('RUN_SPM_REAL_OBJECT_'+stage.upper()+'_'+uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    cfg = read(root/'CONFIG.json')
    meta = dict(study_id=cfg['study_id'], run_id=run.name, stage=stage, command=sys.argv,
                executor=platform.node(), pid=os.getpid(), started_at=utc(),
                execution_status='running', optimizer_steps=0, slurm_job_id=os.getenv('SLURM_JOB_ID'),
                protocol_sha256=sha(root/'PROTOCOL.md'), config_sha256=sha(root/'CONFIG.json'),
                source_freeze_sha256=sha(root/'FREEZE.json'))
    dump(run/'run.json', meta); dump(run/'EVAL_CONTRACT.json', cfg)
    print('RUN', run.name, flush=True)
    try:
        frozen = read(root/'FREEZE.json')
        authorization = read(root/'EXECUTION_AUTHORIZATION.json')
        assert authorization['protocol_sha256'] == sha(root/'PROTOCOL.md')
        assert authorization['config_sha256'] == sha(root/'CONFIG.json')
        meta['code_commit'] = frozen['code_commit']
        for name, digest in frozen['files'].items():
            assert sha(root/name) == digest, name
            target = run/'SOURCE'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root/name, target)
        for name, digest in frozen['external_files'].items(): assert sha(name) == digest, name
        shutil.copy2(root/'FREEZE.json', run/'SOURCE/FREEZE.json')
        yield run, meta, cfg
        meta.update(execution_status='completed', ended_at=utc(), exit_code=0)
        dump(run/'run.json', meta)
        dump(root/(stage.upper()+'_POINTER.json'), {'run_id': run.name, 'slurm_job_id': meta['slurm_job_id']})
    except BaseException:
        (run/'ERROR.log').write_text(traceback.format_exc(), encoding='utf-8')
        meta.update(execution_status='failed', ended_at=utc(), exit_code=1)
        dump(run/'run.json', meta)
        raise
    finally:
        dump(run/'ARTIFACT_MANIFEST.json', {p.relative_to(run).as_posix():
             {'sha256': sha(p), 'size_bytes': p.stat().st_size} for p in sorted(run.rglob('*'))
             if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'})
        lock.unlink()
