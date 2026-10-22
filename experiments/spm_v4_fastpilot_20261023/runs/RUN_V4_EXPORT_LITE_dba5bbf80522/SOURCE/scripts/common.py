from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def artifact_manifest(run):
    dump(run/'ARTIFACT_MANIFEST.json', {p.relative_to(run).as_posix(): {'sha256': sha(p), 'size_bytes': p.stat().st_size}
                                      for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'})


@contextmanager
def run_context(root, stage):
    lock = root/('LOCK_'+stage)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    run = root/'runs'/('RUN_V4_'+stage.upper()+'_'+uuid.uuid4().hex[:12]); run.mkdir(parents=True)
    meta = {'study_id': 'STUDY_SPM_V4_FASTPILOT_20261023', 'run_id': run.name, 'stage': stage,
            'started_at': utc(), 'execution_status': 'running', 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
            'executor': os.uname().nodename, 'command': sys.argv, 'pid': os.getpid(),
            'protocol_sha256': sha(root/'PROTOCOL.md'), 'optimizer_steps': 0}
    for name in ['PILOT_CONFIG.json', 'TRAIN_FREEZE.json', 'DATASET.json']:
        if (root/name).exists():
            meta[name+'_sha256'] = sha(root/name)
    if (root/'TRAIN_FREEZE.json').exists():
        for name, digest in read(root/'TRAIN_FREEZE.json')['files'].items():
            assert sha(root/name) == digest, name
            target = run/'SOURCE'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/name, target)
    dump(run/'run.json', meta)
    dump(run/'EVAL_CONTRACT.json', read(root/'PILOT_CONFIG.json') if (root/'PILOT_CONFIG.json').exists() else {})
    print('RUN', run.name, flush=True)
    try:
        yield run, meta
        meta.update(execution_status='completed', ended_at=utc(), exit_code=0)
        dump(run/'run.json', meta)
        dump(root/(stage.upper()+'_POINTER.json'), {'run_id': run.name, 'slurm_job_id': meta['slurm_job_id']})
    except BaseException:
        (run/'ERROR.log').write_text(traceback.format_exc(), encoding='utf-8')
        meta.update(execution_status='failed', ended_at=utc(), exit_code=1)
        dump(run/'run.json', meta)
        raise
    finally:
        artifact_manifest(run); lock.unlink()


def state_hash(model, include_head=True):
    import torch
    h = hashlib.sha256()
    items = model.state_dict().items()
    if not include_head:
        head_prefix = f'model.{len(model.model)-1}.'
        items = [(k, t) for k, t in items if not k.startswith(head_prefix) and not k.startswith('spm.')]
    for k, t in sorted(items):
        h.update(k.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()
