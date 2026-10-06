"""Run the locked smoke, paired training, and independent COCO evaluation on the laptop."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

PIPELINE_RUN = 'RUN_ACD_PIPELINE_S0'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def update(root, rid, **values):
    p = root / 'runs' / rid / 'run.json'
    data = read(p) if p.exists() else {'run_id': rid}
    data.update(values)
    dump(p, data)


def execute(root, rid, args):
    out = root / 'runs' / rid
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'TRAINING_COMPLETE.json').exists() and not (out / 'SUMMARY.json').exists(), 'Run already complete; preserve it'
    env = os.environ.copy()
    env.update(PYTHONPATH='D:/coco_wire/vendor_8.4.100', PYTHONUNBUFFERED='1',
               PYTHONUTF8='1', YOLO_OFFLINE='true', WANDB_MODE='disabled')
    source_script = Path(args[0]).resolve()
    snapshot = out / 'source' / source_script.name
    snapshot.parent.mkdir(exist_ok=True)
    script_bytes = source_script.read_bytes()
    snapshot.write_bytes(script_bytes)
    protocol_bytes = (root / 'PROTOCOL.md').read_bytes()
    (snapshot.parent / 'PROTOCOL.md').write_bytes(protocol_bytes)
    dump(out / 'SOURCE.json', {'source_script': str(source_script), 'executed_snapshot': str(snapshot),
                              'script_sha256': hashlib.sha256(script_bytes).hexdigest(),
                              'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
                              'captured_at': now()})
    cmd = [sys.executable, str(snapshot), *args[1:]]
    update(root, rid, status='running', execution_status='running', started_at=now(),
           command=cmd, artifact_status='partial', unverified=['completion and results pending'])
    dump(root / 'runs' / PIPELINE_RUN / 'PROGRESS.json', {'stage': rid, 'updated_at': now()})
    print('START', rid, flush=True)
    with (out / 'stdout.log').open('w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=f, stderr=subprocess.STDOUT)
        dump(out / 'PROCESS.json', {'pid': proc.pid, 'started_at': now(), 'command': cmd})
        code = proc.wait()
    update(root, rid, status='completed' if code == 0 else 'failed',
           execution_status='completed' if code == 0 else 'failed', exit_code=code,
           ended_at=now(), artifact_status='generated' if code == 0 else 'partial',
           unverified=[] if code == 0 else ['failed execution; partial artifacts retained'])
    print('END', rid, code, flush=True)
    if code:
        raise RuntimeError(f'{rid}: exit {code}, see stdout.log')


def verify_identity(root, baseline, acd, name):
    p = root / 'runs'
    a = p / baseline / 'input_batch_hashes.jsonl'
    b = p / acd / 'input_batch_hashes.jsonl'
    av = [json.loads(x) for x in a.read_text(encoding='utf-8').splitlines() if x.strip()]
    bv = [json.loads(x) for x in b.read_text(encoding='utf-8').splitlines() if x.strip()]
    good = bool(av) and av == bv
    receipt = {'baseline': baseline, 'acd': acd, 'steps': len(av), 'same_transformed_inputs': good,
               'baseline_sha256': hashlib.sha256(a.read_bytes()).hexdigest(),
               'acd_sha256': hashlib.sha256(b.read_bytes()).hexdigest()}
    dump(p / PIPELINE_RUN / name, receipt)
    assert good, receipt


def verify_optimizer_schedule(root, baseline, acd):
    p = root / 'runs'
    rows = {}
    for arm, rid in [('baseline', baseline), ('acd', acd)]:
        rows[arm] = [json.loads(line) for line in (p / rid / 'optimizer_step_audit.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    keys = ['epoch', 'attempt', 'input_batches_seen', 'learning_rates', 'accumulate', 'amp', 'scaler_enabled']
    schedules = {arm: [{k: row[k] for k in keys} for row in records] for arm, records in rows.items()}
    equal = bool(schedules['baseline']) and schedules['baseline'] == schedules['acd']
    receipt = {'baseline': baseline, 'acd': acd, 'same_attempt_schedule': equal,
               'attempts': {arm: len(records) for arm, records in rows.items()},
               'applied': {arm: sum(row['optimizer_applied'] for row in records) for arm, records in rows.items()},
               'overflow_skips': {arm: sum(row['skipped_overflow_step'] for row in records) for arm, records in rows.items()},
               'limitation': 'Actual applied-update counts may differ through stock AMP; do not force equality by changing scale or settings.'}
    dump(p / PIPELINE_RUN / 'OPTIMIZER_SCHEDULE_PARITY.json', receipt)
    assert equal, receipt


def main():
    global PIPELINE_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--pipeline-run', default=PIPELINE_RUN)
    ap.add_argument('--baseline-smoke-run', default='RUN_BASELINE_SMOKE_S0')
    ap.add_argument('--acd-smoke-run', default='RUN_ACD_SMOKE_S0')
    ap.add_argument('--baseline-feasibility-run', default='RUN_BASELINE_FEASIBILITY_S0')
    a = ap.parse_args()
    root = a.root.resolve()
    PIPELINE_RUN = a.pipeline_run
    pipeline = root / 'runs' / PIPELINE_RUN
    pipeline.mkdir(parents=True, exist_ok=True)
    update(root, PIPELINE_RUN, status='running', execution_status='running',
           started_at=now(), interpreter=sys.executable, remote_directory=str(root), pid=os.getpid())
    try:
        trainer = root / 'scripts' / 'train_acd.py'
        for stage, epochs in [('SMOKE', 1), ('FEASIBILITY', 3)]:
            for arm in ['baseline', 'acd']:
                rid = f'RUN_{arm.upper()}_{stage}_S0'
                if stage == 'SMOKE' and arm == 'baseline':
                    rid = a.baseline_smoke_run
                elif stage == 'SMOKE' and arm == 'acd':
                    rid = a.acd_smoke_run
                elif stage == 'FEASIBILITY' and arm == 'baseline':
                    rid = a.baseline_feasibility_run
                data = root / 'data' / ('smoke.yaml' if stage == 'SMOKE' else 'fit.yaml')
                execute(root, rid, [str(trainer), '--arm', arm, '--data', str(data),
                                   '--weight', 'D:/coco_wire/models/yolo26m-seg.pt',
                                   '--out', str(root / 'runs' / rid), '--epochs', str(epochs),
                                   '--batch', '2', '--workers', '0', '--seed', '0'])
                receipt = read(root / 'runs' / rid / 'TRAINING_COMPLETE.json')
                assert receipt['audit']['finite_training_loss'] and receipt['audit']['frozen_state_unchanged'], receipt
            verify_identity(root, a.baseline_smoke_run if stage == 'SMOKE' else a.baseline_feasibility_run, a.acd_smoke_run if stage == 'SMOKE' else f'RUN_ACD_{stage}_S0', stage + '_PARITY.json')
            if stage == 'FEASIBILITY':
                verify_optimizer_schedule(root, a.baseline_feasibility_run, 'RUN_ACD_FEASIBILITY_S0')
            if stage == 'SMOKE':
                s = read(root / 'runs' / a.acd_smoke_run / 'TRAINING_COMPLETE.json')
                assert sum(v['action_positive'] for v in s['stats'].values()) > 0, 'Smoke had no accepted auxiliary term'
        models = {'official': 'D:/coco_wire/models/yolo26m-seg.pt',
                  'baseline': {'base_weights': 'D:/coco_wire/models/yolo26m-seg.pt', 'coefficients': str(root / 'runs' / a.baseline_feasibility_run / 'coeff_final_ema.pt')},
                  'acd': {'base_weights': 'D:/coco_wire/models/yolo26m-seg.pt', 'coefficients': str(root / 'runs' / 'RUN_ACD_FEASIBILITY_S0' / 'coeff_final_ema.pt')}}
        models_path = pipeline / 'MODELS.json'
        dump(models_path, models)
        ev = 'RUN_ACD_PAIRED_EVAL_S0'
        execute(root, ev, [str(root / 'scripts' / 'evaluate.py'), '--models-json', str(models_path),
                          '--images-list', str(root / 'data' / 'val_full.txt'),
                          '--annotations', 'D:/coco_wire/data/annotations/instances_val2017.json',
                          '--out', str(root / 'runs' / ev), '--bootstrap-seed', '0'])
        dump(pipeline / 'COMPLETE.json', {'completed': True, 'ended_at': now(),
                                         'scope': 'Locked smoke, paired three-epoch training and full val2017 evaluation'})
        update(root, PIPELINE_RUN, status='completed', execution_status='completed', ended_at=now(), exit_code=0)
    except Exception:
        dump(pipeline / 'FAILED.json', {'time': now(), 'traceback': traceback.format_exc()})
        update(root, PIPELINE_RUN, status='failed', execution_status='failed', ended_at=now(), exit_code=1)
        raise


if __name__ == '__main__':
    main()
