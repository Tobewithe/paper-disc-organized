"""Recover the preserved durable payload after the owned process was stopped."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from frozen_io import dump_json, sha256

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--run-id', required=True)
a = p.parse_args()
old = a.root / 'runs/RUN_TRIFLOW_20K_TRAIN_S0'
control = a.root / 'runs' / a.run_id
meta = json.loads((old / 'run.json').read_text(encoding='utf-8-sig'))
request = json.loads((old / 'USER_EPOCH_CONTROL_CHANGE.json').read_text(encoding='utf-8-sig'))
if meta.get('execution_status') != 'failed' or request.get('reason') != 'user_requested_evaluation_at_every_epoch':
    raise ValueError('Actual controlled stop required; no running-writer copy')
record = json.loads((old / 'CHECKPOINT_LATEST.json').read_text())
source = old / 'checkpoint_latest.pt'
if sha256(source) != record['sha256']:
    raise ValueError('Actual preserved durable payload and receipt differ')
target = control / 'resume_snapshot.pt'
shutil.copyfile(source, target)
if sha256(target) != record['sha256']:
    raise ValueError('Recovered immutable continuation payload differs')
receipt = {'passed': True, 'source_run': old.name, 'source_pid': meta['pid'],
           'checkpoint_sha256': record['sha256'], 'checkpoint_receipt': record,
           'resume_snapshot': str(target), 'scientific_training_budget': 8,
           'recovery_reason': 'Owned stop completed; Windows PowerShell lacked Get-FileHash during copy bookkeeping',
           'failed_controller_preserved': 'RUN_TRIFLOW_20K_EPOCH_CONTINUATION_S0',
           'history_preserved': True, 'created_at': datetime.now(timezone.utc).isoformat()}
dump_json(control / 'CONTINUATION_RECEIPT.json', receipt)
print(json.dumps(receipt), flush=True)
