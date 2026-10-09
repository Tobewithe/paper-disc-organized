"""Restore observed diagnostic status after planned metadata was copied over it."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006')
for rid, filename in (('RUN_PROTO_TAIL_NUMERICAL_VERIFY_S0', 'TRAINING_VERIFICATION.json'),
                      ('RUN_PROTO_TAIL_LOADER_VERIFY_S0', 'LOADER_VERIFICATION.json')):
    run = ROOT / 'runs' / rid
    receipt = json.loads((run / filename).read_text(encoding='utf-8-sig'))
    assert receipt['status'] == 'PASS'
    meta = json.loads((run / 'run.json').read_text(encoding='utf-8-sig'))
    meta.update(status='completed', execution_status='completed', exit_code=0, artifact_status='generated',
                started_at=None, ended_at=None,
                unverified=['Exact wrapper start/end timestamps unavailable after metadata reconciliation'],
                verification_receipt=filename,
                diagnostic_started_at=receipt.get('started_at_utc'),
                diagnostic_completed_at=receipt.get('completed_at_utc', receipt.get('completed_at')),
                metadata_reconciled_at=datetime.now(timezone.utc).isoformat(),
                metadata_reconciliation_reason='Copying planned Run templates overwrote diagnostic run.json; observed zero exit codes and preserved PASS receipts restore status, not unknown wrapper timestamps.')
    if (run / 'PROCESS.json').exists():
        meta['command'] = json.loads((run / 'PROCESS.json').read_text())['command']
    else:
        meta['command'] = ['C:/Users/28358/anaconda3/envs/pytorch/python.exe',
                           str(ROOT / 'scripts' / 'verify_loader.py')]
    (run / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if rid.endswith('LOADER_VERIFY_S0'):
        source_path = run / 'SOURCE.json'
        source = json.loads(source_path.read_text())
        source.update(executed_script=str(ROOT / 'scripts' / 'verify_loader.py'),
                      captured_snapshot=str(run / 'source' / 'verify_loader.py'),
                      executed_snapshot=str(ROOT / 'scripts' / 'verify_loader.py'),
                      execution_mode='Root driver executed with a byte-identical contemporaneous source snapshot; evaluator imported from Run/source')
        source_path.write_text(json.dumps(source, indent=2), encoding='utf-8')
    print(json.dumps({'run': rid, 'status': meta['status'], 'exit_code': 0}), flush=True)
