"""Execute the bounded numerical verification in its own Run."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rid = 'RUN_PROTO_TAIL_NUMERICAL_VERIFY_S0'
run = root / 'runs' / rid
run.mkdir(parents=True, exist_ok=True)
assert not (run / 'TRAINING_VERIFICATION.json').exists(), 'Use a new Run for retries'
source = run / 'source'
source.mkdir(exist_ok=True)
files = ('verify_training.py', 'train_acd.py')
hashes = {}
for name in files:
    raw = (root / 'scripts' / name).read_bytes()
    (source / name).write_bytes(raw)
    hashes[name] = hashlib.sha256(raw).hexdigest()
protocol = (root / 'PROTOCOL.md').read_bytes()
(source / 'PROTOCOL.md').write_bytes(protocol)
now = lambda: datetime.now(timezone.utc).isoformat()
meta = {'run_id': rid, 'purpose': 'Real-forward numerical/gradient verification; synthetic labels, no AP claim',
        'status': 'running', 'execution_status': 'running', 'artifact_status': 'partial',
        'transfer_status': 'not_started', 'started_at': now(), 'ended_at': None,
        'environment': 'laptop_28358lan', 'interpreter': sys.executable,
        'remote_directory': str(run), 'local_directory':
        'C:/Dpan/codexproject/paper-disc-organized/experiments/acd_proto_tail_unfreeze_20261006/runs/' + rid}
(run / 'SOURCE.json').write_text(json.dumps({'script_sha256': hashes['verify_training.py'],
    'executed_snapshot': str(source / 'verify_training.py'), 'training_script_sha256': hashes['train_acd.py'],
    'protocol_sha256': hashlib.sha256(protocol).hexdigest(), 'captured_at': now()}, indent=2), encoding='utf-8')
cmd = [sys.executable, str(source / 'verify_training.py'), '--vendor', 'D:/coco_wire/vendor_8.4.100',
       '--weight', 'D:/coco_wire/models/yolo26m-seg.pt', '--output', str(run / 'TRAINING_VERIFICATION.json'),
       '--device', '0', '--imgsz', '128']
meta['command'] = cmd
(run / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
with (run / 'stdout.log').open('w', encoding='utf-8') as output:
    process = subprocess.Popen(cmd, cwd=str(root), stdout=output, stderr=subprocess.STDOUT)
    (run / 'PROCESS.json').write_text(json.dumps({'pid': process.pid, 'command': cmd}), encoding='utf-8')
    code = process.wait()
meta.update(status='completed' if code == 0 else 'failed', execution_status='completed' if code == 0 else 'failed',
            exit_code=code, ended_at=now(), artifact_status='generated' if code == 0 else 'partial')
(run / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
receipt = json.loads((run / 'TRAINING_VERIFICATION.json').read_text()) if (run / 'TRAINING_VERIFICATION.json').exists() else {}
print(json.dumps({'run': rid, 'exit_code': code, 'status': receipt.get('status'),
                  'exception': receipt.get('exception'), 'trainable_parameter_count': receipt.get('trainable_parameter_count')}), flush=True)
sys.exit(code)
