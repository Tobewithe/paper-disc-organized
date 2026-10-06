"""Read only this experiment's processes and recent logs."""
import json
import sys
from pathlib import Path
import psutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

root = Path('D:/coco_wire/experiments/acd_native_coefficient_20261006')
for path in sorted((root / 'runs').glob('*/run.json')):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    rid = data['run_id']
    print(rid, data.get('execution_status', data.get('status')),
          'COMPLETE' if (path.parent / 'TRAINING_COMPLETE.json').exists() else '')
    if data.get('execution_status') == 'running':
        log = path.parent / 'stdout.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 2400))
                print(f.read().decode('utf-8', errors='replace'))
    if (path.parent / 'TRAINING_COMPLETE.json').exists():
        r = json.loads((path.parent / 'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
        print('receipt', json.dumps({'arm': r['arm'], 'epochs': r['epochs'], 'stats': r['stats'],
                                     'frozen': r['audit']['frozen_state_unchanged'],
                                     'input_batches': r['audit']['input_batches']}))
for proc in psutil.process_iter(['pid', 'cmdline']):
    try:
        cmd = proc.info['cmdline'] or []
        if any('acd_native_coefficient_20261006' in s for s in cmd) and not any('status.py' in s for s in cmd):
            print('process', proc.info)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
