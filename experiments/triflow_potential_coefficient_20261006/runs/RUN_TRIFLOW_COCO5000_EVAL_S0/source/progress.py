"""Read only this experiment's factual stage status."""
import json
from pathlib import Path

ROOT = Path('D:/coco_wire/experiments/triflow_potential_coefficient_20261006')
for run in sorted((ROOT/'runs').glob('*')):
    metadata = run/'run.json'
    if not metadata.is_file():
        continue
    meta = json.loads(metadata.read_text(encoding='utf-8-sig'))
    value = {'run': run.name, 'state': meta.get('execution_status'), 'exit_code': meta.get('exit_code')}
    if value['state'] in ('running', 'failed'):
        log = run/'stdout.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                lines = stream.read().decode('utf-8', errors='replace').splitlines()
            value['tail'] = lines[-3:]
    print(json.dumps(value), flush=True)
