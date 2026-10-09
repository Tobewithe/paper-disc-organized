import json
import re
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
root = Path('D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs')
for path in sorted(root.glob('*/run.json')):
    state = json.loads(path.read_text(encoding='utf-8-sig'))
    if state.get('execution_status') not in ['running', 'failed', 'completed']:
        continue
    rid = state['run_id']
    value = {'run': rid, 'state': state['execution_status']}
    log = path.parent / 'stdout.log'
    if log.exists() and state['execution_status'] == 'running':
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 6000))
            tail = f.read().decode('utf-8', errors='replace')
        tail = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', tail)
        lines = [s.strip() for s in re.split('[\r\n]', tail) if s.strip()]
        relevant = [s for s in lines if re.search(r'\d/3\s|\d/1\s|PREDICT |COCO_AP |START |END |PAIRED |FAILED|Error:|EVALUATION_COMPLETE|Class.*Images', s)]
        value['latest'] = relevant[-1][:500] if relevant else (lines[-1][:250] if lines else '')
        pair_progress = {}
        for image_records in sorted(path.parent.glob('paired_*/IMAGES.jsonl')):
            with image_records.open('rb') as pair_stream:
                rows = sum(chunk.count(b'\n') for chunk in iter(lambda: pair_stream.read(1024 * 1024), b''))
            pair_progress[image_records.parent.name] = {'image_rows_written': rows}
        if pair_progress:
            value['paired_progress'] = pair_progress
    if (path.parent / 'trainer' / 'results.csv').is_file():
        csv = (path.parent / 'trainer' / 'results.csv').read_text(encoding='utf-8').splitlines()
        value['epochs_logged'] = max(0, len(csv)-1)
    step_log = path.parent / 'optimizer_step_audit.jsonl'
    if state['execution_status'] == 'running' and step_log.is_file():
        records = []
        for line in step_log.read_text(encoding='utf-8').splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        value['optimizer_attempts_logged'] = len(records)
        value['optimizer_applied_logged'] = sum(r['optimizer_applied'] for r in records)
        value['overflow_skips_logged'] = sum(r['skipped_overflow_step'] for r in records)
    print(json.dumps(value, ensure_ascii=False))
