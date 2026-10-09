"""Quiet read-only wait: emit only the owned pipeline's terminal state."""
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
args = parser.parse_args()
deadline = time.monotonic() + 12 * 3600
while time.monotonic() < deadline:
    try:
        value = json.loads((args.run / 'run.json').read_text(encoding='utf-8-sig'))
        if value.get('execution_status') in ('completed', 'failed'):
            print(json.dumps({'run_id': value['run_id'], 'execution_status': value['execution_status'],
                              'exit_code': value.get('exit_code'), 'ended_at': value.get('ended_at')}), flush=True)
            break
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    time.sleep(30)
else:
    print(json.dumps({'status': 'wait_timeout', 'run': str(args.run)}), flush=True)
