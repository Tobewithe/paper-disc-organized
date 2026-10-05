"""Single pipeline dependency, exits after its one analysis run."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1]
    ids=json.loads((root/'execution_ids.json').read_text())
    source=root/'runs'/ids['supervision_run'];out=root/'runs'/ids['analysis_run']
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        if (source/'run.json').exists():
            state=json.loads((source/'run.json').read_text())['status']
            if state=='completed':break
            if state in ['failed','cancelled','interrupted']:raise RuntimeError('Upstream probe '+state)
        state=json.loads((root/'queue_status.json').read_text())
        if state.get('status')=='failed':raise RuntimeError('Upstream queue failed')
        time.sleep(5)
    else:raise TimeoutError('Probe dependency did not complete within one hour')
    command=[sys.executable,str(root/'runner.py'),'--study',ids['study_id'],'--run-id',ids['analysis_run'],
             '--output',str(out),'--cwd',str(root),'--snapshot',str(root/'scripts/analyze.py'),
             '--input',str(source/'instances.jsonl'),'--expect',str(out/'SUMMARY.json'),
             '--',sys.executable,str(root/'scripts/analyze.py'),'--source',str(source),'--out',str(out)]
    raise SystemExit(subprocess.call(command))


if __name__=='__main__':main()
