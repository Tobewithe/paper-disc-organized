"""Finish this finite experiment with a recorded, non-selective analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for key in ['root', 'learned-reference', 'runner']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    status_path = args.root/'analysis_status.json'
    def status(value, **kwargs):
        status_path.write_text(json.dumps({'status': value, 'run_id': args.run_id, **kwargs}), encoding='utf-8')
    try:
        status('waiting_for_three_evaluations')
        deadline = time.monotonic()+10800
        while True:
            queue = json.loads((args.root/'queue_status.json').read_text())
            if queue['status'] == 'completed':
                break
            if queue['status'] == 'failed':
                raise RuntimeError('Experiment queue failed: '+str(queue))
            if time.monotonic() >= deadline:
                raise TimeoutError('Three-hour analysis wait limit reached; no process interrupted or retried.')
            time.sleep(10)
        ids = json.loads((args.root/'execution_ids.json').read_text())
        study = json.loads((args.root/'study.json').read_text())['study_id']
        out = args.root/'runs'/args.run_id
        script = args.root/'scripts/summarize_template_sources.py'
        inputs = [args.root/'execution_ids.json', args.root/'PROTOCOL.json']
        for base in [args.learned_reference]+[args.root/'runs'/ids[arm+'_eval'] for arm in ['analytic_center','shuffled_learned','fit_residual']]:
            inputs.extend(base/name for name in ['run.json', 'RESULTS.json', 'MATCHED_GT75.json'])
        inputs += [args.root/'runs'/ids[arm+'_train']/'SELECTION.json' for arm in ['analytic_center','shuffled_learned','fit_residual']]
        command = [sys.executable, str(args.runner), '--study', study, '--run-id', args.run_id,
                   '--output', str(out), '--cwd', str(args.root), '--snapshot', str(script)]
        for item in inputs:
            command += ['--input', str(item)]
        for name in ['ANALYSIS.json', 'RESULTS.md', 'COMPLETE.json']:
            command += ['--expect', str(out/name)]
        command += ['--', sys.executable, '-u', str(script), '--root', str(args.root),
                    '--learned-reference', str(args.learned_reference), '--out', str(out)]
        status('analyzing')
        subprocess.run(command, check=True)
        status('completed', result=str(out/'RESULTS.md'))
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
