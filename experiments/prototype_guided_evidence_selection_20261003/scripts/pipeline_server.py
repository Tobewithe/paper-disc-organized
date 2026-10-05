"""Bounded Linux GPU sequence. One process; strict preflights; no automatic retry."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if os.name == 'nt' or not str(root).startswith('/root/autodl-tmp/'):
        raise RuntimeError('Use the authorized Linux GPU server')
    claim = root/'PIPELINE.claim'
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    cfg = json.loads((root/'RUN_CONFIG.json').read_text())
    ids = json.loads((root/'RUN_IDS.json').read_text())
    state = dict(status='running', stage='waiting_for_verified_assets', pid=os.getpid(),
                 started_at=datetime.now(timezone.utc).isoformat(), runs=ids['runs'],
                 host='connect.bjb2.seetacloud.com:33953', automatic_retry=False)
    def save():
        temp = root/'PIPELINE_STATUS.json.tmp'
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(root/'PIPELINE_STATUS.json')
    os.environ.update(PYTHONPATH=str(root/'py'), PYTHONUTF8='1', OMP_NUM_THREADS='6')
    save()
    def run(stage, script, extra=()):
        rid = ids['runs'][stage]
        out = root/'runs'/rid
        if (out/'run.json').exists():
            raise RuntimeError(f'{stage}: existing Run, refusing duplicate launch')
        state.update(stage=stage, current_run=rid); save()
        command = [sys.executable, str(root/'runner.py'), '--study', ids['study_id'],
                   '--run-id', rid, '--cwd', str(root), '--output', str(out)]
        for source in [root/'RUN_CONFIG.json', root/'SPLIT.json', root/'assets/COMPLETE.json',
                       root/'assets/MIGRATION_IDENTITY.json', root/'models/yolo26m-seg.pt']:
            command += ['--input', str(source)]
        for source in [root/'PROTOCOL.md', root/'RUN_CONFIG.json', root/'RUN_IDS.json',
                       *sorted((root/'scripts').glob('*.py'))]:
            command += ['--snapshot', str(source)]
        command += ['--expect', str(out/'COMPLETE.json'), '--', sys.executable, '-u',
                    str(root/'scripts'/script), '--config', str(root/'RUN_CONFIG.json'),
                    '--out', str(out), *map(str,extra)]
        subprocess.run(command, cwd=root, env=os.environ.copy(), stdin=subprocess.DEVNULL, check=True)
        result = json.loads((out/'COMPLETE.json').read_text())
        if not (result.get('passed') or result.get('completed')):
            raise RuntimeError(f'{stage}: scientific/preflight receipt not passed')
        return result
    try:
        while not (root/'ASSETS_TRANSFER_COMPLETE.json').is_file():
            if (root/'ASSETS_TRANSFER_FAILURE.json').exists():
                raise RuntimeError('Transfer failed; inspect preserved transport log')
            time.sleep(30)
        receipt = json.loads((root/'ASSETS_TRANSFER_COMPLETE.json').read_text())
        if receipt.get('status') != 'complete':
            raise RuntimeError('Assets not verified as complete')
        if not (root/'CODE_READY.json').exists():
            raise RuntimeError('Implementation not yet verified for launch')
        run('replay_smoke', 'replay_smoke.py')
        run('model_smoke', 'point_smoke.py')
        run('decode_smoke', 'evaluate_points.py', ['--smoke'])
        deadline = time.time() + 3600*cfg['max_training_hours']
        state['training_deadline_unix'] = deadline; save()
        checkpoints = {}
        for arm in cfg['arms']:
            run(arm, 'train_points.py', ['--mode', arm, '--deadline', deadline])
            checkpoints[arm] = str(root/'runs'/ids['runs'][arm]/'final.pt')
        (root/'FINAL_CHECKPOINTS.json').write_text(json.dumps(checkpoints,indent=2))
        run('evaluation','evaluate_points.py',['--checkpoints',root/'FINAL_CHECKPOINTS.json'])
        state.update(status='completed',stage='finished',finished_at=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc(),
                     finished_at=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == '__main__':
    main()
