"""Single bounded server process for the independently registered rapid screen."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    if os.name == 'nt' or str(root) != '/root/prototype_readout_fast_screen_20261003':
        raise RuntimeError('Only the authorized server root may run this pipeline')
    fd = os.open(root/'PIPELINE.claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    cfg = json.loads((root/'RUN_CONFIG.json').read_text())
    identity = json.loads((root/'RUN_IDS.json').read_text())
    ids = identity['runs']
    source = Path(cfg['source_root'])
    environment = dict(os.environ, PYTHONPATH=cfg['source_python'], PYTHONUTF8='1', OMP_NUM_THREADS='6')
    state = dict(status='running', stage='waiting_for_preparation_code', pid=os.getpid(),
                 started_at=datetime.now(timezone.utc).isoformat(), runs=ids,
                 scope='rapid screening; not confirmation', automatic_retry=False)

    def save():
        temp = root/'PIPELINE_STATUS.json.tmp'
        temp.write_text(json.dumps(state, indent=2), encoding='utf8')
        temp.replace(root/'PIPELINE_STATUS.json')

    def ready(name):
        start = time.monotonic()
        while not (root/name).exists():
            if time.monotonic()-start > 3600:
                raise TimeoutError('Implementation readiness not published within one hour: '+name)
            time.sleep(10)

    def run(stage, script, extra=()):
        out = root/'runs'/ids[stage]
        if (out/'run.json').exists():
            raise RuntimeError('Refusing duplicate Run: '+stage)
        command = [sys.executable, str(source/'runner.py'), '--study', identity['study_id'],
                   '--run-id', ids[stage], '--cwd', str(root), '--output', str(out)]
        for f in [root/'RUN_CONFIG.json', root/'SPLIT.json', Path(cfg['weights'])]:
            command += ['--input', str(f)]
        for f in [root/'PROTOCOL.md', root/'RUN_CONFIG.json', root/'RUN_IDS.json',
                  *sorted((root/'scripts').glob('*.py'))]:
            command += ['--snapshot', str(f)]
        command += ['--expect', str(out/'COMPLETE.json'), '--', sys.executable, '-u',
                    str(root/'scripts'/script), '--config', str(root/'RUN_CONFIG.json'),
                    '--out', str(out), *map(str, extra)]
        state.update(stage=stage, current_run=ids[stage]); save()
        child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL)
        state['child_pid'] = child.pid; save()
        code = child.wait()
        state.pop('child_pid', None)
        if code:
            raise RuntimeError(f'{stage} execution failed, exit={code}; preserve Run and stop')
        report = json.loads((out/'COMPLETE.json').read_text())
        if not (report.get('passed') or report.get('completed') or report.get('complete')):
            raise RuntimeError(stage+' did not complete its required checks')
        return report

    try:
        save()
        ready('PREPARATION_CODE_READY.json')
        run('prepare', 'prepare_online.py')
        state['stage'] = 'waiting_for_training_code'; save()
        ready('CODE_READY.json')
        run('smoke', 'screen_smoke.py')
        deadline = time.time()+3600*cfg['max_training_hours']
        state['training_deadline_unix'] = deadline; save()
        checkpoints = {}
        for arm in cfg['arms']:
            run(arm, 'train_screen.py', ['--mode', arm, '--deadline', deadline])
            checkpoints[arm] = str(root/'runs'/ids[arm]/'final.pt')
        target = root/'FINAL_CHECKPOINTS.json'
        target.write_text(json.dumps(checkpoints, indent=2))
        run('evaluation', 'evaluate_screen.py', ['--checkpoints', target])
        state.update(status='completed', stage='finished', finished_at=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as exc:
        state.update(status='failed', error=str(exc), traceback=traceback.format_exc(),
                     finished_at=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == '__main__':
    main()
