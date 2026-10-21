"""One-shot autonomous evaluation; no automatic scientific retries or training."""
import os, subprocess, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
os.environ.update(PYTHONPATH=str(ROOT / '_site'), YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                  YOLO_CONFIG_DIR=str(ROOT / '_ultralytics_config'), OMP_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
os.environ.pop('MKL_NUM_THREADS', None)
from lea_common_v2 import STUDY, check_ownership, read_json, write_new, save, now, verify_files


def main():
    check_ownership()
    status_path = STUDY / 'AUTONOMOUS_STATUS_V2.json'
    if status_path.exists():
        raise RuntimeError('Existing autonomous state; inspect, never restart automatically')
    claim = STUDY / '.AUTONOMOUS_OWNER.guard'
    state = dict(study_id='STUDY_MASK_LOCAL_EDITABILITY_20261017', pid=os.getpid(),
                 started_at=now(), status='initializing', phases=[], new_training_steps=0)
    write_new(claim, state)
    save(status_path, state)

    def child(label, script, args, timeout):
        state.update(current_phase=label, status='running')
        save(status_path, state)
        command = [sys.executable, '-B', str(STUDY / 'scripts' / script), *args]
        with (STUDY / (label + '.stdout.log')).open('x', encoding='utf8') as stdout, \
             (STUDY / (label + '.stderr.log')).open('x', encoding='utf8') as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            state['child_pid'] = proc.pid
            save(status_path, state)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
                state['timeout'] = label
            state['phases'].append(dict(label=label, pid=proc.pid, command=command, exit_code=code, ended_at=now()))
            state['child_pid'] = None
            save(status_path, state)
        if code != 0 or state.get('timeout'):
            raise RuntimeError('Stage failed or timed out: ' + label)

    def one(scope):
        runs = list((STUDY / 'runs').glob('RUN_LEA_' + scope.upper() + '_*'))
        if len(runs) != 1:
            raise RuntimeError('Expected unique Run: ' + scope)
        return runs[0]

    try:
        verify_files(read_json(STUDY / 'EXECUTION_FREEZE_V2.json')['files'])
        child('IMPORT_TESTS_V2', 'test_imports_v2.py', [], 120)
        child('SMOKE_RECOVERY_V2', 'recover_smoke_v2.py', [], 720)
        smoke = one('smoke_recovery')
        state['smoke_run'] = smoke.name
        child('SMOKE_VERIFY_V2', 'lea_verify_v2.py', [str(smoke), '--replay', '--output', 'INDEPENDENT_LAPTOP.json'], 720)
        child('FULL_V2', 'lea_run_v2.py', ['--scope', 'full', '--parent-run', smoke.name], 3720)
        full = one('full')
        state['full_run'] = full.name
        child('FULL_VERIFY_V2', 'lea_verify_v2.py', [str(full), '--replay', '--output', 'INDEPENDENT_LAPTOP.json'], 3720)
        state.update(status='completed_and_independently_verified', current_phase='awaiting_transfer_no_training',
                     scientific_decision=read_json(full / 'SUMMARY.json')['decision'])
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        (STUDY / 'CONTROLLER_ERROR_V2.txt').write_text(state['error'], encoding='utf8')
        raise
    finally:
        state['ended_at'] = now()
        state['child_pid'] = None
        if claim.exists() and read_json(claim).get('pid') == os.getpid():
            claim.unlink()
        disable = subprocess.run(
            ['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-Command',
             "Disable-ScheduledTask -TaskName 'Codex-Local-Editability-Recovery-20261017' | Out-Null"],
            capture_output=True, text=True)
        state['disable_task_exit_code'] = disable.returncode
        save(status_path, state)


if __name__ == '__main__':
    main()
