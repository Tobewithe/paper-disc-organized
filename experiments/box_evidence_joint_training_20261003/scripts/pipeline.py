"""One independent laptop joint-training sequence, then unchanged-mask evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    cfg = json.loads((root / 'RUN_CONFIG.json').read_text(encoding='utf-8-sig'))
    ids = json.loads((root / 'RUN_IDS.json').read_text(encoding='utf-8-sig'))
    state = dict(status='running', stage='preflight', runs=ids['runs'],
                 started_at=time.time(), automatic_followup=False)

    def save():
        temp = root / 'PIPELINE_STATUS.json.tmp'
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(root / 'PIPELINE_STATUS.json')

    save()
    python = str(Path(sys.executable).with_name('python.exe'))
    os.environ['PYTHONPATH'] = r'D:\coco_wire\py'
    os.environ['PYTHONUTF8'] = '1'
    snapshots = [root / name for name in ('RUN_CONFIG.json', 'PROTOCOL.md', 'RUN_IDS.json')]
    snapshots += sorted((root / 'scripts').glob('*.py'))

    def run(stage, script, extra):
        rid = ids['runs'][stage]
        out = root / 'runs' / rid
        if (out / 'run.json').exists():
            raise RuntimeError(f'{stage}: Run already exists; no automatic retry or overwrite')
        state.update(stage=stage, current_run=rid)
        save()
        command = [python, r'D:\coco_wire\runner.py', '--study', ids['study_id'],
                   '--run-id', rid, '--cwd', str(root), '--output', str(out),
                   '--input', str(root / 'RUN_CONFIG.json'), '--input', str(root / 'SPLIT.json'),
                   '--input', str(Path(cfg['cache']) / 'INDEX.json'), '--input', cfg['weights'],
                   '--input', str(Path(cfg['operator_cache']) / 'COMPLETE.json')]
        for source in snapshots:
            command += ['--snapshot', str(source)]
        command += ['--expect', str(out / 'COMPLETE.json'), '--', python, '-u',
                    str(root / 'scripts' / script), '--config', str(root / 'RUN_CONFIG.json'),
                    '--out', str(out), *extra]
        subprocess.run(command, check=True, cwd=root, env=os.environ.copy(),
                       creationflags=subprocess.CREATE_NO_WINDOW)

    try:
        for source in (cfg['cache'], cfg['operator_cache']):
            if not (Path(source) / 'COMPLETE.json').is_file():
                raise FileNotFoundError(f'Reused cache incomplete: {source}')
        run('model_smoke', 'train_joint.py', ['--smoke'])
        smoke = json.loads((root / 'runs' / ids['runs']['model_smoke'] / 'COMPLETE.json').read_text(encoding='utf-8-sig'))
        if not (smoke.get('passed') or smoke.get('completed')):
            raise RuntimeError('Joint gradient/BN smoke did not pass')
        run('decode_smoke', 'evaluate_joint.py', ['--smoke'])
        deadline = time.time() + 3600 * cfg['max_training_hours']
        state['shared_training_deadline'] = deadline
        save()
        checkpoints = {}
        for mode in cfg['training_order']:
            run(mode, 'train_joint.py', ['--mode', mode, '--deadline', str(deadline)])
            checkpoints[mode] = str(root / 'runs' / ids['runs'][mode] / 'final.pt')
        path = root / 'FINAL_CHECKPOINTS.json'
        path.write_text(json.dumps(checkpoints, indent=2), encoding='utf-8')
        run('evaluation', 'evaluate_joint.py', ['--checkpoints', str(path)])
        state.update(status='completed', stage='finished', finished_at=time.time())
        save()
    except Exception as exc:
        state.update(status='failed', error=str(exc), finished_at=time.time(), automatic_retry=False)
        save()
        raise


if __name__ == '__main__':
    main()
