"""Launch one registered laptop CPU storage continuation, never a model run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--run-id', required=True)
p.add_argument('--previous-run', required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
expected = Path('D:/coco_wire/prototype_guided_evidence_selection_20261003')
if os.name != 'nt' or root.resolve() != expected.resolve():
    raise RuntimeError('Run only on the authorized laptop')
ids = json.loads((root/'RUN_IDS.json').read_text(encoding='utf-8-sig'))
previous = root/'runs'/args.previous_run
out = root/'runs'/args.run_id
if (out/'run.json').exists():
    raise RuntimeError('Existing Run; do not duplicate the continuation')
state = json.loads((previous/'run.json').read_text(encoding='utf-8-sig'))
failure = json.loads((previous/'FAILURE.json').read_text(encoding='utf-8-sig'))
if state['status'] != 'failed' or '47 GiB' not in failure.get('error', ''):
    raise RuntimeError('Original packing must first end with the known storage-cap failure')
python = str(Path(sys.executable).with_name('python.exe'))
env = os.environ.copy()
env.update(PYTHONPATH='D:/coco_wire/py', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='')
command = [python, 'D:/coco_wire/runner.py', '--study', ids['study_id'],
           '--run-id', args.run_id, '--cwd', str(root), '--output', str(out)]
for source in [root/'RUN_CONFIG.json', root/'SPLIT.json', previous/'FAILURE.json',
               previous/'run.json', root/'assets/MIGRATION_IDENTITY.json']:
    command += ['--input', str(source)]
for source in [Path(__file__), root/'scripts/prepare_transfer_assets.py',
               root/'scripts/prepare_transfer_assets_storage_resume.py']:
    command += ['--snapshot', str(source)]
command += ['--expect', str(out/'COMPLETE.json'), '--', python, '-X', 'utf8', '-u',
            str(root/'scripts/prepare_transfer_assets_storage_resume.py'),
            '--config', str(root/'RUN_CONFIG.json'), '--assets', str(root/'assets'),
            '--previous-run', str(previous), '--run-id', args.run_id, '--out', str(out)]
with (root/('storage_resume_'+args.run_id+'.log')).open('ab', buffering=0) as log:
    result = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
raise SystemExit(result.returncode)
