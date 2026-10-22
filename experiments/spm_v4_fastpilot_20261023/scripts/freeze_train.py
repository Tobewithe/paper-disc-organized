import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
files = ['PROTOCOL.md', 'PILOT_CONFIG.json']+['scripts/'+name for name in [
    'common.py', 'data.py', 'model.py', 'train.py', 'prototype.py', 'engineering.py', 'evaluate.py', 'run_pipeline.py', 'train.sbatch']]
result = {'files': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
          'git_parent': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'scope': 'data_training_and_evaluation_before_first_training; visual_artifact_code_recorded_by_its_run'}
(root/'TRAIN_FREEZE.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print('Frozen training inputs', len(files))
