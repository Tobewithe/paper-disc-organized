"""Freeze this study only; never mutate a prior study or its protocol."""
import argparse
import json
from pathlib import Path
import subprocess

from rt import dump, read, sha, utc


p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--supersedes')
a = p.parse_args(); root = a.root.resolve()
if (root/'FREEZE.json').exists():
    assert a.supersedes and sha(root/'FREEZE.json') == sha(root/a.supersedes), 'Archive identical prior freeze first'
files = ['CONFIG.json', 'PROTOCOL.md']+[x.relative_to(root).as_posix() for x in sorted((root/'scripts').glob('*.py'))]
inventory = read(root/'INVENTORY.json')
authorization = dict(timestamp=utc(), direct_user_message='试试',
    preceding_proposal='真实图像中的对象对齐及小预算分割效用验证；输出稳定性作伴随检查，不为潜码另开优化轮次',
    scope='new v1 real-data pilot on registered a25023104 server; overrides no-remote-training for this request only',
    protocol_sha256=sha(root/'PROTOCOL.md'), config_sha256=sha(root/'CONFIG.json'),
    max_GPUs=1, max_total_GPU_minutes=120, formal_steps_per_arm=4000, formal_arms=2,
    engineering_steps_per_arm=8, test_authorized=False, further_training_authorized=False)
dump(root/'EXECUTION_AUTHORIZATION.json', authorization)
files += ['EXECUTION_AUTHORIZATION.json', 'INVENTORY.json', 'engineering.sbatch', 'formal.sbatch']
if a.supersedes: files += [p.name for p in sorted(root.glob('ENGINEERING_REVISION_R*.json'))]
dump(root/'FREEZE.json', dict(frozen_at=utc(), files={name: sha(root/name) for name in files},
    external_files=inventory['external_files'], code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    source_status='new files frozen by exact per-file hashes; unrelated worktree edits excluded',
    prior_v0_unchanged=True, supersedes=a.supersedes))
print('FROZEN', len(files), 'local files,', len(inventory['external_files']), 'external inputs')
