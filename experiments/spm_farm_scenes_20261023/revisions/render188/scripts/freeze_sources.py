"""Freeze this study's current generation contract; never edits previous Runs."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
repo = root.parents[1]
files = [root/'PROTOCOL.md', root/'CONFIG.json', root/'SCENES.json', root/'REAL_TRAIN_PROFILE.json', root/'ASSET_INSPECTION.json']
files += sorted((root/'scripts').glob('*.py')) + sorted((root/'scripts').glob('*.sbatch'))
patch = subprocess.run(['git', 'diff', '--binary', '--', 'project/ACTIVE_PLAN.md', str(root.relative_to(repo))],
                       cwd=repo, capture_output=True, check=True).stdout
manifest = {'frozen_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'tracked_dirty_patch_sha256': hashlib.sha256(patch).hexdigest(),
            'untracked_sources_included_by_content_hash': True,
            'files': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(root/'FREEZE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('frozen_files', len(manifest['files']))
