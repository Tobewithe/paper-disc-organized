import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rt import dump, sha, utc
p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
a = p.parse_args(); root = a.root.resolve()
assert not (root/'POSTPROCESS_SOURCE.json').exists()
files = ['postprocess.sbatch']+[p.relative_to(root).as_posix() for p in sorted((root/'presentation_tools').glob('*.py'))]
dump(root/'POSTPROCESS_SOURCE.json', dict(frozen_at=utc(), scope='independent CPU acceptance and figures only; no new training',
                                       files={name: sha(root/name) for name in files}))
