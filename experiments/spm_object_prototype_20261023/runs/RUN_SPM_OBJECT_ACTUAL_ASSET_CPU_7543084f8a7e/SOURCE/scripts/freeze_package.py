"""Freeze local source bytes; does not authorize or submit any execution."""
import argparse
from pathlib import Path
import subprocess

from runtime import dump, sha


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    target=root/'FREEZE_V0.json'
    if target.exists(): raise SystemExit('Freeze already exists; preserve it and register an explicit engineering revision.')
    files=[root/'CONFIG_V0.json',root/'PROTOCOL_V0.md']+sorted((root/'scripts').glob('*.py'))+sorted((root/'scripts').glob('*.sbatch'))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip()
    dump(target,dict(code_commit=commit,files={f.relative_to(root).as_posix():sha(f) for f in files},
                     external_files='asset SHA in frozen config; exported geometry locked after independent CPU export',
                     scope='source freeze only; no formal GPU/training authorization implied'))
    print('FROZEN_FILES',len(files))
