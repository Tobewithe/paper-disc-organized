import argparse
from pathlib import Path
import tarfile

from common import dump, read, sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--weights', action='store_true')
    args = p.parse_args(); root = Path(args.root)
    if args.weights:
        files = []
        for stage in ['baseline', 'continued', 'adapter', 'spm', 'export_lite']:
            ptr = root/(stage.upper()+'_POINTER.json')
            if ptr.exists():
                files.append(root/'runs'/read(ptr)['run_id']/'best.pt')
        files.append(root/'runs'/read(root/'PROTOTYPE_POINTER.json')['run_id']/'memory.pt')
        bundle = root/'SPM_V4_WEIGHTS.tar.gz'
    else:
        files = []
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if rel.parts[0] not in ['runs', 'scripts'] and len(rel.parts) != 1:
                continue
            if path.suffix.lower() not in ['.json', '.md', '.py', '.log', '.svg', '.png', '.jpg', '.sbatch']:
                continue
            if 'scenes' in rel.parts and path.name != 'scene.json':
                continue
            files.append(path)
        bundle = root/'SPM_V4_RESULTS_LIGHT.tar.gz'
    files = sorted(set(files))
    with tarfile.open(bundle, 'w:gz', compresslevel=1) as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    manifest = {'bundle': bundle.name, 'sha256': sha(bundle), 'size_bytes': bundle.stat().st_size,
                'files': {f.relative_to(root).as_posix(): {'sha256': sha(f), 'size_bytes': f.stat().st_size} for f in files}}
    dump(bundle.with_suffix('.receipt.json'), manifest)
    print(bundle, manifest['size_bytes'], manifest['sha256'], len(files), flush=True)


if __name__ == '__main__':
    main()
