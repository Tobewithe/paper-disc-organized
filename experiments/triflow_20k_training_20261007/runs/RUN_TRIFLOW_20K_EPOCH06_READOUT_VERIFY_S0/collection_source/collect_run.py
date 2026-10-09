"""Bundle selected stage outputs and inventories without copying large caches."""
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    run = (root/'runs'/a.run).resolve()
    assert run.is_relative_to(root/'runs') and run.is_dir()
    meta = json.loads((run/'run.json').read_text(encoding='utf-8-sig'))
    if meta['execution_status'] not in ('completed', 'failed'):
        raise RuntimeError('Do not bundle a running writer')
    collection_source = run / 'collection_source'
    collection_source.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), collection_source / 'collect_run.py')
    files, remote = [], []
    for path in sorted(run.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(run)
        if path.name in ('collection.zip', 'manifest.sha256', 'transfer.json'):
            continue
        required_baseline_replay_image = len(rel.parts) >= 3 and rel.parts[:2] == ('baseline', 'images')
        if 'cache' in rel.parts or ('images' in rel.parts and not required_baseline_replay_image):
            remote.append({'path': rel.as_posix(), 'bytes': path.stat().st_size,
                           'reason': 'Large frozen cache or redundant per-image predictions retained on laptop'})
        else:
            files.append((path, rel.as_posix(), sha(path)))
    (run/'manifest.sha256').write_text(''.join(f'{h}  {r}\n' for _,r,h in files), encoding='utf-8')
    transfer = {'run_id': a.run, 'status': 'prepared', 'source_host': '28358lan',
                'source_directory': str(run), 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
                'file_count': len(files), 'collected_bytes': sum(p.stat().st_size for p,_,_ in files),
                'collector_sha256': sha(collection_source / 'collect_run.py'),
                'returned_baseline_image_files': sum(int(r.startswith('baseline/images/')) for _,r,_ in files),
                'files_retained_remote': remote, 'scope': 'Selected outputs plus immutable source snapshots; failed Runs stay failed'}
    (run/'transfer.json').write_text(json.dumps(transfer,indent=2),encoding='utf-8')
    bundle = run/'collection.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for path,rel,_ in files:
            archive.write(path,rel)
        archive.write(run/'manifest.sha256','manifest.sha256')
        archive.write(run/'transfer.json','transfer.json')
    print(json.dumps({'run':a.run,'files':len(files),'bundle_bytes':bundle.stat().st_size,
                      'bundle_sha256':sha(bundle),'retained_remote_files':len(remote)}),flush=True)


if __name__ == '__main__':
    main()
