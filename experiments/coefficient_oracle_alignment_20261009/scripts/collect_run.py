"""Prepare a hash-checked collection bundle for a Run, keeping large per-image caches remote.

Mirrors experiments/acd_native_coefficient_20261006/scripts/collect_run.py:
manifest.sha256 + transfer.json + collection.zip inside the Run directory. Per-image bank
caches (images/) are explicitly retained remote; per-arm tables, oracle/readout JSON and
protcol/audit files are collected.
"""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    run = a.root.resolve() / 'runs' / a.run
    assert run.is_dir(), run
    files, kept_remote = [], []
    for p in sorted(run.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(run)
        if p.name in ('collection.zip', 'manifest.sha256', 'transfer.json'):
            continue
        if 'images' in rel.parts or '__pycache__' in rel.parts:
            kept_remote.append({'path': rel.as_posix(), 'bytes': p.stat().st_size,
                                'reason': 'Per-image bank cache retained on the laptop; identity and readouts are collected'})
            continue
        files.append((p, rel.as_posix(), sha(p)))
    (run / 'manifest.sha256').write_text(''.join(f'{d}  {r}\n' for _, r, d in files), encoding='utf-8')
    transfer = {'run_id': a.run, 'source_host': '28358lan', 'source_directory': str(run),
                'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared',
                'file_count': len(files), 'collected_bytes': sum(p.stat().st_size for p, _, _ in files),
                'files_retained_remote': kept_remote,
                'scope': 'Selected generated artifacts are hash checked; excluded caches stay remote. Failed or partial Runs remain failed or partial.'}
    (run / 'transfer.json').write_text(json.dumps(transfer, indent=2), encoding='utf-8')
    bundle = run / 'collection.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p, rel, _ in files:
            z.write(p, rel)
        z.write(run / 'manifest.sha256', 'manifest.sha256')
        z.write(run / 'transfer.json', 'transfer.json')
    print(json.dumps({'run': a.run, 'files': len(files), 'bundle_bytes': bundle.stat().st_size,
                      'bundle_sha256': sha(bundle), 'retained_remote_files': len(kept_remote)}), flush=True)


if __name__ == '__main__':
    main()
