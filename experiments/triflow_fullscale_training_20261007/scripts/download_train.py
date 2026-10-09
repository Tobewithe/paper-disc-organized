"""Acquire the official public COCO archive with bounded, resumable ranges."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from datetime import datetime, timezone

URL = 'http://images.cocodataset.org/zips/train2017.zip'
SOURCE = 'https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm'


def dump(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, indent=2), encoding='utf-8')
    t.replace(p)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--connections', type=int, default=8)
    a = ap.parse_args()
    run = a.root / 'runs' / a.run_id
    dst = a.destination.resolve()
    workspace = a.root.resolve().parents[1]
    if not dst.is_relative_to(workspace / 'shared' / 'coco2017'):
        raise ValueError('Destination must stay in this workspace shared/coco2017')
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=45) as r:
        size = int(r.headers['Content-Length'])
        etag = r.headers.get('ETag')
        modified = r.headers.get('Last-Modified')
    if size != 19336861798 or not etag:
        raise RuntimeError('Official object identity changed; preserve and inspect')
    staging = dst.with_name(dst.name + '.parts')
    staging.mkdir(exist_ok=True)
    identity = {'url': URL, 'bytes': size, 'etag': etag, 'connections': a.connections}
    identity_path = staging / 'IDENTITY.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError('Existing partial download belongs to another object/layout')
    dump(identity_path, identity)
    dump(run / 'DOWNLOAD_SOURCE.json', {**identity, 'official_listing': SOURCE,
         'last_modified': modified, 'destination': str(dst), 'transport': 'official listing HTTP URL',
         'published_sha256': None, 'started_at': datetime.now(timezone.utc).isoformat()})
    if shutil.disk_usage(dst.parent).free < size * 2 + 5*1024**3:
        raise RuntimeError('Insufficient free disk for resumable archive and assembly')
    width = (size + a.connections - 1) // a.connections
    ranges = [(i, i*width, min(size-1, (i+1)*width-1)) for i in range(a.connections)]

    def fetch(part):
        i, start, end = part
        path = staging / ('part_%02d.bin' % i)
        length = end-start+1
        for retry in range(9):
            have = path.stat().st_size if path.exists() else 0
            if have == length:
                return {'index': i, 'start': start, 'end': end, 'bytes': have, 'sha256': sha(path)}
            if have > length:
                raise RuntimeError('Partial range is longer than expected')
            request = urllib.request.Request(URL, headers={
                'Range': f'bytes={start+have}-{end}', 'If-Range': etag, 'Accept-Encoding': 'identity'})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected_range = f'bytes {start+have}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                        raise RuntimeError('Server did not honor exact byte range')
                    if response.headers.get('ETag') != etag:
                        raise RuntimeError('Object changed during download')
                    with path.open('ab') as out:
                        while True:
                            block = response.read(1024**2)
                            if not block:
                                break
                            out.write(block)
                if path.stat().st_size != length:
                    raise RuntimeError('Truncated byte range')
            except Exception:
                if retry == 8:
                    raise
                time.sleep(min(30, 2*(retry+1)))
        raise RuntimeError('Range retries exhausted')

    if not dst.exists():
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.connections) as pool:
            futures = [pool.submit(fetch, part) for part in ranges]
            while not all(f.done() for f in futures):
                done_bytes = sum(p.stat().st_size for p in staging.glob('part_*.bin'))
                dump(run / 'DOWNLOAD_PROGRESS.json', {'downloaded_bytes': done_bytes,
                     'total_bytes': size, 'updated_at': datetime.now(timezone.utc).isoformat()})
                time.sleep(10)
            parts = [f.result() for f in futures]
        assembled = dst.with_suffix('.zip.assembling')
        with assembled.open('wb') as out:
            for i, _, _ in ranges:
                with (staging / ('part_%02d.bin' % i)).open('rb') as source:
                    shutil.copyfileobj(source, out, 8*1024**2)
        if assembled.stat().st_size != size:
            raise RuntimeError('Assembled archive size differs')
        assembled.replace(dst)
        dump(run / 'DOWNLOAD_PARTS.json', parts)
    if dst.stat().st_size != size:
        raise RuntimeError('Archive size differs')
    digest = sha(dst)
    with zipfile.ZipFile(dst) as z:
        names = [p.filename for p in z.infolist() if not p.is_dir()]
        if len(names) != 118287 or len(set(names)) != len(names):
            raise RuntimeError('Archive train2017 image count differs')
        if any(not n.startswith('train2017/') or Path(n).suffix != '.jpg' or len(Path(n).stem) != 12 for n in names):
            raise RuntimeError('Unexpected archive entry')
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError('Archive CRC failed: ' + bad)
    receipt = {**identity, 'archive_sha256': digest, 'image_count': len(names),
               'archive_crc_all_entries_passed': True, 'destination': str(dst),
               'completed_at': datetime.now(timezone.utc).isoformat()}
    dump(run / 'DOWNLOAD_RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
