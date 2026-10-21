"""Download a pinned official Blender distribution without touching installed tools."""
import argparse
import hashlib
import json
import pathlib
import time
import urllib.request

URL = 'https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz'
SHA = 'be0fbaa0c1e52d4552023220b4c67351efbb707cac49bb381fcbee2182447005'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / URL.rsplit('/', 1)[-1]
    if not dest.exists() or digest(dest) != SHA:
        tmp = dest.with_suffix(dest.suffix + '.part')
        request = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        start = last = time.time()
        total = 0
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, 'wb') as f:
            while True:
                b = response.read(4 * 1024 * 1024)
                if not b:
                    break
                f.write(b)
                total += len(b)
                if time.time() - last >= 10:
                    print(json.dumps({'downloaded_mb': round(total / 1e6, 1), 'elapsed_s': round(time.time()-start)}), flush=True)
                    last = time.time()
        if digest(tmp) != SHA:
            raise RuntimeError('Official Blender SHA256 mismatch; partial file retained')
        tmp.replace(dest)
    manifest = {'source': URL, 'filename': dest.name, 'sha256': digest(dest), 'expected_sha256': SHA, 'size_bytes': dest.stat().st_size}
    (root / 'SOFTWARE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
