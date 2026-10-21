from pathlib import Path
import hashlib
import json
import zipfile
import datetime

root = Path(r'D:\coco_wire\baseline_native_20261013')
archive = Path(r'D:\coco_wire\RFC_DEPLOYMENT_20261017.zip')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive) == '595452b2a8f187322c219b2a9be05aa0fadc022619a065c5d440771099209d83'
with zipfile.ZipFile(archive) as z:
    manifest = json.loads(z.read('RFC_DEPLOYMENT_MANIFEST.json'))
    for rel, digest in manifest.items():
        p = (root / rel).resolve()
        assert p.is_relative_to(root.resolve()), rel
        assert hashlib.sha256(z.read(rel)).hexdigest() == digest, rel
        assert not p.exists() or sha(p) == digest, 'existing differs: ' + rel
    written = []
    for rel, digest in manifest.items():
        p = root / rel
        if not p.exists():
            p.parent.mkdir(exist_ok=True, parents=True)
            with p.open('xb') as f:
                f.write(z.read(rel))
            written.append(rel)
        assert sha(p) == digest, rel
receipt = dict(at=datetime.datetime.now().isoformat(), files=len(manifest),
               written=written, archive_sha256=sha(archive), status='passed')
study = root / 'experiments/mask_region_f_correspondence_20261017'
(study / 'DEPLOYMENT_RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(dict(status='passed', files=len(manifest), written=len(written))))
