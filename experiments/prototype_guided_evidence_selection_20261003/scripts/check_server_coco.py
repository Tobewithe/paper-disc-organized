"""Read-only COCO source inventory; does not run models or extract whole archives."""
from pathlib import Path
import json
import hashlib
import zipfile

root = Path(__file__).resolve().parents[1]
source = Path('/autodl-pub/data/COCO2017')
split = json.loads((root/'SPLIT.json').read_text())
checks = json.loads((root/'SOURCE_IMAGE_CHECKS.json').read_text())
audit = dict(source=str(source), images={}, planned={}, source_hash_checks=[],
             downloaded=False, full_archive_extracted=False, status='running')
def save():
    tmp = root/'SERVER_COCO_AUDIT.json.tmp'
    tmp.write_text(json.dumps(audit,indent=2)); tmp.replace(root/'SERVER_COCO_AUDIT.json')
save()
try:
    lookup = {}
    for domain in ['train','val']:
        with zipfile.ZipFile(source/(domain+'2017.zip')) as z:
            members = [n for n in z.namelist() if n.endswith('.jpg')]
            audit['images'][domain] = len(members)
            lookup.update({int(Path(n).stem):(domain,n) for n in members})
            for item in checks:
                name = f'{domain}2017/{item["image_id"]:012d}.jpg'
                if name in z.NameToInfo:
                    actual = hashlib.sha256(z.read(name)).hexdigest()
                    audit['source_hash_checks'].append(dict(image_id=item['image_id'],source_split=domain,
                                                           sha256=actual,identical=actual==item['sha256']))
        save()
    for group, ids in split.items():
        audit['planned'][group] = dict(images=len(ids),missing=[i for i in ids if i not in lookup])
    with zipfile.ZipFile(source/'annotations_trainval2017.zip') as z:
        audit['annotations'] = {}
        for name in ['annotations/instances_train2017.json','annotations/instances_val2017.json']:
            digest = hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):
                    digest.update(block)
            audit['annotations'][name] = dict(bytes=z.getinfo(name).file_size,sha256=digest.hexdigest())
            save()
    audit['passed'] = (len(audit['source_hash_checks']) == len(checks)
                       and all(v['identical'] for v in audit['source_hash_checks'])
                       and all(not v['missing'] for v in audit['planned'].values()))
    audit['status'] = 'complete'
    save()
except BaseException as exc:
    audit.update(status='failed',error=str(exc));save();raise
