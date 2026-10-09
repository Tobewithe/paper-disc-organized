"""Create a separate shared complete split; never modify historical subsets."""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TRAIN_SHA = '610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d'
VAL_SHA = 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for k in ('root', 'archive', 'download-receipt', 'annotations', 'val-annotations', 'destination', 'val-images'):
        p.add_argument('--' + k, required=True, type=Path)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    run = a.root / 'runs' / a.run_id
    dst = a.destination.resolve()
    allowed = Path('D:/coco_wire/shared/coco2017/images').resolve()
    if dst != allowed / 'train2017':
        raise ValueError('Full split must be materialized only in the new shared dataset directory')
    receipt = json.loads(a.download_receipt.read_text(encoding='utf-8-sig'))
    if receipt.get('archive_crc_all_entries_passed') is not True or receipt.get('image_count') != 118287:
        raise ValueError('Actual complete download receipt required')
    archive_sha = sha(a.archive)
    if archive_sha != receipt['archive_sha256']:
        raise ValueError('Transferred ZIP differs from the verified local acquisition')
    if sha(a.annotations) != TRAIN_SHA or sha(a.val_annotations) != VAL_SHA:
        raise ValueError('Original COCO annotation identity differs')
    data = json.loads(a.annotations.read_text(encoding='utf-8'))
    images = data['images']
    original_annotation_count = len(data['annotations'])
    ordinary = sum(not r.get('iscrowd', 0) for r in data['annotations'])
    del data
    images.sort(key=lambda r: int(r['id']))
    by_name = {r['file_name']: r for r in images}
    if len(by_name) != len(images) or len(images) != 118287 or len({r['id'] for r in images}) != len(images):
        raise ValueError('Original split image identities differ')
    val = json.loads(a.val_annotations.read_text(encoding='utf-8'))
    val_images = sorted(val['images'], key=lambda r: int(r['id']))
    del val
    overlap = sorted({r['id'] for r in images} & {r['id'] for r in val_images})
    if overlap or len(val_images) != 5000:
        raise ValueError('Train/validation split identity overlaps or differs')
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dst).free < 22*1024**3:
        raise RuntimeError('Insufficient disk for complete image extraction')
    created = reused = 0
    manifest = run / 'TRAIN_IMAGES.sha256'
    with zipfile.ZipFile(a.archive) as z, manifest.open('w', encoding='utf-8') as out_manifest:
        entries = [r for r in z.infolist() if not r.is_dir()]
        filenames = [Path(e.filename).name for e in entries]
        if set(filenames) != set(by_name) or len(entries) != len(by_name):
            raise ValueError('ZIP members differ from original COCO identities')
        for entry in sorted(entries, key=lambda r: r.filename):
            name = Path(entry.filename).name
            if entry.filename != 'train2017/' + name or not name.endswith('.jpg'):
                raise ValueError('Unexpected archive member')
            target = (dst / name).resolve()
            if not target.is_relative_to(dst):
                raise ValueError('Archive member escaped dataset root')
            # Reading the complete member validates its stored ZIP CRC.
            expected = z.read(entry)
            digest = hashlib.sha256(expected).hexdigest()
            if target.exists():
                if target.stat().st_size != len(expected) or sha(target) != digest:
                    raise RuntimeError('Existing shared image differs; preserve it: ' + str(target))
                reused += 1
            else:
                temp = target.with_suffix('.jpg.materializing')
                temp.write_bytes(expected)
                temp.replace(target)
                created += 1
            out_manifest.write(digest + '  ' + name + '\n')
    paths = [dst / image['file_name'] for image in images]
    vpaths = [a.val_images.resolve() / image['file_name'] for image in val_images]
    if not all(p.is_file() for p in paths + vpaths):
        raise FileNotFoundError('Full train/validation image unavailable')
    (a.root / 'data').mkdir(exist_ok=True)
    train_list = a.root / 'data' / 'train_full.txt'
    val_list = a.root / 'data' / 'val_full.txt'
    train_list.write_text(''.join(p.as_posix() + '\n' for p in paths), encoding='utf-8')
    val_list.write_text(''.join(p.as_posix() + '\n' for p in vpaths), encoding='utf-8')
    final = {'status': 'PASS', 'created_images': created, 'reused_images': reused,
        'original_train_images': len(images), 'original_annotations': original_annotation_count,
        'ordinary_annotations': ordinary, 'val_images': len(val_images), 'train_val_overlap': overlap,
        'all_zip_members_crc_passed': True, 'train_image_content_manifest_sha256': sha(manifest),
        'archive': str(a.archive), 'archive_sha256': archive_sha,
        'download_receipt_sha256': sha(a.download_receipt), 'destination': str(dst),
        'train_annotations': str(a.annotations), 'train_annotation_sha256': TRAIN_SHA,
        'val_annotations': str(a.val_annotations), 'val_annotation_sha256': VAL_SHA,
        'train_list': str(train_list), 'train_list_sha256': sha(train_list),
        'val_list': str(val_list), 'val_list_sha256': sha(val_list),
        'interpreter': sys.executable, 'free_disk_bytes': shutil.disk_usage(dst).free,
        'completed_at': datetime.now(timezone.utc).isoformat()}
    dump(run / 'MATERIALIZATION_RECEIPT.json', final)
    dump(a.root / 'data' / 'FULL_DATA_READY.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
