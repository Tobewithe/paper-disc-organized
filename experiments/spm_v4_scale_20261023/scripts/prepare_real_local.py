"""Prepare all original train RGB on Windows without installing new runtime packages."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import threading
import zipfile

from PIL import Image

root = Path(__file__).resolve().parents[1]
source = Path('C:/Dpan/document/model_datasets/datasets/piglife')
local = threading.local()

def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()

def main():
    ann_path = source/'derived/task05_v1/pig_coco_train_task05_v1.json'
    coco = json.loads(ann_path.read_text(encoding='utf-8'))
    old = json.loads((root.parent/'spm_v4_fastpilot_20261023/DATASET.json').read_text(encoding='utf-8'))
    groups = {r['group']: r['split'] for r in old['rows'] if r['domain'] == 'real'}
    folder = root/'inputs/real_images'; folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source/'Image/train.zip') as archive:
        names = {Path(name).name: name for name in archive.namelist() if name.lower().endswith('.jpg')}
    def work(row):
        if not hasattr(local, 'archive'):
            local.archive = zipfile.ZipFile(source/'Image/train.zip')
        name = Path(row['file_name']).name
        raw = local.archive.read(names[name]); im = Image.open(io.BytesIO(raw)).convert('RGB')
        assert im.size == (row['width'], row['height'])
        ratio = 640/max(im.size); ww, hh = [round(d*ratio) for d in im.size]
        left, top = (640-ww)//2, (640-hh)//2
        canvas = Image.new('RGB', (640, 640), (114, 114, 114)); canvas.paste(im.resize((ww, hh), Image.Resampling.BILINEAR), (left, top))
        path = folder/f"{row['id']}.jpg"; canvas.save(path, quality=95, subsampling=0)
        group = name.split('s5')[0]; assert group in groups
        return {**row, 'source_id': f"real/{row['id']}", 'domain': 'real', 'split': groups[group], 'group': group,
                'source_file_name': row['file_name'], 'source_sha256': sha_bytes(raw),
                'image': f"inputs/real_images/{row['id']}.jpg", 'image_sha256': sha_bytes(path.read_bytes()),
                'valid_rect': [left, top, ww, hh]}
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(work, sorted(coco['images'], key=lambda r: r['id'])):
            rows.append(row)
            if len(rows) % 100 == 0:
                print('REAL_PREPARED', len(rows), 1704, flush=True)
    counts = dict(Counter(r['split'] for r in rows)); assert counts == {'train': 1272, 'dev': 432} or counts == {'dev': 432, 'train': 1272}
    manifest = {'rows': rows, 'counts': counts, 'source_annotation_sha256': sha_bytes(ann_path.read_bytes()),
                'source_archive': str(source/'Image/train.zip'), 'archive_members_verified_individually': True,
                'test_images_included': False, 'RGB_preprocessing': 'PIL_bilinear_common640_JPEG95_subsampling0',
                'pillow': Image.__version__, 'split_policy': 'preserved_old_13_train_4_dev_filename_families'}
    (root/'inputs/REAL_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    shutil.copy2(ann_path, root/'inputs/REAL_SOURCE_COCO.json')
    bundle = root/'REAL_INPUTS.tar'
    with tarfile.open(bundle, 'w') as archive:
        for path in [root/'inputs/REAL_MANIFEST.json', root/'inputs/REAL_SOURCE_COCO.json', *sorted(folder.glob('*.jpg'))]:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    receipt = {'bundle_sha256': sha_bytes(bundle.read_bytes()), 'size_bytes': bundle.stat().st_size, 'images': len(rows), 'counts': counts}
    (root/'REAL_INPUT_RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('REAL_READY', receipt, flush=True)


if __name__ == '__main__':
    main()
