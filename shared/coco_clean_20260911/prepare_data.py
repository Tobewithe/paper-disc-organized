"""Extract official COCO and preserve original annotations; never polygon-split."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path('/root/autodl-tmp/coco_clean_20260911')
PUBLIC = Path('/autodl-pub/data/COCO2017')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    selected = sorted({int(Path(s.strip()).stem) for s in (ROOT/'train_dense_source.txt').read_text().splitlines() if s.strip()})
    assert len(selected) == 37433, len(selected)
    selected_set = set(selected)
    data_dir = ROOT/'data'
    (data_dir/'annotations').mkdir(parents=True, exist_ok=True)
    audit = {'source': str(PUBLIC), 'train_selection': 'Historical fixed GT-only image IDs; no model results used', 'splits': {}}
    with zipfile.ZipFile(PUBLIC/'annotations_trainval2017.zip') as z:
        for split in ['train2017', 'val2017']:
            member = f'annotations/instances_{split}.json'
            dest = data_dir/member
            if not dest.exists():
                with z.open(member) as src, dest.open('wb') as out: shutil.copyfileobj(src, out)
            obj = json.loads(dest.read_text())
            ids = selected_set if split == 'train2017' else {im['id'] for im in obj['images']}
            images = sorted([im for im in obj['images'] if im['id'] in ids], key=lambda im: im['id'])
            assert len(images) == len(ids)
            anns = [a for a in obj['annotations'] if a['image_id'] in ids]
            regular = [a for a in anns if not a.get('iscrowd', 0)]
            assert len({a['id'] for a in anns}) == len(anns)
            multi = [a for a in regular if isinstance(a['segmentation'], list) and len(a['segmentation']) > 1]
            # Original val JSON remains byte-for-byte unchanged, including crowd annotations.
            subset = dict(images=images, annotations=anns, categories=obj['categories'], info=obj.get('info', {}), licenses=obj.get('licenses', []))
            subset_path = data_dir/'annotations'/f'instances_{split}_selected.json'
            subset_path.write_text(json.dumps(subset, separators=(',', ':')))
            with zipfile.ZipFile(PUBLIC/f'{split}.zip') as iz:
                wanted = {f'{split}/{im["file_name"]}' for im in images}
                # Read in archive order, avoiding tens of thousands of random seeks
                # against the shared public-data mount. Existing completed files resume.
                members = [entry for entry in iz.infolist() if entry.filename in wanted]
                assert len(members) == len(images)
                for k, entry in enumerate(members):
                    name = entry.filename
                    target = data_dir/'images'/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists() or target.stat().st_size != entry.file_size:
                        with iz.open(name) as src, target.open('wb') as out: shutil.copyfileobj(src, out)
                    assert target.stat().st_size == entry.file_size
                    if (k+1)%5000 == 0: print(split, 'extracted', k+1, flush=True)
            paths = [str(data_dir/'images'/split/im['file_name']) for im in images]
            (data_dir/f'{split}.txt').write_text('\n'.join(paths)+'\n')
            audit['splits'][split] = dict(images=len(images), noncrowd_instances=len(regular), crowd=len(anns)-len(regular), multipolygon_instances=len(multi),
                original_json_sha256=sha(dest), selected_json_sha256=sha(subset_path), image_ids_sha256=hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest())
            print(split, audit['splits'][split], flush=True)
            if split=='train2017': assert len(regular)==488877 and len(multi)==50616
            else: assert len(images)==5000 and len(regular)==36335
    (ROOT/'audits/data_source_audit.json').write_text(json.dumps(audit, indent=2))
    print('DATA_EXTRACTION_PASS', flush=True)

if __name__ == '__main__': main()
