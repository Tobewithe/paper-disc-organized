"""Freeze the available official-converted training cohort and full COCO evaluation list."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / 'data'
    out.mkdir(parents=True, exist_ok=True)
    old = Path('D:/coco_wire/data/official_tal_affine_20260930/runs/official_cache/official_data')
    rows = {}
    sources = {}
    for split in ('fit', 'dev', 'val'):
        p = old / (split + '.txt')
        ids = [Path(s.strip()).name for s in p.read_text().splitlines() if s.strip()]
        image_split = 'val2017' if split == 'val' else 'train2017'
        paths = [(old / 'images' / image_split / name).as_posix() for name in ids]
        missing = [s for s in paths if not Path(s).is_file()]
        missing_labels = [s for s in paths if not Path(s.replace('/images/', '/labels/')).with_suffix('.txt').is_file()]
        assert not missing and not missing_labels, (missing, missing_labels)
        assert len(paths) == len(set(paths)), 'Duplicate images'
        rows[split] = paths
        sources[split] = {'source_list': str(p), 'source_sha256': sha(p), 'count': len(paths)}
        (out / (split + '.txt')).write_text('\n'.join(paths) + '\n', encoding='utf-8')
    assert not set(rows['fit']) & set(rows['dev'])
    assert not set(rows['fit']) & set(rows['val'])
    for name, paths in [('smoke_train', rows['fit'][:32]), ('smoke_val', rows['val'][:8])]:
        (out / (name + '.txt')).write_text('\n'.join(paths) + '\n', encoding='utf-8')
    full = sorted(Path('D:/coco_wire/data/images/val2017').glob('*.jpg'))
    ann = Path('D:/coco_wire/data/annotations/instances_val2017.json')
    gt = json.loads(ann.read_text())
    assert {int(p.stem) for p in full} == {x['id'] for x in gt['images']}
    assert len(full) == 5000
    (out / 'val_full.txt').write_text('\n'.join(p.as_posix() for p in full) + '\n', encoding='utf-8')
    names = yaml.safe_load(Path('D:/coco_wire/vendor_8.4.100/ultralytics/cfg/datasets/coco.yaml').read_text(encoding='utf-8'))['names']
    for name, train, val in [('smoke.yaml', 'smoke_train.txt', 'smoke_val.txt'), ('fit.yaml', 'fit.txt', 'val.txt')]:
        (out / name).write_text(yaml.safe_dump({'path': root.as_posix() + '/data', 'train': train, 'val': val, 'names': names}, sort_keys=False), encoding='utf-8')
    receipt = {'training_source': 'Existing official convert_coco labels; no new label conversion or GT-based selection',
               'sources': sources, 'smoke_train_images': 32, 'smoke_val_images': 8,
               'final_evaluation_images': len(full), 'annotations': str(ann), 'annotation_sha256': sha(ann),
               'lists': {p.name: sha(p) for p in out.glob('*.txt')},
               'scope': '796-image feasibility training cohort reused from historical research; full val2017 is not a new blind test'}
    (out / 'DATA_RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
