"""Verify and unpack only the registered 20k image bundle on the offline laptop."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from datetime import datetime, timezone
from frozen_io import dump_json, sha256
from stream_data import select_declared_train_ids, COCO_TRAIN_ANNOTATION_SHA256


def identity(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'bundle', 'receipt', 'manifest', 'selected-ids', 'annotations'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    run = root / 'runs' / a.run_id
    run.mkdir(parents=True, exist_ok=True)
    selected = json.loads(a.selected_ids.read_text(encoding='utf-8-sig'))
    if len(selected) != 20000 or selected != sorted(set(selected)):
        raise ValueError('Actual selected ID file is not exactly20000 sorted unique images')
    if sha256(a.annotations) != COCO_TRAIN_ANNOTATION_SHA256:
        raise ValueError('Original training JSON differs')
    data = json.loads(a.annotations.read_text(encoding='utf-8'))
    images = data['images']
    del data
    expected_ids = select_declared_train_ids([r['id'] for r in images])
    if selected != expected_ids:
        raise ValueError('Actual input bundle did not use the registered image-independent20k selector')
    records = {r['id']: r for r in images}
    names = {records[iid]['file_name']: iid for iid in selected}
    receipt = json.loads(a.receipt.read_text(encoding='utf-8-sig'))
    expected_bundle_sha = receipt.get('tar_sha256') or receipt.get('bundle_sha256')
    if expected_bundle_sha is None or sha256(a.bundle) != expected_bundle_sha:
        raise ValueError('Transferred selected-image bundle SHA differs from actual local preparation')
    expected = {}
    for line in a.manifest.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if name in expected or name not in names:
            raise ValueError('Selected-image content manifest identity differs')
        expected[name] = digest
    if set(expected) != set(names):
        raise ValueError('Actual content manifest does not cover selected20000 images')
    destination = run / 'images/train2017'
    destination.mkdir(parents=True, exist_ok=True)
    copied = set()
    with tarfile.open(a.bundle, 'r:*') as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or member.name != 'train2017/' + name or name not in expected or name in copied:
                raise ValueError('Unexpected, duplicate or nonregular selected tar member')
            content = tar.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != expected[name]:
                raise ValueError('Selected JPEG content changed during transfer: ' + name)
            target = destination / name
            if target.exists() and sha256(target) != expected[name]:
                raise ValueError('Existing owned image differs; preserve it: ' + str(target))
            temporary = target.with_suffix('.jpg.materializing')
            temporary.write_bytes(content)
            temporary.replace(target)
            copied.add(name)
    if copied != set(expected):
        raise ValueError('Transferred bundle is incomplete')
    paths = [destination.resolve() / records[iid]['file_name'] for iid in selected]
    data_dir = root / 'data'
    data_dir.mkdir(exist_ok=True)
    plist = data_dir / 'train20k.txt'
    plist.write_text(''.join(path.as_posix() + '\n' for path in paths), encoding='utf-8')
    for source, name in ((a.receipt, 'LOCAL20K_RECEIPT.json'), (a.manifest, 'SELECTED_IMAGES.sha256'),
                         (a.selected_ids, 'SELECTED_IMAGE_IDS.json')):
        (run / name).write_bytes(source.read_bytes())
    result = {'passed': True, 'status': 'PASS', 'selected_images': 20000,
        'complete_original_train_split': False, 'complete_declared_train_subset': True,
        'selected_image_identity_sha256': identity(selected), 'registered_selector_exact': True,
        'all_selected_jpeg_content_hashes_verified': True, 'bundle_sha256': expected_bundle_sha,
        'source_local_preparation_receipt_sha256': sha256(a.receipt),
        'selected_manifest_sha256': sha256(a.manifest), 'selected_ids_file_sha256': sha256(a.selected_ids),
        'annotations': str(a.annotations), 'annotation_sha256': COCO_TRAIN_ANNOTATION_SHA256,
        'destination': str(destination.resolve()), 'images_list': str(plist), 'images_list_sha256': sha256(plist),
        'materialization_run': a.run_id, 'completed_at': datetime.now(timezone.utc).isoformat()}
    dump_json(run / 'MATERIALIZATION_RECEIPT.json', result)
    dump_json(data_dir / 'DATA20K_READY.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
