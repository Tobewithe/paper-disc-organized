"""Remote CPU-only export: fixed COCO image IDs and installed Python sources."""
import hashlib
import importlib.metadata
import io
import json
import tarfile
import zipfile
from pathlib import Path

ROOT = Path('/root/autodl-tmp/coco_clean_20260911')
OUT = ROOT / 'diagnostics/local_readout_transfer_v2_20260912'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rank(iid):
    value = hashlib.sha256(f'readout-input:20260912:{iid}'.encode()).digest()
    return int.from_bytes(value[:8], 'little') % (2**63-1)


def main():
    OUT.mkdir(exist_ok=False)
    annotation = ROOT / 'data/annotations/instances_train2017.json'
    selection=json.loads((ROOT/'local_readout_selection_20260912.json').read_text())
    images = {r['id']: r for r in selection['images']}
    chosen = selection['fit']+selection['transfer']
    annotation_hash = sha(annotation)
    if annotation_hash!=selection['annotation_sha256']:
        raise RuntimeError('Remote annotations differ from local selection source')
    package = Path(importlib.metadata.distribution('ultralytics').locate_file('ultralytics'))
    files = {}
    datafiles = {}
    with tarfile.open(OUT/'runtime.tar.gz', 'w:gz') as tar:
        for path in sorted(package.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            name = 'vendor/ultralytics/' + path.relative_to(package).as_posix()
            tar.add(path, arcname=name, recursive=False)
            files[name] = sha(path)
    with zipfile.ZipFile('/autodl-pub/data/COCO2017/train2017.zip') as archive, tarfile.open(OUT/'images.tar', 'w') as tar:
        for iid in chosen:
            member = 'train2017/' + images[iid]['file_name']
            existing = ROOT / 'data/images' / member
            content = existing.read_bytes() if existing.exists() else archive.read(member)
            name = 'images/' + member
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            tar.addfile(entry, io.BytesIO(content))
            datafiles[name] = hashlib.sha256(content).hexdigest()
        # One existing val cache gives a cross-runtime comparison witness. It
        # never selects target images for the fitting or transfer evaluation.
        cachedir = ROOT/'diagnostics/full_val_cache_20260911/val'
        paths = sorted(cachedir.glob('*.npz'), key=lambda p:int(p.stem))
        reference = paths[0]
        iid = int(reference.stem)
        im = ROOT/'data/images/val2017'/f'{iid:012d}.jpg'
        for path, name in [(reference,'reference/remote_val.npz'),(im,'reference/remote_val.jpg')]:
            tar.add(path,arcname=name,recursive=False)
            datafiles[name]=sha(path)
    receipt = dict(version='1.0',cpu_only_export=True,selection=dict(fit=chosen[:32],transfer=chosen[32:],
                   all_train_images=selection['all_train_images'],filter='raw image ID hash, no ICI/prediction filter'),
                   annotation_sha256=annotation_hash,weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),
                   versions={n:importlib.metadata.version(n) for n in ['ultralytics','torch','torchvision','numpy','opencv-python','pycocotools','pillow']},
                   runtime_files=files,data_files=datafiles,
                   archive_sha256={name:sha(OUT/name) for name in ['runtime.tar.gz','images.tar']})
    (OUT/'RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(directory=str(OUT),annotation_sha256=annotation_hash,
          selected=len(chosen),runtime_files=len(files),archive_sha256=receipt['archive_sha256'])),flush=True)


if __name__=='__main__':
    main()
