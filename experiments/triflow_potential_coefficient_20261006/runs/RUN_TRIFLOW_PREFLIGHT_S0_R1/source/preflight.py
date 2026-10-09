"""Verify actual laptop, original GT availability and native frozen inputs."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    sys.path.insert(0, str(Path('D:/coco_wire/vendor_8.4.100')))
    import torch
    import ultralytics
    import cv2
    import scipy
    from frozen_io import FrozenYOLO, dump_json
    assert ultralytics.__version__ == '8.4.100'
    choices = [Path('D:/coco_wire/data/annotations/instances_train2017.json'),
               Path('D:/coco_wire/bgcr_native_20261002/data/annotations/instances_train2017.json')]
    expected_train_sha = '610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d'
    observed_train_candidates = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
                                 for p in choices if p.is_file()]
    train = next((p for p in choices if p.is_file() and sha(p) == expected_train_sha), None)
    if train is None:
        raise FileNotFoundError('Neither known original train annotation path exists')
    assert sha(train) == expected_train_sha
    val = Path('D:/coco_wire/data/annotations/instances_val2017.json')
    assert sha(val) == 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'
    lists = {}
    for name, count in [('fit', 796), ('dev', 197), ('val', 196), ('val_full', 5000), ('smoke_train', 32)]:
        p = root / 'data' / (name + '.txt')
        paths = [Path(line.strip()) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        assert len(paths) == count and len({p.name for p in paths}) == count
        assert all(p.is_file() for p in paths), name
        lists[name] = {'path': str(p), 'sha256': sha(p), 'count': count}
    import pycocotools.coco
    coco = pycocotools.coco.COCO(str(train))
    fit = [Path(p) for p in (root / 'data' / 'fit.txt').read_text(encoding='utf-8-sig').splitlines() if p.strip()]
    assert all(int(p.stem) in coco.imgs for p in fit)
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                          '--format=csv,noheader'], capture_output=True, text=True)
    observed = gpu.stdout.strip().splitlines()
    if any('python' in line.lower() for line in observed):
        raise RuntimeError('Another Python compute process observed; preserve it and inspect before shared GPU use')
    frozen = FrozenYOLO('D:/coco_wire/models/yolo26m-seg.pt', 'D:/coco_wire/vendor_8.4.100', device='cuda')
    sample = frozen.extract(fit[0])
    frozen_check = frozen.verify_frozen()
    free = shutil.disk_usage(root).free
    assert free >= 20 * 1024**3, 'Less than 20 GiB free for frozen caches and prediction outputs'
    result = {'status': 'PASS', 'interpreter': sys.executable, 'torch': torch.__version__,
              'ultralytics': ultralytics.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__,
              'gpu': torch.cuda.get_device_name(0), 'gpu_processes_before_probe': observed,
              'free_disk_bytes': free, 'train_annotations': str(train), 'train_annotations_sha256': sha(train),
              'annotation_candidates': observed_train_candidates,
              'val_annotations': str(val), 'val_annotations_sha256': sha(val),
              'lists': lists, 'features_shape': list(sample['F'].shape),
              'prototype_shape': list(sample['P'].shape), 'hidden_query_shape': list(sample['h'].shape),
              'coefficient_shape': list(sample['c0'].shape), 'native_replay_exact': sample['native_replay_exact'],
              'frozen_state': frozen_check, 'image_id_probe': int(fit[0].stem),
              'scope': 'Input/environment/frozen native-forward contract; no method training or AP result'}
    dump_json(root / 'runs' / a.run_id / 'PREFLIGHT.json', result)
    dump_json(root / 'data' / 'DATA_RECEIPT.json', {'source': 'prior fixed image lists, original COCO raster GT',
        'train_annotations': str(train), 'train_annotations_sha256': sha(train),
        'val_annotations': str(val), 'val_annotations_sha256': sha(val), 'lists': lists,
        'historical_cohort': True, 'new_blind_test': False})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
