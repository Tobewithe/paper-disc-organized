"""Audit the actual complete input split and idle laptop before formal training."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from frozen_io import FrozenYOLO, OFFICIAL_SHA256, dump_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    ready_path = root / 'data/FULL_DATA_READY.json'
    ready = json.loads(ready_path.read_text(encoding='utf-8'))
    if ready.get('status') != 'PASS' or ready.get('original_train_images') != 118287:
        raise ValueError('Actual complete materialization receipt required')
    materialization_run = root / 'runs/RUN_TRIFLOW_FULL_DATA_MATERIALIZE_S0'
    meta = json.loads((materialization_run / 'run.json').read_text(encoding='utf-8-sig'))
    if meta.get('execution_status') != 'completed':
        raise ValueError('Dataset materialization has not completed successfully')
    if (materialization_run / 'MATERIALIZATION_RECEIPT.json').read_bytes() != ready_path.read_bytes():
        raise ValueError('Ready marker differs from actual materialization receipt')
    if sha256(materialization_run / 'TRAIN_IMAGES.sha256') != ready['train_image_content_manifest_sha256']:
        raise ValueError('All-image source-byte manifest differs')
    dataset_root = Path(ready['destination']).resolve()
    verified_names = set()
    for line in (materialization_run / 'TRAIN_IMAGES.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        path = (dataset_root / name).resolve()
        if path.parent != dataset_root or name in verified_names or sha256(path) != digest:
            raise ValueError('Actual complete image bytes differ from official ZIP materialization: ' + name)
        verified_names.add(name)
    if len(verified_names) != 118287:
        raise ValueError('Actual full image manifest is incomplete')
    for prefix, n in (('train', 118287), ('val', 5000)):
        plist = Path(ready[prefix + '_list'])
        if sha256(plist) != ready[prefix + '_list_sha256']:
            raise ValueError('Complete input list changed: ' + prefix)
        paths = [Path(line) for line in plist.read_text(encoding='utf-8').splitlines() if line]
        if len(paths) != n or len({p.name for p in paths}) != n or not all(p.is_file() for p in paths):
            raise ValueError('Complete input files missing/duplicated: ' + prefix)
        if prefix == 'train' and {p.name for p in paths} != verified_names:
            raise ValueError('Actual content-verified images and full train list differ')
        if sha256(Path(ready[prefix + '_annotations'])) != ready[prefix + '_annotation_sha256']:
            raise ValueError('Original annotation changed: ' + prefix)
    import torch
    device = torch.cuda.get_device_name(0)
    probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                            '--format=csv,noheader'], capture_output=True, text=True, check=True)
    active = [row for row in probe.stdout.splitlines() if row.strip()]
    others = [row for row in active if 'python' in row.lower() and row.split(',')[0].strip() != str(os.getpid())]
    if others:
        raise RuntimeError('Another Python GPU process is active; preserve its work and defer this launch: ' + str(others))
    weights = Path('D:/coco_wire/models/yolo26m-seg.pt')
    if sha256(weights) != OFFICIAL_SHA256:
        raise ValueError('Official checkpoint identity differs')
    contract_path = root / 'runs/RUN_TRIFLOW_FULL_STREAM_VERIFY_S0/VERIFY_STREAM.json'
    stream_contract = json.loads(contract_path.read_text(encoding='utf-8'))
    if stream_contract.get('passed') is not True:
        raise ValueError('Actual streaming/pilot byte-equivalence proof required')
    resume_path = root / 'runs/RUN_TRIFLOW_FULL_RESUME_VERIFY_S0_R1_RECHECK/RESUME_VERIFICATION.json'
    resume_contract = json.loads(resume_path.read_text(encoding='utf-8'))
    if resume_contract.get('passed') is not True:
        raise ValueError('Actual uninterrupted/resumed optimizer-state equivalence proof required')
    for name, digest in resume_contract['training_sources'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('Resume evidence tested another training implementation: ' + name)
    for name, digest in stream_contract['sources'].items():
        current = Path(__file__).with_name(name)
        if current.exists() and name in ('stream_data.py', 'frozen_io.py', 'triflow_model.py') and sha256(current) != digest:
            raise ValueError('Streaming-equivalence evidence tested another implementation: ' + name)
    evaluator_contract_path = root / 'runs/RUN_TRIFLOW_FULL_EVALUATOR_CONTRACT_S0/CONTRACT_VERIFICATION.json'
    evaluator_contract = json.loads(evaluator_contract_path.read_text(encoding='utf-8'))
    if evaluator_contract.get('passed') is not True or evaluator_contract.get('evaluator_sha256') != sha256(Path(__file__).with_name('evaluate_fullscale.py')):
        raise ValueError('Actual evaluator engineering/pilot rejection contract is missing or stale')
    frozen = FrozenYOLO(weights, Path('D:/coco_wire/vendor_8.4.100'), device='cuda')
    train_paths = [Path(line) for line in (root / 'data/train_full.txt').read_text().splitlines() if line]
    samples = [frozen.extract(train_paths[i]) for i in (0, len(train_paths)-1)]
    integrity = frozen.verify_frozen()
    frozen.close()
    if integrity.get('passed') is not True or not all(s['native_replay_exact'] for s in samples):
        raise RuntimeError('Actual frozen/native replay failed')
    free = shutil.disk_usage(root).free
    if free < 15*1024**3:
        raise RuntimeError('Insufficient output disk for checkpoints and complete evaluation')
    result = {'passed': True, 'scope': 'Complete inputs and execution contracts, no method AP result',
              'data_ready_sha256': sha256(ready_path), 'input': ready, 'torch': torch.__version__,
              'device': device, 'gpu_processes_before_probe': active, 'free_disk_bytes': free,
              'stream_equivalence_sha256': sha256(contract_path),
              'resume_equivalence_sha256': sha256(resume_path), 'frozen_integrity': integrity,
              'actual_train_jpeg_files_rehashed': len(verified_names),
              'evaluator_contract_sha256': sha256(evaluator_contract_path),
              'native_probe_image_ids': [int(Path(s['image_path']).stem) for s in samples],
              'interpreter': sys.executable}
    dump_json(root / 'runs' / a.run_id / 'PREFLIGHT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
