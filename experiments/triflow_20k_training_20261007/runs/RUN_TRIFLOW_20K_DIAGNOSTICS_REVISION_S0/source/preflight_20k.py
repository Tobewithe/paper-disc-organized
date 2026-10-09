"""Check selected20k bytes, source contracts and actual native inputs before training."""
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
    ready_path = root / 'data/DATA20K_READY.json'
    ready = json.loads(ready_path.read_text(encoding='utf-8'))
    if ready.get('passed') is not True or ready.get('selected_images') != 20000 or ready.get('complete_original_train_split') is not False:
        raise ValueError('Actual completed declared20k preparation required')
    materialization = root / 'runs' / ready['materialization_run']
    metadata = json.loads((materialization / 'run.json').read_text(encoding='utf-8-sig'))
    if metadata.get('execution_status') != 'completed':
        raise ValueError('Selected data materialization has not successfully completed')
    if (materialization / 'MATERIALIZATION_RECEIPT.json').read_bytes() != ready_path.read_bytes():
        raise ValueError('Ready marker differs from actual Run receipt')
    manifest = materialization / 'SELECTED_IMAGES.sha256'
    if sha256(manifest) != ready['selected_manifest_sha256']:
        raise ValueError('Actual selected JPEG content manifest changed')
    directory = Path(ready['destination']).resolve()
    names = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (directory / name).resolve()
        if path.parent != directory or name in names or sha256(path) != digest:
            raise ValueError('Selected JPEG differs from the verified local bundle: ' + name)
        names.add(name)
    plist = Path(ready['images_list'])
    paths = [Path(line) for line in plist.read_text().splitlines() if line]
    if len(names) != 20000 or len(paths) != 20000 or {p.name for p in paths} != names or sha256(plist) != ready['images_list_sha256']:
        raise ValueError('Actual complete selected-image list differs')
    if sha256(Path(ready['annotations'])) != ready['annotation_sha256']:
        raise ValueError('Original training JSON changed')
    val_list = root / 'data/val_full.txt'
    val_paths = [Path(line) for line in val_list.read_text().splitlines() if line]
    if len(val_paths) != 5000 or len({p.name for p in val_paths}) != 5000 or not all(p.is_file() for p in val_paths):
        raise ValueError('Complete5000 validation inputs unavailable')
    if names & {p.name for p in val_paths}:
        raise ValueError('Selected training and validation image identities overlap')
    proofs = [('stream', root / 'runs/RUN_TRIFLOW_20K_STREAM_VERIFY_S0/VERIFY_STREAM.json'),
              ('resume', root / 'runs/RUN_TRIFLOW_20K_RESUME_VERIFY_S0_R1/RESUME_VERIFICATION.json'),
              ('evaluator', root / 'runs/RUN_TRIFLOW_20K_EVALUATOR_CONTRACT_S0_R1/CONTRACT_VERIFICATION.json')]
    proof_hashes = {}
    for name, path in proofs:
        proof = json.loads(path.read_text(encoding='utf-8'))
        if proof.get('passed') is not True:
            raise ValueError('Actual execution proof has not passed: ' + name)
        proof_hashes[name] = sha256(path)
        if name == 'resume':
            for source, digest in proof['training_sources'].items():
                if sha256(Path(__file__).with_name(source)) != digest:
                    raise ValueError('Engineering resume proof tested another trainer version: ' + source)
        if name == 'evaluator' and proof['evaluator_sha256'] != sha256(Path(__file__).with_name('evaluate_20k.py')):
            raise ValueError('Actual evaluator guard source changed')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv,noheader'], capture_output=True, text=True, check=True)
    rows = [line for line in gpu.stdout.splitlines() if line.strip()]
    if any('python' in line.lower() and line.split(',')[0].strip() != str(os.getpid()) for line in rows):
        raise RuntimeError('Another Python GPU process is active; preserve its work and defer this execution')
    weights = 'D:/coco_wire/models/yolo26m-seg.pt'
    frozen = FrozenYOLO(weights, 'D:/coco_wire/vendor_8.4.100', device='cuda')
    samples = [frozen.extract(paths[i]) for i in (0, len(paths)-1)]
    integrity = frozen.verify_frozen()
    frozen.close()
    if not all(s['native_replay_exact'] for s in samples) or integrity.get('passed') is not True:
        raise ValueError('Actual native frozen input contract failed')
    free = shutil.disk_usage(root).free
    if free < 15*1024**3:
        raise RuntimeError('Insufficient output disk')
    result = {'passed': True, 'selected_train_images': 20000, 'validation_images': 5000,
        'all_selected_image_bytes_rehashed': True, 'registered_selection_receipt': ready,
        'data_ready_sha256': sha256(ready_path), 'execution_proof_sha256': proof_hashes,
        'actual_native_probe_image_ids': [int(Path(s['image_path']).stem) for s in samples],
        'frozen_integrity': integrity, 'base_weights_sha256': OFFICIAL_SHA256,
        'interpreter': sys.executable, 'free_disk_bytes': free,
        'scope': 'Selected-data and execution correctness; method outcome not measured'}
    dump_json(root / 'runs' / a.run_id / 'PREFLIGHT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
