"""This desktop supplies complete public data to the offline laptop once ready."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from frozen_io import dump_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    run = root / 'runs' / a.run_id
    download = root / 'runs/RUN_TRIFLOW_FULL_DATA_DOWNLOAD_S0'
    remote = 'D:/coco_wire/experiments/triflow_fullscale_training_20261007'
    dataset = 'D:/coco_wire/shared/coco2017/archives'
    dump_json(run / 'BRIDGE_INPUTS.json', {'ssh_alias': '28358lan', 'source_download_run': str(download),
        'remote_root': remote, 'remote_dataset_directory': dataset, 'execution': 'wait, transfer, checksum, materialize',
        'wait_limit_hours': 48, 'desktop_required_until_transfer_finishes': True})
    began = time.monotonic()
    while True:
        metadata = json.loads((download / 'run.json').read_text(encoding='utf-8-sig'))
        if metadata.get('execution_status') == 'completed':
            break
        if metadata.get('execution_status') == 'failed':
            raise RuntimeError('Download failed; preserve partial archive; no formal training started')
        if time.monotonic()-began > 48*3600:
            raise TimeoutError('Complete archive unavailable after 48 hours')
        time.sleep(10)
    receipt_file = download / 'DOWNLOAD_RECEIPT.json'
    receipt = json.loads(receipt_file.read_text(encoding='utf-8'))
    archive = Path(receipt['destination'])
    if sha256(archive) != receipt['archive_sha256']:
        raise ValueError('Actual local verified archive changed before transfer')
    dump_json(run / 'BRIDGE_PROGRESS.json', {'stage': 'copying_verified_full_archive',
              'started_at': datetime.now(timezone.utc).isoformat(), 'bytes': archive.stat().st_size})
    # Upload to an owned incoming filename. The remote materializer first hashes
    # these bytes and never writes into historical subset directories.
    with (run / 'scp.log').open('wb') as log:
        for attempt in range(3):
            code = subprocess.run(['scp', str(archive), f'28358lan:{dataset}/train2017.zip.incoming'],
                                  stdout=log, stderr=subprocess.STDOUT).returncode
            if code == 0:
                break
            if attempt == 2:
                raise RuntimeError('Archive transfer failed; owned incoming file and logs retained')
            time.sleep(10)
        subprocess.run(['scp', str(receipt_file), f'28358lan:{remote}/data/DOWNLOAD_RECEIPT.json'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    command = (f'& C:/Users/28358/anaconda3/envs/pytorch/python.exe -X utf8 {remote}/scripts/run_stage.py '
        f'--root {remote} --run-id RUN_TRIFLOW_FULL_DATA_MATERIALIZE_S0 --purpose original_complete_COCO_train2017_materialization '
        f'--script materialize_train.py -- --root {remote} --run-id RUN_TRIFLOW_FULL_DATA_MATERIALIZE_S0 '
        f'--archive {dataset}/train2017.zip.incoming --download-receipt {remote}/data/DOWNLOAD_RECEIPT.json '
        '--annotations D:/coco_wire/bgcr_native_20261002/data/annotations/instances_train2017.json '
        '--val-annotations D:/coco_wire/data/annotations/instances_val2017.json '
        '--destination D:/coco_wire/shared/coco2017/images/train2017 '
        '--val-images D:/coco_wire/data/images/val2017')
    dump_json(run / 'REMOTE_MATERIALIZATION_COMMAND.json', {'command': command, 'ssh_alias': '28358lan',
              'archive_sha256': receipt['archive_sha256'], 'download_receipt_sha256': sha256(receipt_file)})
    with (run / 'remote_materialization.log').open('wb') as log:
        subprocess.run(['ssh', '28358lan', command], stdout=log, stderr=subprocess.STDOUT, check=True)
    # The independent remote scheduler observes the completed Run + marker;
    # training and later evaluation then continue without this desktop process.
    dump_json(run / 'BRIDGE_COMPLETE.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'verified_archive_sha256': receipt['archive_sha256'], 'remote_full_data_ready': True,
        'remote_materialization_run': 'RUN_TRIFLOW_FULL_DATA_MATERIALIZE_S0',
        'subsequent_training_independent_of_desktop': True, 'local_return_status': 'pending'})
    print(json.dumps({'full_data_transfer_and_materialization_complete': True, 'remote_root': remote}), flush=True)


if __name__ == '__main__':
    main()
