"""Local receipt-checked transfer; skip already verified weights and preserve failed partials."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import uuid


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def target(root, name):
    result = (root/name).resolve()
    if root not in result.parents: raise ValueError('Path outside intended study: '+name)
    return result


def transfer(root, remote, remote_root, name, expected):
    destination = target(root, name); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected['size_bytes'] and sha(destination) == expected['sha256']:
        return {'file': name, 'sha256': expected['sha256'], 'status': 'existing_verified'}
    partial = destination.with_name(destination.name+'.'+uuid.uuid4().hex[:8]+'.part')
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4',
                    remote+':'+remote_root+'/'+name, str(partial)], check=True)
    assert partial.stat().st_size == expected['size_bytes'] and sha(partial) == expected['sha256'], name
    partial.replace(destination)
    return {'file': name, 'sha256': expected['sha256'], 'status': 'received_verified'}


def main(args):
    root = Path(args.root).resolve(); receipt_name = 'SPM_V4_1_'+('WEIGHTS' if args.kind == 'weights' else 'RESULTS_LIGHT')+'.tar.gz.receipt.json'
    receipt = json.loads((root/receipt_name).read_text(encoding='utf-8'))
    if args.kind == 'weights':
        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(transfer, root, args.remote, args.remote_root, name, expected) for name, expected in receipt['files'].items()]
            for future in as_completed(futures):
                result = future.result(); results.append(result); print(result['status'], result['file'], flush=True)
        dump(root/'LOCAL_WEIGHTS_ACCEPTANCE.json', {'mode': 'per_file_transfer_skip_verified_existing',
               'source_receipt_sha256': sha(root/receipt_name), 'verified_files': results,
               'complete_expected_weights': len(results) == len(receipt['files']),
               'server_bundle_not_downloaded_as_a_whole': True})
    else:
        transfer(root, args.remote, args.remote_root, receipt['bundle'], receipt)
        bundle = root/receipt['bundle']
        with tarfile.open(bundle, 'r:gz') as archive:
            for member in archive.getmembers():
                target(root, member.name)
                if not member.isfile(): raise ValueError('Expected regular files only: '+member.name)
            archive.extractall(root)
        for name, expected in receipt['files'].items():
            path = target(root, name)
            assert path.stat().st_size == expected['size_bytes'] and sha(path) == expected['sha256'], name
        dump(root/'LOCAL_LIGHT_ACCEPTANCE.json', {'bundle_sha256': sha(bundle), 'bundle_size_bytes': bundle.stat().st_size,
               'verified_file_count': len(receipt['files']), 'all_expected_files_SHA_verified': True,
               'scope': 'server_bundle_snapshot_at_packaging_time'})
        print('LIGHT_RECEIVED_AND_VERIFIED', len(receipt['files']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('kind', choices=['weights', 'light'])
    parser.add_argument('--root', required=True); parser.add_argument('--remote', default='a25023104@192.168.71.10')
    parser.add_argument('--remote-root', default='/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023')
    main(parser.parse_args())
