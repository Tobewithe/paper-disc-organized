"""Fail closed on a changed frozen bundle or incomplete data/environment preparation."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def verify(config_path, require_ready=False):
    root = Path(__file__).resolve().parents[3]
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    frozen = json.loads(Path(config['frozen_manifest']).read_text(encoding='utf-8-sig'))
    if frozen['config_sha256'] != sha(config_path): raise RuntimeError('CONFIG hash drift')
    if frozen['protocol_sha256'] != sha(config['protocol']): raise RuntimeError('PROTOCOL hash drift')
    for item in frozen['files']:
        if sha(root/item['path']) != item['sha256']: raise RuntimeError('frozen file drift: '+item['path'])
    if require_ready:
        ready = json.loads((Path(config['study_root'])/'data_environment_status.json').read_text())
        if ready.get('all_required_inputs_ready') is not True: raise RuntimeError('data/environment not ready')
        if '.'.join(map(str,sys.version_info[:3])) != config['expected_python']: raise RuntimeError('Python drift')
        for package, expected in config['expected_packages'].items():
            if importlib.metadata.version(package) != expected: raise RuntimeError('package drift: '+package)
        for split, digest in config['expected_annotations'].items():
            if sha(config[split+'_annotations']) != digest: raise RuntimeError('annotation drift: '+split)
    return {'status':'passed','frozen_files':len(frozen['files']), 'data_ready_checked':require_ready,
            'frozen_manifest_sha256':sha(config['frozen_manifest'])}

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--require-ready',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.config,args.require_ready)))
