"""Read representative immutable cache records; no model is instantiated."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import traceback

import torch
from online_runtime import dump, load_json, resolve_runtime_config, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if os.name == 'nt':
        raise RuntimeError('Cache stays on the authorized remote server')
    cfg = resolve_runtime_config(load_json(args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index = load_json(Path(cfg['cache'])/'INDEX.json')
    records = []
    for split in ('fit', 'dev'):
        ids = sorted([int(r['image_id']) for r in index[split]], key=lambda i: hashlib.sha256(str(i).encode()).hexdigest())[:10]
        for iid in ids:
            path = Path(cfg['cache'])/'images'/f'{iid:012d}.pt.gz'
            with gzip.open(path, 'rb') as f:
                x = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)
            meta = load_json(path.with_name(f'{iid:012d}.meta.json'))
            op = x['operator']
            records.append({'split': split, 'image_id': iid, 'proto_dtype': str(x['proto'].dtype),
                            'c0_dtype': str(x['c0'].dtype), 'h0_dtype': str(op['h0'].dtype),
                            'input_uint8_present': 'input_uint8' in x, 'archive_sha256': sha256(path),
                            'metadata_archive_hash_matches': sha256(path) == meta['compressed_sha256'],
                            'proto_sha256_in_meta': meta.get('proto_sha256'), 'n': len(x['rows'])})
    dump(out/'STORAGE_AUDIT.json', {'sample_rule': '10 smallest SHA256(image_id decimal) per fixed FIT/DEV index, not outcome selected',
                                  'scope': '20 sampled images only; not a full-corpus storage census', 'records': records,
                                  'model_instantiated': False, 'cache_mutated': False})
    dump(out/'COMPLETE.json', {'completed': True, 'sample_images': len(records), 'model_instantiated': False})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        import sys
        if '--out' in sys.argv:
            dump(Path(sys.argv[sys.argv.index('--out')+1])/'FAILURE.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
