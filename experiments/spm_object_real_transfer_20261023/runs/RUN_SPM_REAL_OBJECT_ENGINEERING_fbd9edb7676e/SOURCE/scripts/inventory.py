"""Lightweight read-only inventory, no tensor allocation or training."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


root = Path(__file__).resolve().parents[1]
cfg = json.loads((root/'CONFIG.json').read_text())
parent, obj = Path(cfg['parent_root']), Path(cfg['object_root'])
paths = [parent/'ALL_DATASET.json', parent/'DEV_COCO.json', parent/cfg['native_checkpoint'],
         obj/'CONFIG_V0.json', obj/'DATA_LOCATION.json', obj/'artifacts/structured_last.pt',
         obj/'artifacts/template_inference.npz']
paths += sorted((parent/'scripts').glob('*.py'))
paths += [obj/'scripts/object_model.py', obj/'scripts/geometry.py', obj/'scripts/dataset.py', obj/'scripts/runtime.py']
data = json.loads((parent/'ALL_DATASET.json').read_text())
rows = [r for r in data['rows'] if r['domain'] == 'real' and r['split'] in ('train', 'dev')]
counts = {s: sum(r['split'] == s for r in rows) for s in ('train', 'dev')}
loc = json.loads((obj/'DATA_LOCATION.json').read_text())
paths.append(Path(loc['directory'])/'MANIFEST.json')
summary = {'external_files': {str(p): sha(p) for p in paths}, 'real_counts': counts,
           'disk_free_bytes': shutil.disk_usage(root).free, 'sample': rows[0], 'data_location': loc,
           'source_scope': 'hash-only read-only inventory, no GPU and no training'}
(root/'INVENTORY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ('sample', 'external_files')}, indent=2))
print('NATIVE_SHA', summary['external_files'][str(parent/cfg['native_checkpoint'])])
