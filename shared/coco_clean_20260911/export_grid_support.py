"""Archive completed S019 results, execution code, and smoke independently."""
import hashlib,json,shutil,tarfile
from pathlib import Path
R=Path('/root/autodl-tmp/coco_clean_20260911');out=R/'diagnostics/grid_support887_20260912'
assert json.loads((out/'COMPLETE.json').read_text())['status']=='COMPLETE'
dest=out/'execution_source';dest.mkdir(exist_ok=True)
for name in ['grid_support_factorial_probe.py','launch_grid_support.py','crossimage_response_experiment.py','verify_grid_support_decode.py']:
    shutil.copy2(R/name,dest/name)
for name in ['loss.py','ops.py']:
    shutil.copy2(R/'diagnostics/assignment300_20260912/official_source'/name,dest/name)
shutil.copy2(R/'logs/grid_support887_20260912.log',out/'execution.log')
shutil.copy2(R/'logs/grid_support887_20260912.exit',out/'exit_code.txt')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='EXPORT_SHA256.json'}
(out/'EXPORT_SHA256.json').write_text(json.dumps(manifest,indent=2))
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=out.name)
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),files=len(manifest))))
