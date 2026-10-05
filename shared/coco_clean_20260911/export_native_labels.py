"""Archive S020, its source, and execution receipts after decode verification."""
import hashlib,json,shutil,tarfile
from pathlib import Path
R=Path('/root/autodl-tmp/coco_clean_20260911');out=R/'diagnostics/native_label887_20260912'
assert json.loads((out/'COMPLETE.json').read_text())['status']=='COMPLETE'
assert json.loads((out/'DECODE_WITNESS.json').read_text())['all_saved_coefficients_replayed']
assert (R/'logs/native_label887_20260912.exit').read_text().strip()=='0'
dest=out/'execution_source';dest.mkdir(exist_ok=True)
for name in ['native_label_pipeline_probe.py','launch_native_labels.py','grid_support_factorial_probe.py',
             'crossimage_response_experiment.py','verify_native_labels_decode.py','candidate_lineage_probe.py']:
    shutil.copy2(R/name,dest/name)
shutil.copytree(R/'diagnostics/native_label_source_20260912',out/'native_source',dirs_exist_ok=True)
shutil.copy2(R/'logs/native_label887_20260912.log',out/'execution.log')
shutil.copy2(R/'logs/native_label887_20260912.exit',out/'exit_code.txt')
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
