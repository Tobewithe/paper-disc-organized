import json,shutil,tarfile
from pathlib import Path
from candidate_lineage_probe import ROOT,sha,write_json
out=ROOT/'diagnostics/shared_head_gradient887_20260912'
assert json.loads((out/'COMPLETE.json').read_text())['status']=='COMPLETE'
assert (ROOT/'logs/shared_head_gradient887_20260912.exit').read_text().strip()=='0'
dest=out/'execution_source';dest.mkdir(exist_ok=True)
for name in ['shared_head_gradient_probe.py','launch_shared_head_gradient.py','native_label_pipeline_probe.py','structure_candidate_trace.py','frozen_mechanism_probe.py']:
    shutil.copy2(ROOT/name,dest/name)
shutil.copy2(ROOT/'logs/shared_head_gradient887_20260912.log',out/'execution.log');shutil.copy2(ROOT/'logs/shared_head_gradient887_20260912.exit',out/'exit_code.txt')
manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='EXPORT_SHA256.json'};write_json(out/'EXPORT_SHA256.json',manifest)
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=out.name)
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),files=len(manifest))))
