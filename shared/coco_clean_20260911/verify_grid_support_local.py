"""Verify copied remote manifest plus derived fixed-cohort S019 arithmetic."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent/'diagnostics';out=BASE/'grid_support887_20260912'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=json.loads((out/'EXPORT_SHA256.json').read_text())
for name,h in manifest.items():assert sha(out/name)==h,name
assert sha(BASE/'grid_support887_20260912.tar.gz')=='c51766c71776fb8ca0e615cc2775a336e73c980d2649bfdd90e96747770a6cdf'
with open(out/'targets.csv',newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert len(rows)==4435 and len({r['annotation_id'] for r in rows})==887
analysis=json.loads((out/'ANALYSIS.json').read_text())
for r in analysis['groups']:
    rr=[q for q in rows if q['density']==r['density'] and q['arm']==r['arm']]
    assert len(rr)==r['n']
    assert sum(float(q['coco_iou'])>=.75 for q in rr)==r['recovered75']
    assert abs(100*np.mean([float(q['coco_iou']) for q in rr])-r['mean_coco_iou'])<1e-10
protocol=json.loads((out/'protocol.json').read_text());execution=out/'execution_source'
assert sha(execution/'grid_support_factorial_probe.py')==protocol['script_sha256']
assert sha(execution/'loss.py')==protocol['loss_sha256']
assert sha(execution/'ops.py')==protocol['ops_sha256']
decode=json.loads((out/'DECODE_WITNESS.json').read_text())
assert decode['targets']==3548 and decode['all_saved_coefficient_iou_replayed'] and decode['coefficient_gradient_amplitude_all_pass']
assert sha(out/'decode_roundoff_witness.csv')==decode['csv_sha256']
derived=['PAIRED_TARGETS.csv','FACTORIAL_ANALYSIS.json','GRID_SUPPORT_COMPARISON.png','GRID_SUPPORT_COMPARISON.svg','GRID_SUPPORT_EFFECTS.png','GRID_SUPPORT_EFFECTS.svg']
doc=dict(status='PASS',archive_sha256=sha(BASE/'grid_support887_20260912.tar.gz'),remote_manifest_files=len(manifest),
    target_rows=4435,unique_targets=887,all_group_arithmetic_recomputed=True,source_hashes_match=True,
    decode_witness_receipt_verified=True,local_derivatives={p:sha(out/p) for p in derived},
    scope='Local file integrity and arithmetic, not semantic audit or independent COCO rerun.')
(out/'LOCAL_VERIFICATION.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
print(json.dumps(doc,indent=2))
