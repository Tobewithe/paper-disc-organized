"""Verify local copies and transition identities; no new experiment."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
out=root/'diagnostics/relative_ownership_20260911'
def read(name):
    with (out/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

checked=[]
for receipt_name in ['TASK_COMPLETE.json','DEVELOPMENT_TRANSFERS_COMPLETE.json']:
    receipt=json.loads((out/receipt_name).read_text())
    for name,digest in receipt.get('hashes',receipt.get('files',{})).items():
        assert (out/name).exists(),name
        assert sha(out/name)==digest,name
        checked.append(name)
base={int(r['target_annotation']):r for r in read('development_spatial.csv') if r['arm']=='initial'}
arms={(r['arm'],int(r['target_annotation'])):r for r in read('development_spatial.csv')}
maximum=0.
for r in read('development_transfers.csv'):
    aid=int(r['annotation_id']);original=base[aid];changed=arms[(r['arm'],aid)]
    for metric,domain in [('coverage','own'),('same_neighbor','same_neighbor'),('background','background')]:
        delta=float(changed[metric])-float(original[metric])
        transfer=float(r['added_'+domain])-float(r['removed_'+domain])
        maximum=max(maximum,abs(delta-transfer))
        assert abs(delta-transfer)<1e-12
summary=read('task_summary.csv');reference=summary[0]
for r in summary[1:]:
    assert all(r[k]==reference[k] for k in reference if k!='arm')
corrections=read('development_corrections.csv')
assert all(int(r['original_area'])==int(r['resulting_area']) for r in corrections)
selection=json.loads((out/'selection.json').read_text())
assert len(selection['development'])==160 and len(selection['evaluation'])==600
assert not set(selection['development'])&set(selection['evaluation'])
result=dict(status='PASS',verified_receipt_entries=len(checked),unique_files=len(set(checked)),
     development_transition_max_error=maximum,exact_area_check_records=len(corrections),
     identical_locked_task_rows=len(summary),training=False,source_sha256=sha(Path(__file__)))
(out/'LOCAL_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
