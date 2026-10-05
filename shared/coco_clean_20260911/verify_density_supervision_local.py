"""Local deterministic archive, receipt and summary checks (no GPU dependency)."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent/'diagnostics'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):
    with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

report={'checks':[]}
for name in ['low_density_readout1021_20260912','assignment300_20260912','official_loss_readout887_v2_20260912']:
    out=BASE/name
    if not (out/'COMPLETE.json').exists():
        report['checks'].append({'name':name,'status':'NOT_AVAILABLE'});continue
    doc=json.loads((out/'COMPLETE.json').read_text())
    assert doc['status']=='COMPLETE'
    for rel,h in doc['hashes'].items():assert sha(out/rel)==h,(name,rel)
    report['checks'].append({'name':name,'status':'PASS','receipt_files':len(doc['hashes'])})
low=BASE/'low_density_readout1021_20260912'
if (low/'DENSITY_ANALYSIS.json').exists():
    analysis=json.loads((low/'DENSITY_ANALYSIS.json').read_text())
    for n,h in analysis['csv_hashes'].items():assert sha(low/n)==h,n
    targets=read(low/'DENSITY_COMPARISON_TARGETS.csv')
    assert len(targets)==996 and len({r['annotation_id'] for r in targets})==996
    for interaction in analysis['interactions']:
        field=interaction['metric']
        high=np.mean([float(r[field]) for r in targets if r['density']=='high'])
        other=np.mean([float(r[field]) for r in targets if r['density']=='other'])
        assert abs(100*(high-other)-interaction['high_minus_other_pp'])<1e-9
    report['checks'].append({'name':'density_aggregate_recalculation','status':'PASS','targets':len(targets)})
assignment=BASE/'assignment300_20260912'
if (assignment/'ANALYSIS.json').exists():
    rows=read(assignment/'targets.csv')
    assert len(rows)==3635 and len({r['annotation_id'] for r in rows})==3635
    high=[r for r in rows if float(r['ici'])>.5+1e-10 and r['official_mask75']=='False' and r['no_good_postconf_mask75']=='True']
    assert len(high)==317
    assert sum(r['final_source_assignment']=='own_gt_positive' for r in high)==215
    assert sum(r['final_source_assignment']!='unmatched_final' for r in high)==237
    spatial=json.loads((assignment/'SUPERVISION_DOMAIN_ANALYSIS.json').read_text())
    assert sha(assignment/'pixel_supervision_domain.csv')==spatial['csv_sha256']
    report['checks'].append({'name':'assignment_target_recalculation','status':'PASS','gt':len(rows)})
report['scope']='Local artifact/hash/CSV arithmetic checks, not independent COCO decoding or semantic review.'
(BASE/'DENSITY_SUPERVISION_LOCAL_VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
