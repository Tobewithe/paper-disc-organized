"""Recheck exported evaluation hashes and means independently of the summarizer."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

root=Path(__file__).parent/'diagnostics/ownership_ranking_eval_20260912'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda name:list(csv.DictReader((root/name).open(encoding='utf-8-sig')))
receipt=json.loads((root/'COMPLETE.json').read_text())
for filename in ['COMPLETE.json','SUMMARY_COMPLETE.json']:
    for name,digest in json.loads((root/filename).read_text())['hashes'].items():assert sha(root/name)==digest,(filename,name)
for name,digest in json.loads((root/'PREDICTION_HASHES.json').read_text()).items():assert sha(root/'predictions'/f'{name}.json.gz')==digest,name
families=['bce_dice','coverage_only','rank_only','rank_coverage','pair_bce_coverage','ccl_coverage']
task={r['arm']:r for r in read('task_summary.csv')};family={r['family']:r for r in read('family_task_summary.csv')}
for f in families:
    for metric in ['mask_ap','mask_ap50','mask_ap75','r75_high','r75_low','pair75_high']:
        assert abs(np.mean([float(task[f'{f}_s{s}'][metric]) for s in [0,1,2]])-float(family[f][metric]))<1e-12
keys={};nrows={}
for name,column in [('gt_recovery.csv','annotation_id'),('spatial.csv','target_annotation'),('pair_recovery.csv',None)]:
    rows=read(name);initial=[r for r in rows if r['arm']=='initial'];nrows[name]=len(initial)
    key=(lambda r:(r['annotation_a'],r['annotation_b'])) if column is None else lambda r:r[column]
    expected={key(r) for r in initial};assert len(expected)==len(initial)
    for arm in receipt['arms']:
        armrows=[r for r in rows if r['arm']==arm];assert len(armrows)==len(initial) and {key(r) for r in armrows}==expected
    if name=='gt_recovery.csv':
        for arm in receipt['arms']:
            rr=[r for r in rows if r['arm']==arm]
            assert len(rr)==int(task[arm]['r75_all_n'])==3616
            for group in ['all','high','low']:
                yy=[r['hit75']=='True' for r in rr if group=='all' or (float(r['ici'])>.5+1e-10)==(group=='high')]
                assert abs(np.mean(yy)-float(task[arm]['r75_'+group]))<1e-12
report=dict(status='PASS',script_sha256=sha(Path(__file__)),evaluation_complete_sha256=sha(root/'COMPLETE.json'),official_prediction_files=19,images=500,cohorts=nrows,paired_cohorts=True,all_seed_and_family_task_means_recomputed=True,scope='Export hashes/cohort/point estimates only; semantic audit separate, no independent rerun of official COCOeval.')
(root.parent/'ownership_ranking_eval_LOCAL_VERIFY_20260912.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
