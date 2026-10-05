"""Post-selection descriptive analysis of frozen development results."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics/relative_ownership_20260911'
def read(name):
    with (OUT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    rows=read('development_spatial.csv');ids=json.loads((OUT/'selection.json').read_text())['development']
    ii={iid:j for j,iid in enumerate(ids)};draw=np.random.default_rng(20260911).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    base={int(r['target_annotation']):r for r in rows if r['arm']=='initial'}
    fields=['coverage','same_neighbor','background','mask_iou','exclusive_same_neighbor','predicted_area_over_gt']
    contrasts=[]
    for arm in sorted({r['arm'] for r in rows}-{'initial'}):
        selected=[r for r in rows if r['arm']==arm]
        assert {int(r['target_annotation']) for r in selected}==set(base)
        for group in ['all','high','low']:
            rr=[r for r in selected if group=='all' or (float(r['target_ici'])>.5+1e-10)==(group=='high')]
            for field in fields:
                count=np.zeros(len(ids));sums=np.zeros(len(ids))
                for r in rr:
                    j=ii[int(r['image_id'])];count[j]+=1;sums[j]+=float(r[field])-float(base[int(r['target_annotation'])][field])
                den=draw@count;valid=den>0;boot=(draw@sums)[valid]/den[valid]*100
                contrasts.append(dict(arm=arm,group=group,metric=field,n=len(rr),mean_pp=float(sums.sum()/count.sum()*100),
                       ci_low_pp=float(np.quantile(boot,.025)),ci_high_pp=float(np.quantile(boot,.975))))
    corrections=read('development_corrections.csv');impact=[]
    for arm in sorted({r['arm'] for r in corrections}):
        rr=[r for r in corrections if r['arm']==arm]
        impact.append(dict(arm=arm,eligible_predictions=len(rr),changed_predictions=sum(int(r['changed_pixels'])>0 for r in rr),
             changed_input_pixels=sum(int(r['changed_pixels']) for r in rr),max_area_error=max(abs(int(r['original_area'])-int(r['resulting_area'])) for r in rr)))
    data=dict(scope='DESCRIPTIVE DEVELOPMENT ONLY. Same data selected hyperparameters, so CIs are not held-out confirmation and must not justify a new tuned arm.',
              images=len(ids),matched_targets=len(base),contrasts=contrasts,impact=impact)
    (OUT/'DEVELOPMENT_PAIRED_ANALYSIS.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    for r in contrasts:
        if r['group']=='high' and r['arm'].startswith('relative'):print(json.dumps(r))
    print(json.dumps(impact))

if __name__=='__main__':main()
