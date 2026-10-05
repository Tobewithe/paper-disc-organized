"""Exact per-target IoU/FP-budget identity on S037's GT-assisted 640 domain."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();p=a.run
    with (p/'coverage.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['status']=='complete' and float(r['requested_coverage'])==.9]
    by=defaultdict(list)
    for r in rows:
        by[int(r['annotation_id']),'original' if r['arm']=='original' else 'coefficient'].append(r)
    summary=[]
    for cohort in ['all_matches','residual_all3fail']:
        for group in ['high','nonhigh']:
            for mode in ['original','coefficient']:
                selected=[v for (aid,m),v in by.items() if m==mode and ((v[0]['high']=='True')==(group=='high'))
                    and (cohort=='all_matches' or v[0]['coefficient_all_fail']=='True')]
                counts=dict(n=len(selected),same_alone_over_budget_allseeds=0,background_alone_over_budget_allseeds=0,
                    total_over_budget_allseeds=0,remove_same_fits_budget_allseeds=0)
                for vv in selected:
                    flag=[]
                    for r in vv:
                        cov=float(r['coverage']);ne=float(r['neighbor']);bg=float(r['background']);ot=float(r['other'])
                        budget=cov/.75-1
                        if abs(float(r['iou'])-cov/(1+ne+bg+ot))>=1e-12:raise RuntimeError('IoU partition identity failed')
                        flag.append((ne>budget,bg>budget,ne+bg+ot>budget,ne+bg+ot>budget and bg+ot<=budget))
                    for j,k in enumerate(list(counts)[1:]):counts[k]+=all(v[j] for v in flag)
                summary.append(dict(cohort=cohort,group=group,mode=mode,**counts))
    (p/'ERROR_BUDGET.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
