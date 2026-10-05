"""Report all contrasts; cluster bootstrap only for paired target metrics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from recording import atomic_json,now

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def cluster_ci(values,groups):
    names=sorted(set(groups));index={g:i for i,g in enumerate(names)}
    sums=np.zeros(len(names));counts=np.zeros(len(names))
    for v,g in zip(values,groups):sums[index[g]]+=v;counts[index[g]]+=1
    draw=np.random.default_rng(20260915).integers(0,len(names),(4000,len(names)))
    means=sums[draw].sum(1)/counts[draw].sum(1)
    return float(np.mean(values)),*map(float,np.quantile(means,[.025,.975]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.study/'protocol.json').read_text());official=[];targets=[]
    for run in cfg['runs']:
        if run['kind']!='evaluation':continue
        folder=a.study/'runs'/run['run_id']
        assert (folder/'COMPLETE.json').exists()
        official.extend(read(folder/'official_metrics.csv'));targets.extend(read(folder/'targeted_per_instance.csv'))
    write(a.out/'all_official_metrics.csv',official)
    contrasts=[('reg_only','original'),('original','baseline'),('reg_only','baseline')]
    omap={(r['checkpoint'],r['branch'],r['task'],r['arm']):r for r in official}
    ap=[];paired=[]
    excluded={'arm','task','images','predictions','branch','checkpoint'}
    target_keys=['raw_best_box_iou','raw_p3_best_box_iou','raw_center_error_norm','raw_same_class_box50',
                 'raw_correct_class_at_best','final_confidence','final_box_iou','mask_iou','target_coverage',
                 'prediction_purity','boundary_f1','final_box50','final_mask50','final_mask75',
                 'same_neighbor_leak_pred','background_leak_pred']
    for checkpoint in ['last','best']:
        for branch in cfg['evaluation_branches']:
            for method,reference in contrasts:
                for task in ['bbox','segm']:
                    base=omap[(checkpoint,branch,task,reference)];new=omap[(checkpoint,branch,task,method)]
                    for key in base:
                        if key in excluded:continue
                        b=float(base[key]);m=float(new[key]);ap.append(dict(checkpoint=checkpoint,branch=branch,task=task,
                            contrast=method+' minus '+reference,metric=key,reference=b,method=m,delta_points=100*(m-b)))
                for cohort in sorted({r['cohort'] for r in targets}):
                    subset=[r for r in targets if r['checkpoint']==checkpoint and r['branch']==branch and r['cohort']==cohort]
                    by={(r['arm'],r['annotation_id']):r for r in subset};ids=sorted({r['annotation_id'] for r in subset})
                    for key in target_keys:
                        b=np.array([float(by[(reference,i)][key]) for i in ids]);m=np.array([float(by[(method,i)][key]) for i in ids])
                        delta,lo,hi=cluster_ci(m-b,[by[(reference,i)]['image_id'] for i in ids])
                        paired.append(dict(checkpoint=checkpoint,branch=branch,cohort=cohort,contrast=method+' minus '+reference,
                                           metric=key,n=len(ids),reference=float(b.mean()),method=float(m.mean()),
                                           delta=delta,ci_low=lo,ci_high=hi))
    write(a.out/'official_contrasts.csv',ap);write(a.out/'targeted_contrasts.csv',paired)
    atomic_json(a.out/'COMPLETE.json',dict(status='complete',finished_at=now(),scope='one seed; fixed last primary; best supplementary',
                 bootstrap='4000 paired image clusters, descriptive intervals without multiplicity correction',
                 ap_uncertainty='point estimates only; no claim of seed stability or AP statistical significance'))
    print('SUMMARY_COMPLETE',a.out,flush=True)

if __name__=='__main__':main()
