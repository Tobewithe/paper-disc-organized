"""GT-informed coefficient-only least-squares diagnostic; no neural model training.

Uses the exact cached prototypes and prediction boxes from the 300-image probe.
This feasible oracle is not a certified optimum for IoU or a deployable method.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import contextlib
import io
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,SEED,FIELDS,sha,write_csv,write_json,decode,pixel_metrics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    run=a.run;out=run/'capacity_check';out.mkdir(exist_ok=False)
    gtpath=ROOT/'data/annotations/instances_val2017.json'
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(gtpath))
    statuses=json.loads((run/'statuses.json').read_text());eligible=[s for s in statuses if s['status']=='ok']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import csv
    with (ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv').open(encoding='utf-8-sig') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
    write_json(out/'protocol.json',dict(training=False,source_run=str(run),script_sha256=sha(__file__),gt_sha256=sha(gtpath),
        formula='min_c mean_pos((cP-1)^2)+mean_neg((cP+1)^2)+lambda||c||^2; lambda=1e-4 trace(G)/32',
        sampling='All matched pairs from frozen 300-image run, no outcome filtering',
        support='Fixed predicted box at input resolution, then exact mask scaling; exclude crowd. GT outside box retained in coverage denominator.',
        limits='GT-informed feasible coefficient replacement, not model training, not strict IoU upper bound. Oracle improvement does not prove a learned predictor will achieve it.'))
    rows=[];start=time.monotonic()
    for k,s in enumerate(eligible,1):
        iid=s['image_id'];cache=np.load(run/'tensors'/f'{iid}.npz')
        p=torch.as_tensor(cache['proto'],device='cuda');c=torch.as_tensor(cache['coeff'],device='cuda');boxes=torch.as_tensor(cache['boxes'],device='cuda')
        ids=cache['annotation_ids'].tolist();shape=tuple(cache['shape']);input_shape=tuple(cache['input_shape']);capture=dict(shape=shape,input_shape=input_shape)
        anns=gt.imgToAnns[iid];raster={ann['id']:torch.as_tensor(gt.annToMask(ann).astype(bool),device='cuda') for ann in anns}
        ordinary=[ann for ann in anns if not ann.get('iscrowd',0)];union=torch.stack([raster[x['id']] for x in ordinary]).any(0)
        same=torch.stack([raster[x['id']] for x in ordinary if x['category_id']==gt.anns[ids[0]]['category_id']]).any(0)
        crowd=[raster[x['id']] for x in anns if x.get('iscrowd',0)];valid=~torch.stack(crowd).any(0) if crowd else torch.ones(shape,device='cuda',dtype=torch.bool)
        expanded=F.interpolate(p[None],input_shape,mode='bilinear',align_corners=False)[0]
        features=ops.scale_masks(expanded[:,None],shape)[:,0]
        support=ops.crop_mask(torch.ones((2,*input_shape),device='cuda',dtype=torch.uint8),boxes)
        support=ops.scale_masks(support[:,None],shape)[:,0]>.5
        fitted=[];diagnostics=[]
        for j,aid in enumerate(ids):
            own=raster[aid]&valid;pos=own&support[j];neg=~own&valid&support[j]
            npos=int(pos.sum());nneg=int(neg.sum())
            if npos==0 or nneg==0:
                fitted.append(c[j]);diagnostics.append(dict(fit_status='missing_positive_or_negative',solve_residual=None));continue
            xp=features[:,pos].double();xn=features[:,neg].double()
            gram=xp@xp.T/npos+xn@xn.T/nneg;rhs=xp.mean(1)-xn.mean(1)
            ridge=1e-4*gram.trace()/len(c[j]);system=gram+ridge*torch.eye(len(c[j]),device='cuda',dtype=torch.float64)
            fit=torch.linalg.solve(system,rhs)
            residual=float((system@fit-rhs).norm()/rhs.norm().clamp_min(1e-12))
            assert torch.isfinite(fit).all() and residual<1e-7
            fitted.append(fit.float());diagnostics.append(dict(fit_status='ok',solve_residual=residual))
        new=torch.stack(fitted);stacked=torch.cat([c,new]);masks,_=decode(stacked,p,boxes.repeat(2,1),capture)
        for j,aid in enumerate(ids):
            values=pixel_metrics(masks['cropped'][[j,j+2]],raster[aid],union,same,valid)
            own=raster[aid]&valid;ga=int(own.sum());ceiling=float((own&support[j]).sum()/max(ga,1))
            row=dict(image_id=iid,target_annotation=aid,target_ici=float(meta[aid]['ici_same']),category_id=gt.anns[aid]['category_id'],
                     gt_area=ga,box_support_coverage_ceiling=ceiling,original_coefficient_norm=float(c[j].norm()),fitted_coefficient_norm=float(new[j].norm()),**diagnostics[j])
            for field,v in values.items():row.update({f'base_{field}':float(v[0]),f'oracle_{field}':float(v[1]),f'delta_{field}':float(v[1]-v[0])})
            rows.append(row)
        if k%25==0 or k==len(eligible):print(json.dumps(dict(completed=k,total=len(eligible),seconds=round(time.monotonic()-start,1))),flush=True)
    write_csv(out/'per_target.csv',rows)
    draws=np.load(run/'bootstrap_draws.npz');image_ids=draws['image_ids'];lookup={int(i):k for k,i in enumerate(image_ids)}
    weights=np.array([np.bincount(d,minlength=len(image_ids)) for d in draws['draws']]);stats=[]
    for group in ['all','high','low']:
        part=[r for r in rows if group=='all' or (r['target_ici']>.5)==(group=='high')]
        sums=np.zeros(len(image_ids));count=np.zeros(len(image_ids))
        for r in part:count[lookup[r['image_id']]]+=1
        denom=weights@count;valid=denom>0
        for metric in FIELDS:
            sums[:]=0
            for r in part:sums[lookup[r['image_id']]]+=r['delta_'+metric]
            boot=(weights[valid]@sums)/denom[valid]
            st=dict(group=group,metric=metric,targets=len(part),mean=float(sums.sum()/count.sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)))
            stats.append(st)
    write_csv(out/'summary.csv',stats)
    result=dict(status='COMPLETE',images=len(eligible),targets=len(rows),elapsed_seconds=time.monotonic()-start,
                fit_missing=sum(r['fit_status']!='ok' for r in rows),file_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'COMPLETE.json',result);print(json.dumps(result),flush=True)
    for row in stats:
        if row['group']=='high':print(json.dumps(row),flush=True)

if __name__=='__main__':main()
