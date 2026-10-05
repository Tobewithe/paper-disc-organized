"""Compare candidate supervised mask objectives under a matched logit-RMS budget.

This is a GT-informed local loss-direction diagnostic, NOT held-out task accuracy.
All objectives use the same frozen prototypes, predicted-box support and decoded GT.
No model checkpoint is changed. The RMS constraint is measured on own box support.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,io,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,decode,pixel_metrics,sha,write_json,write_csv

METRICS=['coverage','same_neighbor','neighbor','background','mask_iou']
OBJECTIVES=['bce','balanced','neighbor_weighted','pair_rank']
BUDGETS=[.02,.05,.10]

def summary(rows,draws,path):
    ids=draws['image_ids'];D=draws['draws'];pos={int(i):k for k,i in enumerate(ids)};stats=[]
    for group in ['all','high','low']:
        part=[r for r in rows if group=='all' or (r['target_ici']>.5)==(group=='high')]
        for budget in BUDGETS:
            q=[r for r in part if r['budget']==budget]
            baseline={(r['image_id'],r['target_annotation']):r for r in q if r['variant']=='bce_euclidean'}
            for name in sorted({r['variant'] for r in q}):
                rr=[r for r in q if r['variant']==name]
                for control in ['zero','bce_euclidean']:
                    for m in METRICS:
                        sums=np.zeros(len(ids));count=sums.copy()
                        for r in rr:
                            b=r['base_'+m] if control=='zero' else baseline[(r['image_id'],r['target_annotation'])]['new_'+m]
                            sums[pos[r['image_id']]]+=r['new_'+m]-b;count[pos[r['image_id']]]+=1
                        den=count[D].sum(1);boot=sums[D].sum(1)/den
                        stats.append(dict(group=group,budget=budget,variant=name,control=control,metric=m,mean=float(sums.sum()/count.sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),targets=int(count.sum()),images=int((count>0).sum())))
    write_csv(path,stats);return stats

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    run=args.run.resolve();out=args.out.resolve();out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    with (ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv').open(encoding='utf-8-sig') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
    eligible=[s for s in json.loads((run/'statuses.json').read_text()) if s['status']=='ok']
    write_json(out/'protocol.json',dict(training=False,uses_gt_for_directions=True,source=str(run),script_sha256=sha(__file__),primary_budget=.05,budgets=BUDGETS,objectives=OBJECTIVES,
        geometry='Euclidean versus regularized prototype Gram; same per-target decoded-logit RMS budget on fixed predicted-box support',
        ridge='0.01 trace(G)/K',scope='Conditional GT-informed loss diagnostic. Image GT informs gradients; not deployable inference or validation accuracy.',
        decision='Choose direction only if IoU and same-neighbor improve relative to ordinary BCE without reducing coverage; independent images and learned model still required.'))
    rows=[];diagnostics=[];start=time.monotonic()
    for n,s in enumerate(eligible,1):
        iid=s['image_id'];cache=np.load(run/'tensors'/f'{iid}.npz');p=torch.tensor(cache['proto'],device='cuda');c=torch.tensor(cache['coeff'],device='cuda');boxes=torch.tensor(cache['boxes'],device='cuda')
        aids=cache['annotation_ids'].tolist();shape=tuple(map(int,cache['shape']));ishape=tuple(map(int,cache['input_shape']));capture=dict(shape=shape,input_shape=ishape)
        anns=gt.imgToAnns[iid];raster={a['id']:torch.tensor(gt.annToMask(a).astype(bool),device='cuda') for a in anns}
        ordinary=[a for a in anns if not a.get('iscrowd',0)];union=torch.stack([raster[a['id']] for a in ordinary]).any(0)
        same=torch.stack([raster[a['id']] for a in ordinary if a['category_id']==gt.anns[aids[0]]['category_id']]).any(0)
        crowd=[raster[a['id']] for a in anns if a.get('iscrowd',0)];valid=~torch.stack(crowd).any(0) if crowd else torch.ones(shape,dtype=torch.bool,device='cuda')
        feature=ops.scale_masks(F.interpolate(p[None],ishape,mode='bilinear',align_corners=False)[0][:,None],shape)[:,0].double()
        support=ops.crop_mask(torch.ones((2,*ishape),device='cuda',dtype=torch.uint8),boxes)
        support=(ops.scale_masks(support[:,None],shape)[:,0]>.5)&valid
        cd=c.double();flat=feature.flatten(1);z=(cd@flat).reshape(2,*shape);common=support.all(0)
        grams=[];grads={k:[] for k in OBJECTIVES};region_log=[]
        for j,aid in enumerate(aids):
            region=support[j];X=feature[:,region];yy=raster[aid][region].double();zz=z[j][region];nn=len(yy)
            assert nn>0
            G=X@X.T/nn;grams.append(G);res=zz.sigmoid()-yy;plain=X@res/nn
            masks=[raster[aid]&region, same&~raster[aid]&region, union&~same&~raster[aid]&region, ~union&region]
            gradparts=[]
            for mr in masks:
                rr=(z[j][mr].sigmoid()-raster[aid][mr].double())
                gradparts.append(feature[:,mr]@rr/max(nn,1))
            torch.testing.assert_close(sum(gradparts),plain,atol=1e-9,rtol=1e-8)
            weights=torch.ones_like(yy);pos=yy>.5;neg=~pos
            if pos.any() and neg.any():weights[pos]=float(nn)/(2*int(pos.sum()));weights[neg]=float(nn)/(2*int(neg.sum()))
            balanced=X@(res*weights)/nn
            sw=torch.ones_like(yy);neighbor=(same&~raster[aid])[region];sw[neighbor]=4.;sw/=sw.mean()
            weighted=X@(res*sw)/nn
            # Ground-truth exclusivity defines pixel owner; no penalty in ambiguous overlap.
            extra=torch.zeros_like(plain);regions=[]
            for owner in [0,1]:
                mr=common&raster[aids[owner]]&~raster[aids[1-owner]]
                if mr.any():
                    delta=z[owner][mr]-z[1-owner][mr]
                    slope=-torch.sigmoid(-delta) if j==owner else torch.sigmoid(-delta)
                    regions.append(feature[:,mr]@slope/mr.sum())
            if regions:extra=sum(regions)/len(regions)
            for name,v in [('bce',plain),('balanced',balanced),('neighbor_weighted',weighted),('pair_rank',plain+extra)]:grads[name].append(v)
            region_log.append(dict(image_id=iid,target_annotation=aid,target_ici=float(meta[aid]['ici_same']),box_pixels=nn,
                own_pixels=int(masks[0].sum()),neighbor_pixels=int(masks[1].sum()),background_pixels=int(masks[3].sum()),
                grad_norm=float(plain.norm()),own_grad_norm=float(gradparts[0].norm()),neighbor_grad_norm=float(gradparts[1].norm()),background_grad_norm=float(gradparts[3].norm()),
                neighbor_to_total_cos=float(F.cosine_similarity(gradparts[1][None],plain[None],dim=1)),
                own_neighbor_cos=float(F.cosine_similarity(gradparts[0][None],gradparts[1][None],dim=1))))
        diagnostics.extend(region_log)
        variants=[('zero',0.,c)];budget_logs={}
        for name in OBJECTIVES:
            grad=torch.stack(grads[name])
            for metric in ['euclidean','gram']:
                ds=[]
                for j in range(2):
                    if metric=='euclidean':d=-grad[j]
                    else:
                        G=grams[j];M=G+.01*G.trace()/len(G)*torch.eye(len(G),device='cuda',dtype=torch.float64)
                        d=-torch.linalg.solve(M,grad[j])
                    assert torch.isfinite(d).all() and float(d@grad[j])<=1e-12
                    ds.append(d)
                ds=torch.stack(ds)
                for budget in BUDGETS:
                    changed=[];errs=[]
                    for j in range(2):
                        response=ds[j]@feature[:,support[j]];rms=response.square().mean().sqrt();original_rms=z[j][support[j]].square().mean().sqrt().clamp_min(1e-8)
                        target=budget*original_rms;step=target/rms.clamp_min(1e-15);new=cd[j]+step*ds[j];actual=((new-cd[j])@feature[:,support[j]]).square().mean().sqrt()/original_rms
                        if rms>1e-14:assert abs(float(actual)-budget)<1e-8
                        changed.append(new.float());errs.append(float(actual))
                    vname=name+'_'+metric;variants.append((vname,budget,torch.stack(changed)));budget_logs[(vname,budget)]=errs
        stacked=torch.cat([v[2] for v in variants]);decoded,_=decode(stacked,p,boxes.repeat(len(variants),1),capture)
        official=ops.process_mask(p,c,boxes,ishape,upsample=True);official=ops.scale_masks(official[:,None],shape)[:,0]>.5
        assert torch.equal(official,decoded['cropped'][:2])
        values=[pixel_metrics(decoded['cropped'][j::2],raster[aid],union,same,valid) for j,aid in enumerate(aids)]
        for k,(variant,budget,cc) in enumerate(variants[1:],1):
            for j,aid in enumerate(aids):
                r=dict(image_id=iid,target_annotation=aid,target_ici=float(meta[aid]['ici_same']),variant=variant,budget=budget,actual_logit_rms=budget_logs[(variant,budget)][j])
                r.update({f'base_{m}':float(values[j][m][0]) for m in METRICS});r.update({f'new_{m}':float(values[j][m][k]) for m in METRICS});rows.append(r)
        if n%50==0:print(json.dumps(dict(completed=n,total=len(eligible),seconds=round(time.monotonic()-start,1))),flush=True)
    write_csv(out/'per_target.csv',rows);write_csv(out/'gradient_regions.csv',diagnostics)
    stats=summary(rows,np.load(run/'bootstrap_draws.npz'),out/'summary.csv')
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',training=False,images=len(eligible),rows=len(rows),seconds=time.monotonic()-start,script_sha256=sha(__file__)))
    for r in stats:
        if r['group']=='high' and r['budget']==.05 and r['control']=='bce_euclidean' and r['metric'] in ['mask_iou','same_neighbor','coverage']:print(json.dumps(r),flush=True)

if __name__=='__main__':main()
