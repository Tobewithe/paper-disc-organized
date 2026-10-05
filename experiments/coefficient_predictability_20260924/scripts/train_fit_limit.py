"""Small training-only fit diagnostic; no validation data or early stopping."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from feature_probes import Probe
from official_pipeline import setup, load, write, target_rois, roi_losses, optimize, predict


def main(a):
    setup()
    index=json.loads((a.bank/'INDEX.json').read_text())
    valid=[r for r in index['fit'] if r['n']>0]
    items=random.Random(20260924).sample(valid,min(a.images,len(valid)))
    write(a.out/'SELECTION.json',dict(images=[r['image_id'] for r in items],
        selection='Uniform fixed-seed sample of fit images with official positives; no loss-based selection.'))
    images=[]; hs=[]; ps=[]; cs=[]
    for item in items:
        image=load(a.bank/'images'/f'{item["image_id"]:012d}.pt')
        item['phi']=image['phi']
        rois=target_rois(image)
        # Keep the small subset's interpolation in GPU memory; every other
        # experiment computes the same ROIs as needed.
        images.append((image,rois))
        ids=[r['raw_id'] for r in image['rows']]
        hs.append(image['h'][ids]);cs.append(image['coeff'][ids]);ps.append(image['phi'].repeat(len(ids),1))
    for j,item in enumerate(items):item['donor_phi']=items[(j+1)%len(items)]['phi']
    h,p,c=torch.cat(hs),torch.cat(ps),torch.cat(cs)
    stats=(h.mean(0),h.std(0).clamp_min(.01),p.mean(0),p.std(0).clamp_min(.01),c.std(0).clamp_min(.1))
    total=len(h)
    models={};optimizers={}
    arms=('linear','mlp','mlp_large','condition_mean','condition_true','condition_shuffle')
    for arm in arms:
        torch.manual_seed(0)
        models[arm]=Probe(arm,*stats).cuda()
        optimizers[arm]=torch.optim.AdamW(models[arm].parameters(),lr=.001,weight_decay=.0001)
    start=time.monotonic();oracle_rows=[]
    for image,rois in images:
        for r,roi in zip(image['rows'],rois):
            initial=image['coeff'][r['raw_id']].cuda()
            _,sp=optimize(initial,roi,a.iterations)
            _,sz=optimize(torch.zeros_like(initial),roi,a.iterations)
            oracle_rows.append({**r,'original_loss':float(roi_losses(initial[None,None],[roi])[0]),
                'oracle_loss':min(sp['loss'],sz['loss']),'from_prediction':sp,'from_zero':sz})
        print(json.dumps(dict(stage='train_subset_oracle',instances=len(oracle_rows),total=total)),flush=True)
    write(a.out/'ORACLE_ROWS.json',oracle_rows)
    original=sum(r['original_loss'] for r in oracle_rows)/total
    oracle=sum(r['oracle_loss'] for r in oracle_rows)/total
    history=[]
    def assess(epoch):
        totals={arm:0.0 for arm in arms}
        with torch.no_grad():
            for (image,rois),item in zip(images,items):
                coeff,_=predict(models,image,item)
                values=roi_losses(coeff,rois)
                for arm,value in zip(arms,values.tolist()):totals[arm]+=value
        losses={arm:value/total for arm,value in totals.items()}
        entry=dict(epoch=epoch,mean_loss=losses,oracle_gap_closed={arm:(original-value)/(original-oracle) for arm,value in losses.items()},elapsed=time.monotonic()-start)
        history.append(entry);write(a.out/'HISTORY.json',history);write(a.out/'PROGRESS.json',entry)
        print(json.dumps(entry),flush=True)
    assess(0)
    for epoch in range(1,a.epochs+1):
        order=list(range(len(items)));random.Random(20260924+epoch).shuffle(order)
        for first in range(0,len(order),8):
            minibatch=order[first:first+8]
            n=sum(len(images[j][0]['rows']) for j in minibatch)
            for opt in optimizers.values():opt.zero_grad(set_to_none=True)
            for j in minibatch:
                image,rois=images[j];coeff,_=predict(models,image,items[j])
                losses=roi_losses(coeff,rois)*image['segmentation_gain']
                assert torch.isfinite(losses).all()
                (losses.sum()/n).backward()
            for arm,net in models.items():
                torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True)
                optimizers[arm].step()
        if epoch%10==0 or epoch==a.epochs:assess(epoch)
    for arm,net in models.items():torch.save(dict(state_dict=net.state_dict(),arm=arm,epoch=a.epochs),a.out/f'{arm}_last.pt')
    write(a.out/'SUMMARY.json',dict(images=len(items),instances=total,epochs=a.epochs,original_loss=original,
        finite_oracle_loss=oracle,last=history[-1],scope='Training-only finite-budget fitting on fixed images; no validation selection and no evidence of test-time generalization.'))
    write(a.out/'COMPLETE.json',dict(images=len(items),instances=total,epochs=a.epochs))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('bank','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--images',type=int,default=16);p.add_argument('--epochs',type=int,default=200);p.add_argument('--iterations',type=int,default=100)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);main(args)
