"""Train-only feasibility of predicting whether a frozen mask correction helps.

Ridge models use prediction-only features. The source development images are
split into gate fitting/calibration; the source selection images are reserved
for diagnostic scoring. Inspect SPLIT.json for original versus fresh cohorts.
This does not report COCO AP.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
import learn_refinement as original
import component_seed_probe as probe


@torch.no_grad()
def gate_features(net, x, base, boxes_wh, scores):
    emb=net.features(x);c=net.output(emb).tanh()*.5
    corrected=probe.roi_correct(base,x,c,'local4')
    def summary(z):
        flat=z.flatten(1)
        vals=[flat.mean(1),flat.std(1),(flat>0).float().mean(1)]
        vals += [(flat.abs()<v).float().mean(1) for v in (.5,1.,2.,4.)]
        return torch.stack(vals,1)
    semantic=torch.cat((emb,c,summary(base),summary(corrected),
        F.adaptive_avg_pool2d(base.sigmoid()[:,None],(4,4)).flatten(1),
        F.adaptive_avg_pool2d(corrected.sigmoid()[:,None],(4,4)).flatten(1),
        boxes_wh.clamp(min=1).log(),scores[:,None]),1)
    return semantic


def main():
    p=argparse.ArgumentParser()
    for k in ('source','checkpoint','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);probe.setup();start=time.monotonic()
    bank=torch.load(a.source/'diagnostic_bank.pt',map_location='cpu',mmap=True,weights_only=False)
    fresh=json.loads((a.source/'SPLIT.json').read_text()).get('fresh_unseen',False)
    rows=[json.loads(l) for l in (a.source/'instances.jsonl').read_text().splitlines()]
    assert len(rows)==len(bank['records'])
    for row,record in zip(rows,bank['records']):
        assert all(row[k]==record[k] for k in ('image_id','annotation_id','raw_id','split'))
    images=sorted({r['image_id'] for r in rows if r['split']=='fit'})
    random.Random(20260916).shuffle(images);cut=int(.8*len(images))
    fit_ids=set(images[:cut]);cal_ids=set(images[cut:]);test_ids={r['image_id'] for r in rows if r['split']=='selection'}
    assert not fit_ids&cal_ids and not (fit_ids|cal_ids)&test_ids
    original.save(a.out/'SPLIT.json',dict(gate_fit_image_ids=sorted(fit_ids),gate_calibration_image_ids=sorted(cal_ids),
        heldout_image_ids=sorted(test_ids),fresh_refiner_unseen=fresh,
        policy='Split source fit images, fixed seed20260916. All are refiner-unseen.' if fresh else
               'Split available original refiner-fit images, fixed seed20260916.'))
    net=original.Refiner('local4').cuda().eval()
    net.load_state_dict(torch.load(a.checkpoint,weights_only=False)['state_dict'])
    features=[]
    for start_i in range(0,len(rows),128):
        rs=rows[start_i:start_i+128]
        x=bank['x'][start_i:start_i+128].cuda().float();b=bank['base'][start_i:start_i+128].cuda().float()
        wh=torch.tensor([[r['box_width_input'],r['box_height_input']] for r in rs],device='cuda')
        scores=torch.tensor([r['score'] for r in rs],device='cuda')
        features.append(gate_features(net,x,b,wh,scores).cpu().numpy())
    x=np.concatenate(features).astype(np.float64)
    np.save(a.out/'features.npy',x)
    fit=np.array([r['image_id'] in fit_ids for r in rows]);cal=np.array([r['image_id'] in cal_ids for r in rows]);test=~(fit|cal)
    y=np.array([r['values']['plain_half']['full'][0]-r['values']['baseline']['full'][0] for r in rows])
    mu=x[fit].mean(0);scale=x[fit].std(0).clip(min=1e-5);xnorm=(x-mu)/scale
    # Equal total fitting weight per image limits dominance by crowded images.
    counts=Counter(r['image_id'] for r in rows if r['image_id'] in fit_ids)
    weight=np.array([1/counts[r['image_id']] for r in rows if r['image_id'] in fit_ids])
    design=np.column_stack([np.ones(len(rows)),xnorm]);xf=design[fit];yf=y[fit]
    candidates=[];models={}
    thresholds=[-.01,-.005,0.,.005,.01]
    for lam in (1.,10.,100.):
        penalty=np.eye(design.shape[1])*lam;penalty[0,0]=0
        coef=np.linalg.solve(xf.T@(weight[:,None]*xf)+penalty,xf.T@(weight*yf))
        pred=design@coef;models[lam]=(coef,pred)
        for threshold in thresholds:
            use=pred[cal]>threshold
            candidates.append(dict(kind='ridge',ridge=lam,threshold=threshold,
                calibration_gain=float((y[cal]*use).mean()),calibration_use_rate=float(use.mean())))
    candidates += [dict(kind='all',calibration_gain=float(y[cal].mean()),calibration_use_rate=1.),
                   dict(kind='none',calibration_gain=0.,calibration_use_rate=0.)]
    # Prefer a simpler fixed policy at a numerical tie.
    chosen=max(candidates,key=lambda c:(c['calibration_gain'],c['kind'] in ('all','none'),-c.get('ridge',0)))
    original.save(a.out/'SELECTION.json',dict(candidates=candidates,chosen=chosen,
        selection_metric='Mean full-image IoU gain over all calibration prediction-GT pairs; no heldout labels used.'))
    if chosen['kind']=='ridge':
        coef,pred=models[chosen['ridge']];use=pred>chosen['threshold']
    else:
        coef=np.zeros(design.shape[1]);pred=np.zeros(len(rows));use=np.full(len(rows),chosen['kind']=='all')
    np.savez(a.out/'gate.npz',mean=mu,scale=scale,coef=coef,threshold=chosen.get('threshold',0.))
    original.save(a.out/'gate.json',dict(kind=chosen['kind'],ridge=chosen.get('ridge'),threshold=chosen.get('threshold'),
        corrected_mode='plain_half',feature_count=x.shape[1],head_checkpoint=str(a.checkpoint),
        input_policy='Existing refiner features plus predicted logits/correction, bbox sizes and score. No GT at inference.'))
    def summarize(ix,policy):
        ds=y[ix]*policy[ix]
        rr=[rows[i] for i in np.where(ix)[0]]
        before=np.array([r['values']['baseline']['full'][0] for r in rr])
        corrected=np.array([r['values']['plain_half']['full'][0] for r in rr])
        after=np.where(policy[ix],corrected,before)
        cov=np.array([r['values']['plain_half']['full'][1]-r['values']['baseline']['full'][1] for r in rr])*policy[ix]
        return dict(n=len(rr),selected=int(policy[ix].sum()),delta_iou_pp=float(100*ds.mean()),
            delta_coverage_pp=float(100*cov.mean()),repaired75=int(((before<.75)&(after>=.75)).sum()),
            damaged75=int(((before>=.75)&(after<.75)).sum()),net75=int((after>=.75).sum()-(before>=.75).sum()))
    result={}
    for label,ix in [('gate_fit',fit),('gate_calibration',cal),('heldout',test)]:
        result[label]={k:summarize(ix,p) for k,p in [('all',np.ones(len(rows),bool)),('learned',use),('oracle',y>0)]}
    # Compare to selecting the same number of corrections uniformly at random.
    all_heldout=result['heldout']['all'];rate=result['heldout']['learned']['selected']/int(test.sum())
    result['heldout_random_same_count_expectation']={k:all_heldout[k]*rate for k in ('delta_iou_pp','repaired75','damaged75','net75')}
    # Paired image bootstrap, all candidate-GT pairs for a drawn image stay together.
    test_order=sorted(test_ids);index={im:i for i,im in enumerate(test_order)}
    count=np.zeros(len(index));d=np.zeros(len(index));net=np.zeros(len(index))
    for i,r in enumerate(rows):
        if not test[i]:continue
        j=index[r['image_id']];count[j]+=1;d[j]+=y[i]*(int(use[i])-1)
        b=r['values']['baseline']['full'][0];z=r['values']['plain_half']['full'][0]
        net[j]+=(int((z if use[i] else b)>=.75)-int(z>=.75))
    draws=np.random.default_rng(20260916).multinomial(len(index),[1/len(index)]*len(index),size=2000)
    result['heldout_learned_minus_all']=dict(delta_mean_iou_pp=float(100*d.sum()/count.sum()),
        mean_iou_ci95=np.quantile(100*(draws@d)/(draws@count),[.025,.975]).tolist(),
        delta_pair_recall75_pp=float(100*net.sum()/count.sum()),
        pair_recall75_ci95=np.quantile(100*(draws@net)/(draws@count),[.025,.975]).tolist())
    original.save(a.out/'SUMMARY.json',dict(results=result,chosen=chosen,split_images=dict(fit=len(fit_ids),calibration=len(cal_ids),heldout=len(test_ids)),
        limitations=['Available-image subset of train2017, not full COCO validation.',
            'Pair-level diagnostics only, not COCO recall or AP. Multiple candidates per GT.',
            'All gate fitting/calibration/heldout images were excluded from original refiner fit and selection.' if fresh else
            'Gate training and calibration images were seen by the original refiner; heldout images were not.',
            'Oracle uses heldout GT for an upper-bound diagnostic only. It is not a deployment method.']))
    with (a.out/'predictions.jsonl').open('w') as f:
        for i,r in enumerate(rows):f.write(json.dumps(dict(image_id=r['image_id'],annotation_id=r['annotation_id'],raw_id=r['raw_id'],
            partition='fit' if fit[i] else 'calibration' if cal[i] else 'heldout',predicted_gain=float(pred[i]),apply=bool(use[i]),actual_gain=float(y[i])))+'\n')
    original.save(a.out/'COMPLETE.json',dict(rows=len(rows),elapsed_s=time.monotonic()-start))
    print(json.dumps(dict(chosen=chosen,heldout=result['heldout'],contrast=result['heldout_learned_minus_all'])),flush=True)


if __name__=='__main__':main()
