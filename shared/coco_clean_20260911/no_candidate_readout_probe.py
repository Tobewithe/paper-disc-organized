"""Trace S014 high-density failures without a good post-conf mask candidate.

Temporary per-GT oracle coefficients only; no network/model training.
All fitting pixels belong to the evaluated image: capacity-opportunity diagnostic,
not held-out/generalization evaluation or a mathematical expressivity bound.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,sha,write_json
from structure_candidate_trace import save_csv
from candidate_lineage_probe import read,need
from crossimage_response_experiment import gt_input_regions


def coefficient_logits(proto,c,shape,bias=0.):
    return F.interpolate((c@proto.flatten(1)).reshape(1,1,*proto.shape[-2:]),shape,
                         mode='bilinear',align_corners=False)[0,0]+bias


def exact_threshold(z,y,total_positive):
    """Best threshold on all valid input support; values > threshold foreground."""
    zz=z.detach().double().cpu().numpy();yy=y.detach().cpu().numpy().astype(bool)
    order=np.argsort(-zz,kind='stable');v=zz[order];labels=yy[order]
    ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1]
    tp=np.cumsum(labels)[ends];fp=ends+1-tp
    iou=tp/(total_positive+fp)
    k=int(np.argmax(iou));end=ends[k]
    tau=(v[end]+v[end+1])/2 if end+1<len(v) else np.nextafter(v[end],-np.inf)
    # Check float32 threshold quantization; zero is always an explicit control.
    options=[np.float32(tau),np.nextafter(np.float32(v[end]),np.float32(-np.inf)),np.float32(0.)]
    def metric(t):
        p=zz>float(t);return float((p&yy).sum()/max(total_positive+(p&~yy).sum(),1))
    chosen=max(options,key=lambda t:(metric(t),-abs(float(t))))
    return float(chosen),metric(chosen)


def pixel_metric(binary,shape,own,neighbor,bg,other,valid,support):
    original=(ops.scale_masks(binary[None,None],shape)[0,0]>.5).cpu().numpy()
    mask=original&valid;area=int(own.sum());tp=int((mask&own).sum())
    sn=int((mask&neighbor).sum());back=int((mask&bg).sum());oth=int((mask&other).sum())
    return dict(mask_iou=tp/max(area+sn+back+oth,1),coverage=tp/max(area,1),
        same_neighbor=sn/max(area,1),background=back/max(area,1),other_category=oth/max(area,1),
        tp=tp,fn=area-tp,same_neighbor_pixels=sn,background_pixels=back,other_pixels=oth,
        gt_valid_pixels=area,crop_coverage=float((own&support).sum()/max(area,1)))


def fit_readout(x,y,c0,has_bias,total_positive,max_iter):
    """Full valid input support, diagonal scaling without centering (no hidden bias)."""
    scale=x.square().mean(0).sqrt().clamp_min(.01)
    xs=x/scale
    initial=torch.cat([c0*scale,torch.zeros(1,device='cuda')]) if has_bias else c0*scale
    w=torch.nn.Parameter(initial.clone());states=[]
    def objective(with_dice):
        z=xs@w[:len(c0)]+(w[-1] if has_bias else 0.)
        bce=F.binary_cross_entropy_with_logits(z,y)
        reg=1e-6*w.square().mean()
        dice=1-(2*(z.sigmoid()*y).sum()+1)/(z.sigmoid().sum()+total_positive+1)
        return bce+reg+(dice if with_dice else 0.)
    def save(phase,iterations,evals):
        loss=objective(phase=='bce_dice');grad=torch.autograd.grad(loss,w)[0]
        states.append(dict(phase=phase,coefficient=(w[:len(c0)]/scale).detach().cpu().numpy(),
            bias=float(w[-1].detach()) if has_bias else 0.,objective=float(loss.detach()),
            gradient_max=float(grad.abs().max()),iterations=iterations,evaluations=evals,
            coefficient_norm=float(torch.linalg.vector_norm(w[:len(c0)]/scale).detach())))
    save('initial',0,0)
    for phase in ['bce','bce_dice']:
        opt=torch.optim.LBFGS([w],lr=1.,max_iter=max_iter,max_eval=max_iter*2,
            tolerance_grad=1e-6,tolerance_change=1e-9,history_size=20,line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad();loss=objective(phase=='bce_dice');need(torch.isfinite(loss).item(),'Nonfinite oracle objective');loss.backward();return loss
        opt.step(closure)
        s=opt.state[w];save(phase,int(s['n_iter']),int(s['func_evals']))
    return states


def process(gt,iid,targets,out,max_iter):
    rawpath=ROOT/'diagnostics/structure_main300_20260911/raw'/f'{iid}.npz'
    cachepath=ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz'
    prior=json.loads((ROOT/'diagnostics/candidate_lineage300_20260912/images'/f'{iid}.json').read_text())
    need(prior['input_hashes']=={'raw':sha(rawpath),'cache':sha(cachepath)},'S014 input cache provenance changed')
    with np.load(rawpath) as z:raw={k:z[k] for k in z.files}
    with np.load(cachepath) as z:item={k:z[k] for k in z.files}
    proto=torch.tensor(item['proto'],device='cuda');shape=tuple(map(int,item['shape']));inp=tuple(map(int,item['input_shape']))
    mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
    gindex={int(aid):g for g,aid in enumerate(raw['annotation_ids'])}
    scores=raw['class_scores'].max(1);cls=raw['class_scores'].argmax(1);cats=sorted(gt.cats)
    raster,_,input_valid=gt_input_regions(gt,iid,item)
    orig={a['id']:gt.annToMask(a).astype(bool) for a in gt.imgToAnns[iid]}
    valid=np.ones(shape,bool);union=np.zeros(shape,bool)
    for a in gt.imgToAnns[iid]:
        if a.get('iscrowd',0):valid&=~orig[a['id']]
        else:union|=orig[a['id']]
    expanded=F.interpolate(proto[None],inp,mode='bilinear',align_corners=False)[0].flatten(1).T
    rows=[];metrics=[];phases=[];parameters={};start=time.monotonic()
    for target in targets:
        aid=int(target['annotation_id']);ann=gt.anns[aid];g=gindex[aid];label=cats.index(ann['category_id'])
        geo=raw['bbox_iou'][g];raw50=geo>=.5;class50=raw50&(cls==label);score50=class50&(scores>.001)
        if not raw50.any():stage='no_raw_box50'
        elif not class50.any():stage='no_correct_argmax_box50'
        elif not score50.any():stage='no_postconf_correct_box50'
        elif aid not in mapping:stage='postconf_box50_but_unmatched_final'
        else:stage='final_bbox50_matched_mask_failure'
        if aid in mapping:
            src=int(raw['nonempty_indices'][mapping[aid]]);anchor='fixed_original_bbox50_match'
        elif score50.any():
            pool=np.flatnonzero(score50);src=int(pool[np.argmax(geo[pool])]);anchor='GT_best_geometry_postconf_candidate'
        else:src=-1;anchor='none'
        own=orig[aid]&valid;area=int(own.sum())
        row=dict(image_id=iid,annotation_id=aid,category_id=ann['category_id'],area=ann['area'],ici=float(target['ici']),
            best_postconf_mask_iou=float(target['best_mask_iou']),stage=stage,anchor=anchor,source_index=src,
            best_raw_box_iou=float(geo.max()),raw_box50_count=int(raw50.sum()),class_box50_count=int(class50.sum()),
            postconf_box50_count=int(score50.sum()),raw_box75_count=int((geo>=.75).sum()),
            postconf_box75_count=int(((geo>=.75)&(cls==label)&(scores>.001)).sum()),gt_valid_pixels=area)
        if src<0 or area==0:
            rows.append(dict(**row,fit_status='no_geometry_anchor' if src<0 else 'no_valid_gt_pixels'));continue
        box=torch.tensor(raw['boxes_input'][src:src+1],device='cuda');c=torch.tensor(raw['coefficients'][src],device='cuda')
        support=ops.crop_mask(torch.ones((1,*inp),device='cuda',dtype=torch.uint8),box)[0].bool()
        original_support=(ops.scale_masks(support[None,None].to(torch.uint8),shape)[0,0]>.5).cpu().numpy()
        same=np.zeros(shape,bool)
        for q in gt.imgToAnns[iid]:
            if not q.get('iscrowd',0) and q['category_id']==ann['category_id']:same|=orig[q['id']]
        neighbor=same&~own&valid;bg=~union&valid;other=union&~same&~own&valid
        fitmask=support&input_valid;pos=fitmask.flatten().nonzero().flatten();y=raster[aid].flatten()[pos].float()
        totalpos=int((raster[aid]&input_valid).sum());z=coefficient_logits(proto,c,inp)
        original=ops.crop_mask((z>0)[None].to(torch.uint8),box)[0]
        need(torch.equal(original,ops.process_mask(proto,c[None],box,inp,upsample=True)[0]),'Original official decode mismatch')
        rle=mu.encode(np.asfortranarray((ops.scale_masks(original[None,None],shape)[0,0]>.5).cpu().numpy().astype(np.uint8)))
        official_iou=float(mu.iou([rle],[gt.annToRLE(ann)],[0])[0,0])
        need(official_iou<.75,'Selected anchor contradicts no-good-candidate cohort')
        base=dict(image_id=iid,annotation_id=aid,ici=row['ici'],anchor=anchor)
        def measure(binary,arm,bias=0.,phase=''):
            m=pixel_metric(binary,shape,own,neighbor,bg,other,valid,original_support)
            pred=binary.bool()&input_valid
            tp=int((pred&raster[aid]).sum());fp=int((pred&~raster[aid]).sum())
            rec=dict(**base,arm=arm,phase=phase,bias=bias,input_iou=tp/max(totalpos+fp,1),**m)
            metrics.append(rec);return rec
        m0=measure(original,'original');row.update(anchor_box_iou=float(geo[src]),anchor_official_mask_iou=official_iou,
            original_valid_mask_iou=m0['mask_iou'],crop_coverage=m0['crop_coverage'],fit_pixels=len(pos),fit_positives=int(y.sum()),input_gt_pixels=totalpos)
        if len(pos)==0 or totalpos==0 or int(y.sum())==0 or int(y.sum())==len(y):
            rows.append(dict(**row,fit_status='degenerate_input_classes'));continue
        x=expanded[pos]
        replay=float((x@c-z.flatten()[pos]).abs().max());need(replay<2e-4,f'Prototype interpolation replay {replay}')
        tau,_=exact_threshold(z.flatten()[pos],y,totalpos)
        threshold=ops.crop_mask((z>tau)[None].to(torch.uint8),box)[0]
        measure(threshold,'threshold_oracle',bias=-tau)
        for has_bias,arm in [(False,'free_coefficient'),(True,'free_coefficient_bias')]:
            states=fit_readout(x,y,c,has_bias,totalpos,max_iter);candidates=[]
            for s in states:
                cc=torch.tensor(s['coefficient'],device='cuda');logit=coefficient_logits(proto,cc,inp,s['bias'])
                binary=ops.crop_mask((logit>0)[None].to(torch.uint8),box)[0]
                pred=binary.bool()&input_valid
                tp=int((pred&raster[aid]).sum());fp=int((pred&~raster[aid]).sum());fit_iou=tp/max(totalpos+fp,1)
                candidates.append((fit_iou,s,binary))
                phases.append(dict(**base,arm=arm,phase=s['phase'],input_iou=fit_iou,bias=s['bias'],objective=s['objective'],
                    gradient_max=s['gradient_max'],iterations=s['iterations'],evaluations=s['evaluations'],coefficient_norm=s['coefficient_norm']))
            # Choose among declared original/BCE/BCE+Dice states using fitting input GT;
            # do not select using original-resolution metric. This is still same-image oracle.
            chosen=max(candidates,key=lambda q:q[0]);_,s,binary=chosen
            measure(binary,arm,s['bias'],s['phase'])
            parameters[f'{aid}_{arm}_coefficient']=s['coefficient'];parameters[f'{aid}_{arm}_bias']=np.array(s['bias'])
            parameters[f'{aid}_original_coefficient']=c.cpu().numpy();parameters[f'{aid}_box']=box.cpu().numpy()
        rows.append(dict(**row,fit_status='ok',feature_replay_max_abs=replay))
    np.savez_compressed(out/'images'/f'{iid}.npz',**parameters)
    doc=dict(image_id=iid,targets=rows,metrics=metrics,optimization=phases,seconds=time.monotonic()-start,
        input_hashes={'raw':sha(rawpath),'cache':sha(cachepath)})
    write_json(out/'images'/f'{iid}.json',doc);return doc


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--images',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-iter',type=int,default=60)
    p.add_argument('--budget-seconds',type=float,default=1800);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'diagnostics/candidate_lineage300_20260912/gt.csv'
    selected=[r for r in read(source) if float(r['ici'])>.5+1e-10 and r['official_mask75']=='False' and r['score_mask75']=='False']
    need(len(selected)==317,'Expected locked S014 cohort of 317')
    ids=sorted({int(r['image_id']) for r in selected})
    if a.images:ids=ids[:a.images]
    selected=[r for r in selected if int(r['image_id']) in ids]
    protocol=dict(images=ids,target_ids=[int(r['annotation_id']) for r in selected],cohort='S014 high ICI, official mask75 failed AND no correct-argmax postconf candidate reaches mask75',
        network_training=False,oracle=True,scope='Same-image full valid input-support fitting; no pixel/image holdout; diagnostic achieved feasibility, NOT optimized upper bound or deployable AP.',
        anchors='Use original official fixed bbox50 match when available. Otherwise GT-best-box-IoU correct argmax postconf raw candidate if bboxIoU>=.5, reported separately. No anchor if neither exists.',
        fit='All valid pixels inside original prediction support, nearest-exact GT resize using rounded original letterbox. Freeze P/box. Float32 prototype interpolation replay <2e-4. Original-resolution official decode for final pixels. Crowd excluded only spatial domain.',
        arms=['original','threshold_oracle','free_coefficient','free_coefficient_bias'],
        optimizer=dict(name='LBFGS',max_iter_per_stage=a.max_iter,line_search='strong_wolfe',stages=['BCE','BCE+softDice'],
            initialization='original coefficient, zero bias',feature_scaling='RMS scale, NO centering',regularization='1e-6 mean squared scaled parameters',
            selection='Best input-grid IoU among original, BCE, BCE+Dice; original-resolution metrics not used for checkpoint selection'),
        limitations='Single deterministic temporary solve, not 3 training seeds. A failed solve cannot establish missing representational capacity. Input rasterization differs from original; report both metrics and crop support.',
        script_sha256=sha(__file__),source_sha256=sha(source),annotations_sha256=sha(ROOT/'data/annotations/instances_val2017.json'),
        helper_hashes={name:sha(Path(__file__).with_name(name)) for name in ['crossimage_response_experiment.py','frozen_mechanism_probe.py']},ops_sha256=sha(ops.__file__))
    if a.resume:need(json.loads((a.out/'protocol.json').read_text())==protocol,'Resume protocol mismatch')
    else:a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir();write_json(a.out/'protocol.json',protocol)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    start=time.monotonic()
    for n,iid in enumerate(ids,1):
        path=a.out/'images'/f'{iid}.json'
        if path.exists():doc=json.loads(path.read_text())
        else:doc=process(gt,iid,[r for r in selected if int(r['image_id'])==iid],a.out,a.max_iter)
        progress=dict(images=n,total=len(ids),seconds=time.monotonic()-start,last_image=iid,last_seconds=doc['seconds'])
        print(json.dumps(progress),flush=True);write_json(a.out/'progress.json',progress)
        if time.monotonic()-start>a.budget_seconds and n<len(ids):write_json(a.out/'BUDGET_PAUSE.json',progress);return
    allrows={k:[] for k in ['targets','metrics','optimization']}
    for iid in ids:
        doc=json.loads((a.out/'images'/f'{iid}.json').read_text())
        for key in allrows:allrows[key].extend(doc[key])
    for key,rows in allrows.items():save_csv(a.out/f'{key}.csv',rows)
    write_json(a.out/'COMPLETE.json',dict(status='COMPLETE',network_training=False,oracle=True,images=len(ids),targets=len(selected),
        seconds=time.monotonic()-start,hashes={str(q.relative_to(a.out)):sha(q) for q in a.out.rglob('*') if q.is_file()}))


if __name__=='__main__':main()
