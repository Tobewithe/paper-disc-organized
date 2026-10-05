"""S019: 2x2 mask-supervision grid/support oracle, same frozen 887 failures.

GT-assisted temporary coefficient optimization only. No network/head training.
The four arms share GT-area normalization, parameter scaling, initialization,
budget, loss, crowd policy, and final prediction-box decoding.
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
from ultralytics.utils.loss import v8SegmentationLoss
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha
from crossimage_response_experiment import gt_input_regions

ARMS={'A160_GT':(160,'gt'),'B160_PRED':(160,'pred'),'C640_GT':(640,'gt'),'D640_PRED':(640,'pred')}


def solve(proto,truth,box,area,c0,scale,max_iter):
    size=truth.shape[-1]
    support=ops.crop_mask(torch.ones_like(truth)[None],box)[0].bool()
    n=int(support.sum())
    if not n:
        return c0.clone(),dict(status='empty_support_unchanged',pixels=0,positive_pixels=0,
            loss_before=0.,loss_after=0.,iterations=0,evaluations=0,gradient_max=0.,
            official_value_abs_error=0.,official_gradient_cosine=None)
    x=proto[:,support].T.contiguous();y=truth[support].float()
    xs=x/scale;w=torch.nn.Parameter((c0*scale).clone())
    # The denominator always uses the GT area, even for prediction-box support.
    factor=1./(size*size*area)
    def objective():return F.binary_cross_entropy_with_logits(xs@w,y,reduction='sum')*factor
    before=objective();cg=torch.autograd.grad(before,w)[0]*scale
    check=torch.nn.Parameter(c0.clone())
    official=v8SegmentationLoss.single_mask_loss(truth[None].float(),check[None],proto,box,area.reshape(1))
    og=torch.autograd.grad(official,check)[0]
    cos=float(F.cosine_similarity(cg,og,dim=0)) if bool(cg.any()) and bool(og.any()) else None
    err=abs(float(before.detach())-float(official.detach()))
    need(err<=1e-5+1e-4*abs(float(official.detach())),f'Official BCE value mismatch {err}')
    if cos is not None:need(cos>.9999,f'Official BCE gradient mismatch {cos}')
    else:need(not bool(cg.any()) and not bool(og.any()),'Only one gradient is zero')
    opt=torch.optim.LBFGS([w],lr=1.,max_iter=max_iter,max_eval=2*max_iter,tolerance_grad=1e-6,
        tolerance_change=1e-9,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=objective();need(bool(torch.isfinite(loss)),'Nonfinite loss');loss.backward();return loss
    opt.step(closure)
    after=objective();grad=torch.autograd.grad(after,w)[0];state=opt.state[w]
    need(float(after.detach())<=float(before.detach())+1e-6,'Objective increased')
    cc=(w/scale).detach();need(bool(torch.isfinite(cc).all()),'Nonfinite coefficient')
    return cc,dict(status='optimized',pixels=n,positive_pixels=int(y.sum()),loss_before=float(before.detach()),loss_after=float(after.detach()),
        iterations=int(state['n_iter']),evaluations=int(state['func_evals']),gradient_max=float(grad.abs().max()),
        official_value_abs_error=err,official_gradient_cosine=cos)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--max-iter',type=int,default=120)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--budget-seconds',type=float,default=2400)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    previous=ROOT/'diagnostics/official_loss_readout887_v2_20260912';assignment=ROOT/'diagnostics/assignment300_20260912'
    alltargets=read(previous/'targets.csv');need(len(alltargets)==887,'Locked cohort mismatch')
    ids=sorted({int(r['image_id']) for r in alltargets})
    if a.smoke:
        # Structural coverage only: first high, first other, and the empty GT crop.
        ids=sorted({min(int(r['image_id']) for r in alltargets if r['density']==d) for d in ['high','other']}|
            {int(r['image_id']) for r in alltargets if r['solve_status']=='empty_GT_crop_kept_unchanged'})
    targets=[r for r in alltargets if int(r['image_id']) in ids]
    protocol=dict(experiment='S019',network_training=False,oracle=True,images=ids,target_ids=[int(r['annotation_id']) for r in targets],
        selection='Exact S018 own-GT-positive fixed-final failed cohort, no success filtering. Smoke uses density/empty-support coverage only.',
        arms=ARMS,optimizer=dict(name='LBFGS',max_iter=a.max_iter,max_eval=2*a.max_iter,line_search='strong_wolfe',
            initialization='original coefficient',scaling='One shared original-prototype RMS inside GT160 crop per target across all four arms; full-prototype RMS fallback when GT160 crop empty; no centering'),
        objective='Pure BCE sum over crop / (grid area * original normalized GT-box area). GT-area denominator fixed across arms. No bias, Dice, regularization, or IoU-based state selection. Final optimizer state.',
        labels='Original COCO annToMask including all polygon parts; nearest-exact letterbox to640 and nearest to160. Full ordinary target, crowd NOT excluded from fitting in any arm. Padding is background in every arm.',
        decoding='All arms use same saved prediction box, 32x160 prototypes, official process_mask upsample640 and scale_masks then >.5; original COCO RLE IoU. Spatial metrics separately exclude crowd.',
        controls='A is rerun with shared support-matrix objective; compare with previous official full-map S018 solver to quantify numerical implementation drift. Four-arm main comparison always uses same new solver.',
        scope='Explored val, per-instance same-image GT-assisted achieved optimization; not method AP, historical training recipe, density causality or convergence upper bound.',
        script_sha256=sha(__file__),source_sha256=sha(previous/'targets.csv'),annotation_sha256=sha(ROOT/'data/annotations/instances_val2017.json'),
        helper_sha256=sha(Path(__file__).with_name('crossimage_response_experiment.py')),ops_sha256=sha(ops.__file__),
        loss_sha256=sha(__import__('ultralytics.utils.loss',fromlist=['']).__file__))
    if a.resume:need(json.loads((a.out/'protocol.json').read_text())==json.loads(json.dumps(protocol)),'Resume protocol changed')
    else:a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir();write_json(a.out/'protocol.json',protocol)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    start=time.monotonic()
    for num,iid in enumerate(ids,1):
        if (a.out/'images'/f'{iid}.json').exists():continue
        rr=[r for r in targets if int(r['image_id'])==iid]
        cachepath=ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz';superpath=assignment/'images'/f'{iid}.npz'
        oldimage=json.loads((previous/'images'/f'{iid}.json').read_text())
        need(sha(cachepath)==oldimage['input_hashes']['cache'] and sha(superpath)==oldimage['input_hashes']['supervision'],'Cache provenance changed')
        with np.load(cachepath) as z:item={k:z[k] for k in z.files}
        with np.load(superpath) as z:supervised={k:z[k] for k in z.files}
        proto=torch.tensor(item['proto'],device='cuda');expanded=F.interpolate(proto[None],(640,640),mode='bilinear',align_corners=False)[0]
        rasters,_,_=gt_input_regions(gt,iid,item);gindex={int(aid):g for g,aid in enumerate(supervised['annotation_ids'])}
        sourceindex={int(v):j for j,v in enumerate(supervised['original_final_sources'])}
        shape=tuple(map(int,item['shape']));original_masks={q['id']:gt.annToMask(q).astype(bool) for q in gt.imgToAnns[iid]}
        valid=np.ones(shape,bool);union=np.zeros(shape,bool)
        for q in gt.imgToAnns[iid]:
            if q.get('iscrowd',0):valid&=~original_masks[q['id']]
            else:union|=original_masks[q['id']]
        rows=[];params={}
        for r in rr:
            aid=int(r['annotation_id']);g=gindex[aid];src=int(r['source_index']);j=sourceindex[src]
            need(bool(supervised['foreground'][src]) and int(supervised['target_gt_index'][src])==g,'Not own GT source')
            c0=torch.tensor(item['coeff'][j],device='cuda');predbox=torch.tensor(item['boxes'][j:j+1],device='cuda')
            norm=torch.tensor(supervised['gt_boxes_normalized'][g:g+1],device='cuda');gtbox=ops.xywh2xyxy(norm)*640;area=norm[0,2:].prod()
            y640=rasters[aid];y160=F.interpolate(y640[None,None].float(),(160,160),mode='nearest')[0,0]
            need(np.array_equal(y160.cpu().numpy(),supervised['gt_masks160'][g]),'GT160 differs from S017')
            gt160support=ops.crop_mask(torch.ones((1,160,160),device='cuda'),gtbox/4)[0].bool()
            scale=proto[:,gt160support].square().mean(1).sqrt().clamp_min(.01) if bool(gt160support.any()) else proto.square().mean((1,2)).sqrt().clamp_min(.01)
            own=original_masks[aid]&valid;ownarea=int(own.sum());same=np.zeros(shape,bool);ann=gt.anns[aid]
            for q in gt.imgToAnns[iid]:
                if not q.get('iscrowd',0) and q['category_id']==ann['category_id']:same|=original_masks[q['id']]
            neighbor=same&~own&valid;bg=~union&valid;other=union&~same&~own&valid
            support_gt=ops.crop_mask(torch.ones((1,640,640),device='cuda'),gtbox)[0].bool()
            support_pred=ops.crop_mask(torch.ones((1,640,640),device='cuda'),predbox)[0].bool()
            pred_orig=(ops.scale_masks(support_pred[None,None].to(torch.uint8),shape)[0,0]>.5).cpu().numpy()
            soft=F.avg_pool2d(y640[None,None].float(),4,4)[0,0]
            nonempty=soft>0;mix=(soft>0)&(soft<1)
            base=dict(image_id=iid,annotation_id=aid,density=r['density'],ici=float(r['ici']),category_id=int(r['category_id']),
                area=float(r['area']),source_index=src,gt_valid_pixels=ownarea,gt_pixels_original=int(original_masks[aid].sum()),
                gt_pixels160=int(y160.sum()),mixed_proto_cell_fraction=float(mix.sum()/nonempty.sum().clamp_min(1)),
                loss_support_iou=float((support_gt&support_pred).sum()/(support_gt|support_pred).sum().clamp_min(1)),
                pred_support_outside_gt_fraction=float((support_pred&~support_gt).sum()/support_pred.sum().clamp_min(1)),
                neighbor_exposure_over_gt=float((neighbor&pred_orig).sum()/max(ownarea,1)),
                original_iou=float(r['original_coco_iou']),s018_official_iou=float(r['official_BCE160_GTbox_coco_iou']),
                previous_fullinput_iou=float(r['previous_fullinput_oracle_coco_iou']))
            def measure(cc,arm,extra):
                binary=ops.process_mask(proto,cc[None],predbox,(640,640),upsample=True)[0]
                mask=(ops.scale_masks(binary[None,None],shape)[0,0]>.5).cpu().numpy()
                value=float(mu.iou([mu.encode(np.asfortranarray(mask.astype(np.uint8)))],[gt.annToRLE(ann)],[0])[0,0])
                spatial=mask&valid;tp=int((spatial&own).sum());sn=int((spatial&neighbor).sum());bp=int((spatial&bg).sum());op=int((spatial&other).sum())
                row=dict(**base,arm=arm,coco_iou=value,coverage=tp/ownarea if ownarea else None,
                    same_neighbor=sn/ownarea if ownarea else None,background=bp/ownarea if ownarea else None,
                    other_error=op/ownarea if ownarea else None,valid_iou=tp/(ownarea+sn+bp+op) if ownarea else None,
                    tp=tp,same_neighbor_pixels=sn,background_pixels=bp,other_pixels=op,coefficient_norm=float(cc.norm()),**extra)
                rows.append(row);return value
            value=measure(c0,'original',{})
            need(abs(value-base['original_iou'])<1e-7,'Original decode mismatch')
            for arm,(size,domain) in ARMS.items():
                pp=proto if size==160 else expanded;yy=y160 if size==160 else y640
                box=(gtbox if domain=='gt' else predbox)*(size/640)
                cc,details=solve(pp,yy,box,area,c0,scale,a.max_iter)
                measure(cc,arm,details);params[f'{aid}_{arm}']=cc.cpu().numpy()
        np.savez_compressed(a.out/'images'/f'{iid}.npz',**params)
        write_json(a.out/'images'/f'{iid}.json',dict(image_id=iid,targets=rows,
            input_hashes=dict(cache=sha(cachepath),supervision=sha(superpath),previous=sha(previous/'images'/f'{iid}.json'))))
        if num%5==0 or num==len(ids):
            progress=dict(images=num,total=len(ids),seconds=time.monotonic()-start)
            write_json(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if time.monotonic()-start>a.budget_seconds and num<len(ids):
            write_json(a.out/'BUDGET_PAUSE.json',dict(images=num,total=len(ids)));return
    rows=[]
    for iid in ids:rows.extend(json.loads((a.out/'images'/f'{iid}.json').read_text())['targets'])
    need(len(rows)==len(targets)*5,'Missing factorial rows');save_csv(a.out/'targets.csv',rows)
    summary=[]
    for density in ['high','other']:
        for arm in ['original',*ARMS]:
            q=[r for r in rows if r['density']==density and r['arm']==arm]
            if not q:continue
            summary.append(dict(density=density,arm=arm,n=len(q),mean_coco_iou=100*float(np.mean([r['coco_iou'] for r in q])),
                recovered75=sum(r['coco_iou']>=.75 for r in q),worse_than_original=sum(r['coco_iou']<r['original_iou']-1e-6 for r in q),
                iteration_limit=sum(r.get('iterations',0)>=a.max_iter for r in q),empty_support=sum(r.get('status')=='empty_support_unchanged' for r in q)))
    write_json(a.out/'ANALYSIS.json',dict(groups=summary,scope=protocol['scope'],csv_sha256=sha(a.out/'targets.csv')))
    write_json(a.out/'COMPLETE.json',dict(status='COMPLETE',images=len(ids),targets=len(targets),network_training=False,oracle=True,
        seconds_this_invocation=time.monotonic()-start,hashes={str(q.relative_to(a.out)):sha(q) for q in a.out.rglob('*') if q.is_file()}))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
