"""S027: saved-head fit/pixel-holdout/COCO behavior plus bounded GT oracle.

No shared model training. Disjoint native sampled positions diagnose pixel fit;
full original COCO masks diagnose task-relevant shape for a fixed box attribution.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,contextlib,csv,io,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from rich_pixel_readout import GlobalHead,PixelHead,instance_features,loss_value
from pixel_ownership_refiner import build_features
from readout_input_probe import sha,write_json,stable_seed
from eval_readout_input_pilot import ici


def read_np(path):
    with np.load(path) as q:return {k:q[k] for k in q.files}


def csv_save(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def norm(x,n):return (x-n['mean'])/n['std']


def numbers(z,y,factor):
    if not len(z):return dict(n=0,bce=None,dice=None,objective=None,native_iou=None)
    bce=float(F.binary_cross_entropy_with_logits(z,y))*factor
    prob=z.sigmoid();dice=float(1-(2*(prob*y).sum()+1)/(prob.sum()+y.sum()+1))
    binary=z>0;truth=y>.5;inter=int((binary&truth).sum());union=int((binary|truth).sum())
    return dict(n=len(z),bce=bce,dice=dice,objective=bce+dice,native_iou=inter/union if union else 1.)


def solve(p,y,c0,factor,steps=120):
    # Same sampled BCE+Dice as shared head, exact fixed labels and positions.
    # No convex/global optimum claim; deterministic per-instance GT-assisted solve.
    scale=p.square().mean(0).sqrt().clamp_min(.01)
    xx=(p/scale).double();yy=y.double();w=torch.nn.Parameter((c0*scale).double())
    def loss():return loss_value((xx@w)[None],yy[None],yy.new_tensor([factor]))
    initial=float(loss().detach());opt=torch.optim.LBFGS([w],lr=1.,max_iter=steps,max_eval=steps*2,
         tolerance_grad=1e-7,tolerance_change=1e-10,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();value=loss()
        if not torch.isfinite(value):raise RuntimeError('Oracle nonfinite')
        value.backward();return value
    opt.step(closure);final=loss();grad=torch.autograd.grad(final,w)[0];state=opt.state[w]
    if float(final)>initial+1e-7:raise RuntimeError('Oracle objective increased')
    c=(w/scale.double()).detach()
    return c,dict(initial_objective=initial,final_objective=float(final.detach()),
        iterations=int(state['n_iter']),gradient_max=float(grad.abs().max()),hit_cap=int(state['n_iter'])>=steps,
        coeff_norm_ratio=float(c.norm()/c0.double().norm().clamp_min(1e-12)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--trained',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False);(out/'oracle_coefficients').mkdir();start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def progress(stage,**kw):
        r=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',r);print(json.dumps(r),flush=True)
    selection=json.loads((a.cache/'selection.json').read_text())
    selected={s:sorted(selection[s],key=lambda i:stable_seed('S027-images',s,i))[:80] for s in ['fit','transfer']}
    hashes={k.replace('\\','/'):v for k,v in json.loads((a.cache/'COMPLETE.json').read_text())['hashes'].items()}
    pairs={s:[] for s in selected}
    for split in selected:
        for iid in selected[split]:
            with np.load(a.cache/'images'/f'{iid}.npz') as q:pairs[split].extend((iid,int(aid)) for aid in q['annotation_ids'])
    oracle={s:set(sorted(pairs[s],key=lambda q:stable_seed('S027-oracle',s,*q))[:80]) for s in selected}
    write_json(out/'protocol.json',dict(experiment='S027 saved readout fitting and sampled-pixel transfer',
       selection=selected,oracle_targets={s:sorted(v) for s,v in oracle.items()},
       selection_rule='80imagehash each from previous1200fit/300transfer; no density/failure selection. '
                      'All fixedbbox50 matched targets in those images evaluated;80targethash per split gets oracle.',
       saved_models='S026 finalepoch15 global/scalar/rich_neighbor, all3seeds, no new shared training or selection.',
       pixel_train='Same first512 cached uniform with-replacement native640 GTbox sample positions and labels used by S026.',
       pixel_holdout='Unique positions from remaining1536 samples excluding ANY first512 position; per target variableN. '
                     'Conditional remainder of sameGT support, not new images; empty sets explicit. '
                     'Dice smoothing/sample size differs across domains, compare models WITHIN each domain.',
       oracle='Per-target independent32Dcoefficient LBFGS on exactlysame512 BCE+Dice, FP64 scaled,120iterations, '
              'original init,no regularization,bias or IoUstate selection. Same labels/objective; not deployable/AP/global optimum.',
       final_decode='Original predicted crop unchanged, rawCOCO maskIoU and valid own/neighbor/background diagnostics. '
                    'Official normal float32 decoder for saved original/global; FP64 prototype-logit interpolation for oracle to avoid extreme-scale signs.',
       scope='Existing exploredtrain2017, fixedbbox matched cohort; diagnostic not full taskAP/Recall,not historical user-training blame.',
       sources=dict(cache_receipt=sha(a.cache/'COMPLETE.json'),trained_receipt=sha(a.trained/'COMPLETE.json'),script=sha(Path(__file__)))))
    shutil.copy2(__file__,out/Path(__file__).name)
    subset=a.cache/'conversion_input/instances_probe.json'
    if sha(subset)!=hashes['conversion_input/instances_probe.json']:raise RuntimeError('RawJSON subset changed')
    obj=json.loads(subset.read_text());wanted=set(selected['fit']+selected['transfer'])
    gt=COCO();gt.dataset=dict(images=[r for r in obj['images'] if r['id'] in wanted],categories=obj['categories'],
                             annotations=[r for r in obj['annotations'] if r['image_id'] in wanted])
    with contextlib.redirect_stdout(io.StringIO()):gt.createIndex()
    del obj
    n0=torch.load(a.trained/'normalizer.pt',weights_only=True)
    normalization={k:{kk:vv.cuda() for kk,vv in v.items()} for k,v in n0.items()}
    models=[]
    for seed in [0,1,2]:
        for mode in ['global','scalar','rich_neighbor']:
            model=(GlobalHead() if mode=='global' else PixelHead()).cuda()
            checkpoint=a.trained/f'{mode}_s{seed}/checkpoints/epoch015.pt'
            rec=json.loads((a.trained/f'{mode}_s{seed}/COMPLETE.json').read_text())
            if sha(checkpoint)!=rec['final_sha256']:raise RuntimeError('Trained weights changed')
            model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model']);models.append((mode,seed,model.eval()))
    pixrows=[];spatial=[];solver=[];targetrows=[]
    for split in ['fit','transfer']:
        for number,iid in enumerate(selected[split],1):
            path=a.cache/'images'/f'{iid}.npz'
            if sha(path)!=hashes[f'images/{iid}.npz']:raise RuntimeError('Cache changed')
            item=read_np(path);tids=item['annotation_ids'];idx=item['prediction_indices']
            if not len(tids):continue
            cc=torch.tensor(item['coeff'],device='cuda');p=torch.tensor(item['proto'],device='cuda')
            boxes=torch.tensor(item['boxes'],device='cuda');det=torch.tensor(item['detections'],device='cuda')
            h=torch.tensor(item['h'],device='cuda');lv=torch.tensor(item['level'],device='cuda').long()
            shape=tuple(item['shape']);ishape=tuple(item['input_shape']);ih,iw=ishape
            with torch.inference_mode():
                x=norm(instance_features(h,lv,boxes,ishape),normalization['x']);hn=norm(h,normalization['h'])
                p640=F.interpolate(p[None],ishape,mode='bilinear',align_corners=False)[0].flatten(1).T
                raw=F.interpolate((cc@p.flatten(1)).reshape(1,len(cc),*p.shape[-2:]),ishape,mode='bilinear')[0]
                globals_c={s:cc+model(x) for mode,s,model in models if mode=='global'}
                global_masks={s:ops.process_mask(p,c,boxes,ishape,upsample=True) for s,c in globals_c.items()}
                original=ops.process_mask(p,cc,boxes,ishape,upsample=True)
            masks={b['id']:gt.annToMask(b).astype(bool) for b in gt.imgToAnns[iid]};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
            ordinary=[b for b in gt.imgToAnns[iid] if not b.get('iscrowd',0)]
            for b in gt.imgToAnns[iid]:
                if b.get('iscrowd',0):crowd|=masks[b['id']]
                else:union|=masks[b['id']]
            for k,(aid0,j0) in enumerate(zip(tids,idx)):
                aid=int(aid0);j=int(j0);aa=gt.anns[aid];density=ici(aa,ordinary)
                base=dict(split=split,image_id=iid,annotation_id=aid,ici=density,high=density>.5+1e-10,oracle_selected=(iid,aid) in oracle[split])
                pos=item['sample_positions'][k];seen=set(map(int,pos[:512]));held=[]
                for t in range(512,len(pos)):
                    v=int(pos[t])
                    if v not in seen:held.append(t);seen.add(v)
                targetrows.append(dict(**base,train_unique=len(set(map(int,pos[:512]))),holdout_unique=len(held),
                                       native_positive512=int(item['sample_y'][k,:512].sum())))
                samplep=torch.tensor(item['sample_p'][k],device='cuda');y=torch.tensor(item['sample_y'][k],device='cuda').float()
                factor=float(item['loss_factor'][k]);pts=torch.tensor(np.stack([pos%iw,pos//iw],1),device='cuda').float()
                with torch.inference_mode():
                    feature=build_features(cc@samplep.T,boxes,det,j,pts,'neighbor')
                    pn=norm(samplep,normalization['p']);z0=samplep@cc[j]
                    zs={('original',-1):z0}
                    for mode,seed,model in models:
                        zs[mode,seed]=(samplep@globals_c[seed][j] if mode=='global' else
                              z0+model(feature[None],pn[None],hn[j:j+1],mode)[0])
                oracle_c=None
                if base['oracle_selected']:
                    oracle_c,info=solve(samplep[:512],y[:512],cc[j],factor)
                    solver.append(dict(**base,**info));zs['oracle',-1]=samplep.double()@oracle_c
                    np.savez_compressed(out/'oracle_coefficients'/f'{iid}_{aid}.npz',coefficient=oracle_c.cpu().numpy())
                for (mode,seed),z in zs.items():
                    for domain,positions in [('train512',slice(0,512)),('unused',held)]:
                        vals=numbers(z[positions],y[positions].to(z.dtype),factor)
                        pixrows.append(dict(**base,mode=mode,seed=seed,domain=domain,**vals))
                # Full fixed-crop masks for the same instance; not an allGT task evaluation.
                with torch.inference_mode():
                    bb=boxes[j];x1=max(0,math.ceil(float(bb[0])));x2=max(x1,min(iw,math.ceil(float(bb[2]))))
                    y1=max(0,math.ceil(float(bb[1])));y2=max(y1,min(ih,math.ceil(float(bb[3]))))
                    yy,xx=torch.meshgrid(torch.arange(y1,y2,device='cuda'),torch.arange(x1,x2,device='cuda'),indexing='ij')
                    flat=(yy*iw+xx).flatten();points=torch.stack([xx.flatten(),yy.flatten()],1).float()
                    sf=build_features(raw.flatten(1)[:,flat],boxes,det,j,points,'neighbor');pp=norm(p640[flat],normalization['p'])
                    binary={('original',-1):original[j],**{('global',s):v[j] for s,v in global_masks.items()}}
                    for mode,seed,model in models:
                        if mode=='global':continue
                        val=torch.zeros(ih*iw,device='cuda',dtype=torch.uint8)
                        for f in range(0,len(flat),16384):
                            ids=flat[f:f+16384]
                            residual=model(sf[f:f+16384][None],pp[f:f+16384][None],hn[j:j+1],mode)[0]
                            val[ids]=(raw[j].flatten()[ids]+residual>0).byte()
                        binary[mode,seed]=val.reshape(ih,iw)
                    if oracle_c is not None:
                        lo=(oracle_c@p.double().flatten(1)).reshape(1,1,*p.shape[-2:])
                        z=F.interpolate(lo,ishape,mode='bilinear',align_corners=False)[0,0]
                        binary['oracle',-1]=ops.crop_mask((z>0).byte()[None],boxes[j:j+1])[0]
                    keys=list(binary);restored=ops.scale_masks(torch.stack(list(binary.values()))[:,None],shape)[:,0]>.5
                    own=masks[aid]&~crowd;area=int(own.sum());same=np.zeros(shape,bool)
                    for b in ordinary:
                        if b['id']!=aid and b['category_id']==aa['category_id']:same|=masks[b['id']]
                    for kk,(mode,seed) in enumerate(keys):
                        pred=restored[kk].cpu().numpy();vp=pred&~crowd
                        inter=int((pred&masks[aid]).sum());uni=int((pred|masks[aid]).sum())
                        spatial.append(dict(**base,mode=mode,seed=seed,coco_iou=inter/uni if uni else 1.,
                            coverage=float((vp&own).sum()/area) if area else None,
                            neighbor=float((vp&same&~own).sum()/area) if area else None,
                            background=float((vp&~union).sum()/area) if area else None))
            if number%10==0:progress('diagnose',split=split,images=number,total=len(selected[split]))
    for name,rows in [('pixels',pixrows),('spatial',spatial),('solver',solver),('targets',targetrows)]:csv_save(out/f'{name}.csv',rows)
    progress('COMPLETE',targets=len(targetrows),oracle_solves=len(solver))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,targets=len(targetrows),
       oracle_solves=len(solver),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--out' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--out')+1])
            if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
