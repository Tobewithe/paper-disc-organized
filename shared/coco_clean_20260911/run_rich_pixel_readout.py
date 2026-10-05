"""Frozen COCO readout study on local GPU, with task metrics and three seeds."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,contextlib,csv,gzip,io,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from pixel_ownership_refiner import build_features
from rich_pixel_readout import MODES,GlobalHead,PixelHead,instance_features,loss_value
from readout_input_probe import sha,write_json,stable_seed
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate


def read_np(path):
    with np.load(path) as q:return {k:q[k] for k in q.files}


def csv_save(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def cuda(a):return torch.as_tensor(a,device='cuda')


def norms(x):return dict(mean=x.mean(0),std=x.std(0,unbiased=False).clamp_min(.01))


def norm(x,n):return (x-n['mean'])/n['std']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);args=ap.parse_args()
    out=args.run;out.mkdir(exist_ok=False);src=args.source;cache=src/'cache';start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=dict(experiment='S026 rich pixel readout incremental test',status='PRE_EXECUTION',
      cache=str(cache.resolve()),source_cache_receipt_sha256=sha(cache/'COMPLETE.json'),
      images='Existing1200fit+300transfer train2017, all image/GT IDs unchanged; no density selection. '
             'Transfer already explored; not fresh benchmark or pristine pretrained holdout.',
      seeds=[0,1,2],modes=MODES,epochs=15,batch_targets=32,pixels=512,lr=1e-4,optimizer='Adam',
      pixels_rule='First512 of existing2048 uniformly sampled native640 GT-box/area BCE pixels. Same all arms. '
                  'Sampled BCE+Dice is a common objective, not full native trainer replication.',
      freeze='All original weights, prototypes, boxes, scores, labels, kept candidates, fixedbbox50 attribution. '
             'Only readout heads trained. Predicted box final crop never expanded.',
      global_control='Same73Dinstance h+level+box geometry and128-wide architecture as previous strong global head; '
                     'new local cache/native support and common BCE+Dice, scores not compared directly to old remote runs.',
      scalar_control='Historical14Dscalar own logit/coordinate/neighbor features in same64-wide rich pixel architecture. '
                     'Extra P/h slots zero, parameter count same but effective capacity differs.',
      rich='Original8own scalar features plus32DP(x),64Dh; six neighbor scalar slots zero.',
      rich_neighbor='Full14scalar features plus32DP(x),64Dh. Predictions only; neighbor score>=.1 boxIoU>.05 sameclass.',
      rich_shuffled='Same as rich_neighbor but P(x) vectors shuffled across sampled positions independently '
         'per target/epoch, preserving h and all scalar features. Transfer:3 fixed within-prediction-box pixel permutations, '
         'report separately then mean metrics, no mask ensemble. This control tests spatially aligned P, not instance-neighbor causality.',
      sampling_scope='Train uniform nativeGT-box samples; inference all pixels inside original predicted crop. '
                     'No GT labels or identity determine inference features/selection. Pixels outside crop stay zero.',
      stopping='Finalepoch15 fixed. No tuning on transfer or testGT. Require rich inputs outperform scalar and global '
               'for highR75 and AP without harming other groups; otherwise stop version. No formal full training.',
      statistics='Exploratory image-clusterpaired2000 intervals, seed metrics first mean; all ordinaryGT official matching beforeICI. '
                 'Not matched-precisionRecall; if positive, precision and area/category controls needed before method claim.')
    write_json(out/'protocol.json',protocol)
    (out/'source').mkdir()
    for name in ['run_rich_pixel_readout.py','rich_pixel_readout.py','pixel_ownership_refiner.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    def progress(stage,**kw):
        row=dict(stage=stage,seconds=time.monotonic()-start,**kw)
        write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    selection=json.loads((cache/'selection.json').read_text());fit=selection['fit'];transfer=selection['transfer']
    if set(fit)&set(transfer):raise RuntimeError('Fit/transfer overlap')
    receipt=json.loads((cache/'COMPLETE.json').read_text())['hashes']
    receipt={k.replace('\\','/'):v for k,v in receipt.items()}
    records=[];identities=[];verified=0
    progress('build_fit')
    for ix,iid in enumerate(fit):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=receipt[f'images/{iid}.npz']:raise RuntimeError('Source cache changed')
        verified+=1;item=read_np(path);idx=item['prediction_indices']
        if not len(idx):continue
        h=cuda(item['h']).float();p=cuda(item['sample_p'][:,:512]).float();c=cuda(item['coeff']).float()
        boxes=cuda(item['boxes']).float();det=cuda(item['detections']).float();level=cuda(item['level']).long()
        scalar=[]
        for k,j in enumerate(idx):
            pos=cuda(item['sample_positions'][k,:512]).long()
            pts=torch.stack([pos%640,torch.div(pos,640,rounding_mode='floor')],1).float()
            z=c@p[k].T
            scalar.append(build_features(z,boxes,det,int(j),pts,'neighbor'))
        x=instance_features(h,level,boxes,tuple(item['input_shape']))
        records.append(dict(h=h[idx].cpu(),x=x[idx].cpu(),p=p.cpu(),
            scalar=torch.stack(scalar).cpu(),c=c[idx].cpu(),y=torch.tensor(item['sample_y'][:,:512]).float(),
            factor=torch.tensor(item['loss_factor']).float()))
        identities.extend((iid,int(item['source_index'][j])) for j in idx)
        if (ix+1)%100==0:progress('build_fit',images=ix+1,total=len(fit))
    data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]};del records
    normalization={'h':norms(data['h']),'x':norms(data['x']),'p':norms(data['p'].reshape(-1,32))}
    torch.save({k:{kk:vv.cpu() for kk,vv in v.items()} for k,v in normalization.items()},out/'normalizer.pt')
    data['hn']=norm(data['h'],normalization['h']);data['xn']=norm(data['x'],normalization['x'])
    data['pn']=norm(data['p'],normalization['p'])
    data['z']=(data['p']*data['c'][:,None]).sum(-1)
    count=len(identities);epochs=protocol['epochs'];batch=protocol['batch_targets'];trained=[];history=[]
    write_json(out/'fit_metadata.json',dict(targets=count,identities=identities,verified_source_images=verified))
    with torch.no_grad():
        witness=PixelHead().cuda();w=witness(data['scalar'][:2],data['pn'][:2],data['hn'][:2],'rich_neighbor')
        if torch.count_nonzero(w):raise RuntimeError('Nonzero initial residual')
        # Check loss has useful output gradients on real labels; no fake expected result.
    for seed in protocol['seeds']:
        for mode in MODES:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=(GlobalHead() if mode=='global' else PixelHead()).cuda()
            opt=torch.optim.Adam(model.parameters(),lr=protocol['lr']);rng=np.random.default_rng(seed)
            run=out/f'{mode}_s{seed}';(run/'checkpoints').mkdir(parents=True)
            initial_sha={k:sha(out/'normalizer.pt') for k in ['normalizer']}
            updates=0
            for epoch in range(epochs):
                order=rng.permutation(count);total=0.
                for first in range(0,count,batch):
                    ids=cuda(order[first:first+batch]).long();y=data['y'][ids]
                    if mode=='global':z=(data['p'][ids]*(data['c'][ids]+model(data['xn'][ids]))[:,None]).sum(-1)
                    else:
                        pp=data['pn'][ids]
                        if mode=='rich_shuffled':
                            # Stateless permutations independent of GT; same sample identity and epoch.
                            orders=np.stack([np.random.default_rng(stable_seed('rich-shuffle',seed,epoch,*identities[t])).permutation(512)
                                             for t in order[first:first+batch]])
                            pp=pp.gather(1,cuda(orders).long()[...,None].expand(-1,-1,32))
                        z=data['z'][ids]+model(data['scalar'][ids],pp,data['hn'][ids],mode)
                    loss=loss_value(z,y,data['factor'][ids])
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                    total+=float(loss.detach())*len(ids);updates+=1
                row=dict(mode=mode,seed=seed,epoch=epoch+1,updates=updates,loss=total/count)
                history.append(row)
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,updates=updates,
                     mode=mode,seed=seed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state),
                     run/'checkpoints'/f'epoch{epoch+1:03d}.pt')
                if (epoch+1)%5==0:progress('train',**row)
            if any(not torch.isfinite(q).all() for q in model.parameters()):raise RuntimeError('Nonfinite weights')
            trained.append((mode,seed,model.eval()))
            write_json(run/'COMPLETE.json',dict(status='COMPLETE',updates=updates,parameters=sum(p.numel() for p in model.parameters()),
                 final_sha256=sha(run/'checkpoints/epoch015.pt'),checkpoints=15))
            write_json(out/'history.json',history)
    del data;torch.cuda.empty_cache()
    progress('training_complete',runs=len(trained),checkpoints=len(trained)*epochs)
    # Build a COCO object from the hash-verified raw annotation subset, retaining all GT.
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=receipt['conversion_input/instances_probe.json']:raise RuntimeError('Annotation subset changed')
    obj=json.loads(subset.read_text());gt=COCO();gt.dataset=dict(info=obj.get('info',{}),categories=obj['categories'],
       images=[im for im in obj['images'] if im['id'] in transfer],annotations=[a for a in obj['annotations'] if a['image_id'] in transfer])
    with contextlib.redirect_stdout(io.StringIO()):gt.createIndex()
    del obj
    meta={a['id']:{'ici_same':ici(a,[b for b in gt.imgToAnns[a['image_id']] if not b.get('iscrowd',0)])}
           for a in gt.anns.values() if not a.get('iscrowd',0)}
    arms=[('original',-1,0,None)]+[(m,s,d,model) for m,s,model in trained for d in (range(3) if m=='rich_shuffled' else [0])]
    predictions={f'{m}_s{s}_d{d}':[] for m,s,d,_ in arms};spatial=[];categories=sorted(gt.cats);xor=0
    eval_start=time.monotonic()
    for number,iid in enumerate(transfer,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=receipt[f'images/{iid}.npz']:raise RuntimeError('Transfer source changed')
        item=read_np(path);c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
        det=cuda(item['detections']).float();h=cuda(item['h']).float();level=cuda(item['level']).long()
        shape=tuple(item['shape']);ishape=tuple(item['input_shape']);ih,iw=ishape
        xn=norm(instance_features(h,level,boxes,ishape),normalization['x']);hn=norm(h,normalization['h'])
        fullp=F.interpolate(p[None],ishape,mode='bilinear',align_corners=False)[0].flatten(1).T
        tid_by_pred={int(j):int(a) for a,j in zip(item['annotation_ids'],item['prediction_indices'])}
        masks={a['id']:gt.annToMask(a).astype(bool) for a in gt.imgToAnns[iid]};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        for a in gt.imgToAnns[iid]:
            if a.get('iscrowd',0):crowd|=masks[a['id']]
            else:union|=masks[a['id']]
        # Build original logits in the stock order; learned global controls also use this exact decoder.
        with torch.inference_mode():
            raw=F.interpolate((c@p.flatten(1)).reshape(1,len(c),*p.shape[-2:]),ishape,mode='bilinear')[0] if len(c) else c.new_empty((0,ih,iw))
            original=ops.process_mask(p,c,boxes,ishape,upsample=True)
            replay=ops.crop_mask((raw>0).byte(),boxes)
            diff=int(torch.count_nonzero(replay!=original));xor+=diff
            if diff:raise RuntimeError('Original mask decoder mismatch')
            global_masks={seed:ops.process_mask(p,c+model(xn),boxes,ishape,upsample=True)
                          for mode,seed,model in trained if mode=='global'}
            for j in range(len(c)):
                box=boxes[j];x1=max(0,math.ceil(float(box[0])));x2=min(iw,math.ceil(float(box[2])))
                y1=max(0,math.ceil(float(box[1])));y2=min(ih,math.ceil(float(box[3])))
                yy,xx=torch.meshgrid(torch.arange(y1,max(y1,y2),device='cuda'),torch.arange(x1,max(x1,x2),device='cuda'),indexing='ij')
                positions=(yy*iw+xx).flatten();points=torch.stack([xx.flatten(),yy.flatten()],1).float()
                pp=fullp[positions];pnormal=norm(pp,normalization['p'])
                # Exact original-order neighboring responses; no ground truth here.
                scalar=build_features(raw.flatten(1)[:,positions],boxes,det,j,points,'neighbor')
                binaries={('original',-1,0):original[j],**{('global',s,0):v[j] for s,v in global_masks.items()}}
                for mode,seed,draw,model in arms:
                    if mode in ['original','global']:continue
                    perm=None
                    if mode=='rich_shuffled':
                        perm=cuda(np.random.default_rng(stable_seed('rich-shuffle-eval',seed,draw,iid,int(item['source_index'][j]))).permutation(len(positions))).long()
                    value=torch.zeros(ih*iw,device='cuda',dtype=torch.uint8)
                    for first in range(0,len(positions),16384):
                        stop=first+16384;pn=pnormal[first:stop] if perm is None else pnormal[perm[first:stop]]
                        residual=model(scalar[first:stop][None],pn[None],hn[j:j+1],mode)[0]
                        value[positions[first:stop]]=(raw[j].flatten()[positions[first:stop]]+residual>0).byte()
                    binaries[mode,seed,draw]=value.reshape(ih,iw)
                ordered=[binaries[m,s,d] for m,s,d,_ in arms]
                restored=ops.scale_masks(torch.stack(ordered)[:,None],shape)[:,0]>.5
                # Decode every prediction before using fixed GT attribution for spatial diagnostics.
                for k,(mode,seed,draw,_) in enumerate(arms):
                    arm=f'{mode}_s{seed}_d{draw}';pm=restored[k].cpu().numpy()
                    if bool(ordered[k].any()):
                        rle=mu.encode(np.asfortranarray(pm.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
                        predictions[arm].append(dict(image_id=iid,category_id=categories[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
                    if j in tid_by_pred:
                        aid=tid_by_pred[j];aa=gt.anns[aid];own=masks[aid]&~crowd;area=int(own.sum());vp=pm&~crowd
                        same=np.zeros(shape,bool)
                        for b in gt.imgToAnns[iid]:
                            if not b.get('iscrowd',0) and b['id']!=aid and b['category_id']==aa['category_id']:same|=masks[b['id']]
                        inter=int((pm&masks[aid]).sum());un=int((pm|masks[aid]).sum())
                        spatial.append(dict(arm=arm,image_id=iid,annotation_id=aid,ici=meta[aid]['ici_same'],
                           iou=inter/un if un else 0.,coverage=float((vp&own).sum()/area) if area else None,
                           neighbor=float((vp&same&~own).sum()/area) if area else None,
                           background=float((vp&~union).sum()/area) if area else None))
        if number%10==0 or number==len(transfer):
            progress('decode',images=number,total=len(transfer),decode_seconds=time.monotonic()-eval_start)
    (out/'predictions').mkdir();task=[];gtrows=[];pairs=[]
    for arm,pp in predictions.items():
        with gzip.open(out/'predictions'/f'{arm}.json.gz','wt',encoding='utf-8') as f:json.dump(pp,f,separators=(',',':'))
        if not pp:raise RuntimeError('All-empty arm requires explicit evaluator support')
        row,rr,pr=evaluate(gt,meta,transfer,pp,arm)
        if len(rr)!=len(meta):raise RuntimeError('Task denominator changed')
        row['gap']=row['r75_low']-row['r75_high'];task.append(row);gtrows.extend(rr);pairs.extend(pr)
        progress('official_task',**row)
    csv_save(out/'task_summary.csv',task);csv_save(out/'gt_recovery.csv',gtrows);csv_save(out/'pair_recovery.csv',pairs);csv_save(out/'spatial.csv',spatial)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,fit_targets=count,
       transfer_images=len(transfer),ordinary_gt=len(meta),checkpoints=15*epochs,decoder_original_pixel_xor=xor,
       hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'checkpoints' not in str(p)}))
    progress('COMPLETE',runs=len(trained))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--run' in sys.argv:
            dest=Path(sys.argv[sys.argv.index('--run')+1])
            if dest.is_dir():write_json(dest/'FAILED.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()))
        raise
