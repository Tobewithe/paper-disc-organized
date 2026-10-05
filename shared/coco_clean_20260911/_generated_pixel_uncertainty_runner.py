"""S032: same frozen global head and budget, three label representations x3seeds.

No model/threshold selection, original predicted crop and scores retained.
Existing-label controls, not a claim of a novel method.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,gzip,io,json,shutil,time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from native_label_pipeline_probe import format_native,mask_of
from readout_input_probe import sha,write_json
from rich_pixel_readout import GlobalHead,instance_features,loss_value
from train_pixel_uncertainty_readout import pixel_uncertainty_loss
from run_rich_pixel_readout import read_np,csv_save,cuda,norm,norms
from eval_readout_input_pilot import ici
from summarize_relative_ownership import evaluate

MODES=['raw_coco','pixel_uncertainty']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False);(out/'source').mkdir();(out/'labels').mkdir();cache=a.cache;start=time.monotonic()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=dict(experiment='S032_SHARED_LABEL_CONTROLS',cache=str(cache.resolve()),modes=MODES,seeds=[0,1,2],
        fit_images=1200,transfer_images=300,epochs=15,batch_targets=32,pixels=512,lr=1e-4,optimizer='Adam',
        method='S066 same frozen global73D coefficient head; raw COCO control versus pixelwise frozen-logit uncertainty weighting.',
        intervention='Only training binaryY varies: nativeoverlap fromcache, officialnative independent Format, rawCOCO nearest-exact640. '
        'All targets/first512positions/prototypes/factors/init/order/optimizer/finalepoch fixed.',
        freeze='Alloriginalmodelweights,features,P,boxes,scores,candidateidentity,fixedbbox50trainingattribution. '
        'Onlysmallglobalheadtrained;GTneverinpredictioninputs;nocropperturbation.',
        evaluation='All300existingtransfertrain2017images/allordinaryGT originalCOCOeval matching thenICI. '
        'Previously explored subset, not pristineval or historicaltrainingreplay. No modelselection.',
        stop='Report all3seeds allarms. Labelswitches areexisting controls,not innovation. '
        'Only proceed towardmethod if highR75 rises, AP maintained andhigh-nonhighgap narrows; no choosing bestseed orlambda.',
        source_hashes=dict(cache_receipt=sha(cache/'COMPLETE.json'),prior_receipt=sha(a.prior/'COMPLETE.json'),script=sha(__file__)))
    write_json(out/'protocol.json',protocol)
    for name in ['train_readout_label_controls.py','rich_pixel_readout.py','native_label_pipeline_probe.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    def progress(stage,**kw):
        row=dict(stage=stage,seconds=time.monotonic()-start,**kw);write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
    selection=json.loads((cache/'selection.json').read_text());fit=selection['fit'];transfer=selection['transfer']
    if len(fit)!=1200 or len(transfer)!=300 or set(fit)&set(transfer):raise RuntimeError('Cohort changed')
    receipt={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=receipt['conversion_input/instances_probe.json']:raise RuntimeError('Changed annotations')
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(subset))
    cfg=get_cfg(overrides=dict(task='segment',imgsz=640,mask_ratio=1,overlap_mask=True,rect=False,cache=False,workers=0,fraction=1.0))
    dd=dict(names={k:gt.cats[c]['name'] for k,c in enumerate(sorted(gt.cats))},nc=80,channels=3)
    with contextlib.redirect_stdout(io.StringIO()):ds=build_yolo_dataset(cfg,str(cache/'converted/images/probe'),1,dd,mode='val',rect=False)
    index={int(Path(l['im_file']).stem):k for k,l in enumerate(ds.labels)}
    records=[];identities=[];labelstats=[];progress('build_labels')
    for num,iid in enumerate(fit,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=receipt[f'images/{iid}.npz']:raise RuntimeError('Changed sourceNPZ')
        item=read_np(path);idx=item['prediction_indices']
        if not len(idx):continue
        imgmeta=json.loads((cache/'images'/f'{iid}.json').read_text());lab=ds.labels[index[iid]]
        labelpath=cache/'converted/labels/probe'/f'{iid:012d}.txt'
        if sha(labelpath)!=imgmeta['label_sha256']:raise RuntimeError('Changed sourceTXT')
        raw=deepcopy(lab);raw.pop('shape',None);raw['img']=cv2.imread(lab['im_file']);h,w=raw['img'].shape[:2]
        if (h,w)!=tuple(item['shape']):raise RuntimeError('Geometry changed')
        raw=ds.update_labels_info(raw);raw['ori_shape']=(h,w);raw['ratio_pad']=(1.,1.)
        ishape=tuple(map(int,item['input_shape']));raw=LetterBox(new_shape=ishape,auto=False,scaleup=True)(raw)
        independent=format_native(raw,imgmeta['native_ids'],1,False)
        gain=min(ishape[0]/h,ishape[1]/w);rh,rw=round(h*gain),round(w*gain)
        top=round((ishape[0]-rh)/2-.1);left=round((ishape[1]-rw)/2-.1)
        yi=[];yc=[]
        for k,aid in enumerate(item['annotation_ids']):
            aid=int(aid);pos=item['sample_positions'][k,:512]
            ind=mask_of(independent,aid);yp=ind.flatten()[pos];yi.append(yp)
            mask=torch.tensor(gt.annToMask(gt.anns[aid]),device='cuda').float()
            yy=torch.zeros(ishape,device='cuda',dtype=torch.bool)
            yy[top:top+rh,left:left+rw]=F.interpolate(mask[None,None],(rh,rw),mode='nearest-exact')[0,0].bool()
            yraw=yy.flatten()[cuda(pos).long()].cpu().numpy().astype(np.uint8);yc.append(yraw)
            labelstats.append(dict(image_id=iid,annotation_id=aid,native_positive=int(item['sample_y'][k,:512].sum()),
                independent_positive=int(yp.sum()),raw_positive=int(yraw.sum()),
                overlap_independent_xor=int(np.count_nonzero(item['sample_y'][k,:512]!=yp)),
                independent_raw_xor=int(np.count_nonzero(yp!=yraw))))
        yi=np.stack(yi);yc=np.stack(yc)
        np.savez_compressed(out/'labels'/f'{iid}.npz',annotation_ids=item['annotation_ids'],native_independent=yi,raw_coco=yc)
        hfeat=cuda(item['h']).float();lv=cuda(item['level']).long();boxes=cuda(item['boxes']).float()
        x=instance_features(hfeat,lv,boxes,ishape)[idx]
        records.append(dict(x=x.cpu(),p=torch.tensor(item['sample_p'][:,:512]).float(),
            c=torch.tensor(item['coeff'][idx]).float(),factor=torch.tensor(item['loss_factor']).float(),
            native_overlap=torch.tensor(item['sample_y'][:,:512]).float(),
            native_independent=torch.tensor(yi).float(),raw_coco=torch.tensor(yc).float()))
        identities.extend((iid,int(item['source_index'][j])) for j in idx)
        if num%100==0:progress('build_labels',images=num,total=1200)
    data={k:torch.cat([r[k] for r in records]).cuda() for k in records[0]}
    data['pixel_uncertainty']=data['raw_coco'];del records,ds
    normalization=norms(data['x']);torch.save({k:v.cpu() for k,v in normalization.items()},out/'normalizer.pt')
    prior_norm=torch.load(a.prior/'normalizer.pt',map_location='cuda',weights_only=True)
    if any(not torch.equal(normalization[k],prior_norm[k]) for k in normalization):raise RuntimeError('S026input normalization replay differed')
    data['xn']=norm(data['x'],normalization);count=len(identities)
    if count!=7811:raise RuntimeError('Fit targets changed')
    write_json(out/'fit_metadata.json',dict(targets=count,identities=identities));csv_save(out/'label_stats.csv',labelstats)
    trained=[];history=[];replay=[]
    for seed in [0,1,2]:
        for mode in MODES:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=GlobalHead().cuda()
            opt=torch.optim.Adam(model.parameters(),lr=1e-4);rng=np.random.default_rng(seed)
            dest=out/f'{mode}_s{seed}';(dest/'checkpoints').mkdir(parents=True);updates=0
            for epoch in range(15):
                order=rng.permutation(count);total=0.
                for first in range(0,count,32):
                    ix=cuda(order[first:first+32]).long()
                    z=(data['p'][ix]*(data['c'][ix]+model(data['xn'][ix]))[:,None]).sum(-1)
                    z0=(data['p'][ix]*data['c'][ix][:,None]).sum(-1)
                    loss=(pixel_uncertainty_loss(z,data['pixel_uncertainty'][ix],z0,data['factor'][ix]) if mode=='pixel_uncertainty' else loss_value(z,data[mode][ix],data['factor'][ix]))
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                    total+=float(loss.detach())*len(ix);updates+=1
                row=dict(mode=mode,seed=seed,epoch=epoch+1,updates=updates,loss=total/count);history.append(row)
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,updates=updates,
                    mode=mode,seed=seed,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state),
                    dest/'checkpoints'/f'epoch{epoch+1:03d}.pt')
                if (epoch+1)%5==0:progress('train',**row)
            trained.append((mode,seed,model.eval()))
            write_json(dest/'COMPLETE.json',dict(status='COMPLETE',checkpoints=15,updates=updates,final_sha256=sha(dest/'checkpoints/epoch015.pt')))
            write_json(out/'history.json',history)
    write_json(out/'prior_replay.json',replay);del data;torch.cuda.empty_cache();progress('training_complete',checkpoints=135)
    gt_eval=COCO();gt_eval.dataset=dict(info=gt.dataset.get('info',{}),categories=list(gt.cats.values()),
        images=[gt.imgs[i] for i in transfer],annotations=[ann for i in transfer for ann in gt.imgToAnns[i]])
    with contextlib.redirect_stdout(io.StringIO()):gt_eval.createIndex()
    meta={ann['id']:{'ici_same':ici(ann,[b for b in gt_eval.imgToAnns[ann['image_id']] if not b.get('iscrowd',0)])}
          for ann in gt_eval.anns.values() if not ann.get('iscrowd',0)}
    categories=sorted(gt_eval.cats);arms=[('original',-1,None)]+trained;predictions={f'{m}_s{s}_d0':[] for m,s,_ in arms};spatial=[]
    for num,iid in enumerate(transfer,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=receipt[f'images/{iid}.npz']:raise RuntimeError('Changed evaluation NPZ')
        item=read_np(path);c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
        det=cuda(item['detections']).float();h=cuda(item['h']).float();level=cuda(item['level']).long()
        shape=tuple(item['shape']);ishape=tuple(item['input_shape']);mapping={int(j):int(t) for t,j in zip(item['annotation_ids'],item['prediction_indices'])}
        masks={ann['id']:gt_eval.annToMask(ann).astype(bool) for ann in gt_eval.imgToAnns[iid]};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        for ann in gt_eval.imgToAnns[iid]:
            if ann.get('iscrowd',0):crowd|=masks[ann['id']]
            else:union|=masks[ann['id']]
        with torch.inference_mode():
            x=norm(instance_features(h,level,boxes,ishape),normalization)
            for mode,seed,model in arms:
                name=f'{mode}_s{seed}_d0';coeff=c if model is None else c+model(x)
                bb=ops.process_mask(p,coeff,boxes,ishape,upsample=True)
                pm=ops.scale_masks(bb[:,None],shape)[:,0]>.5 if len(c) else bb
                for j in range(len(c)):
                    pred=pm[j].cpu().numpy()
                    if bool(bb[j].any()):
                        rle=mu.encode(np.asfortranarray(pred.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
                        predictions[name].append(dict(image_id=iid,category_id=categories[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
                    if j in mapping:
                        aid=mapping[j];ann=gt_eval.anns[aid];own=masks[aid]&~crowd;area=int(own.sum());vp=pred&~crowd;same=np.zeros(shape,bool)
                        for other in gt_eval.imgToAnns[iid]:
                            if not other.get('iscrowd',0) and other['id']!=aid and other['category_id']==ann['category_id']:same|=masks[other['id']]
                        un=int((pred|masks[aid]).sum())
                        spatial.append(dict(arm=name,image_id=iid,annotation_id=aid,ici=meta[aid]['ici_same'],
                            iou=int((pred&masks[aid]).sum())/un if un else 1.,
                            coverage=float((vp&own).sum()/area) if area else None,
                            neighbor=float((vp&same&~own).sum()/area) if area else None,
                            background=float((vp&~union).sum()/area) if area else None))
        if num%30==0:progress('decode',images=num,total=300)
    (out/'predictions').mkdir();task=[];gtrows=[];pairs=[]
    for name,pp in predictions.items():
        with gzip.open(out/'predictions'/f'{name}.json.gz','wt',encoding='utf-8') as f:json.dump(pp,f,separators=(',',':'))
        if not pp:raise RuntimeError('All-empty model requires explicit evaluation handling')
        row,rr,pr=evaluate(gt_eval,meta,transfer,pp,name)
        if len(rr)!=len(meta):raise RuntimeError('AllGTdenominator changed')
        row['gap']=row['r75_low']-row['r75_high'];task.append(row);gtrows.extend(rr);pairs.extend(pr);progress('task',**row)
    for name,value in [('task_summary',task),('gt_recovery',gtrows),('pair_recovery',pairs),('spatial',spatial)]:csv_save(out/f'{name}.csv',value)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start,fit_targets=count,ordinary_gt=len(meta),checkpoints=135,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'checkpoints' not in str(p)}))
    progress('COMPLETE')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--out' in sys.argv:
            dest=Path(sys.argv[sys.argv.index('--out')+1])
            if dest.is_dir():write_json(dest/'FAILED.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc()))
        raise
