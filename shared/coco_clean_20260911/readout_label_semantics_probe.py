"""S031: fixed-target intervention on overlap encoding and polygon raster labels.

Uses the same 160 targets, 512 positions, frozen prototypes, original crop and
predeclared S030 lambda .01. Official independent Format is a diagnostic control,
not a novel method. Both splits use their own GT; no shared network training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, contextlib, io, json, math, shutil, time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from native_label_pipeline_probe import format_native, mask_of
from readout_input_probe import sha, write_json
from readout_fit_transfer_diagnostic import read_np, numbers, csv_save
from readout_regularized_oracle import solve_regularized


def iou(a,b):
    un=int((a|b).sum())
    return int((a&b).sum())/un if un else 1.


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False);(out/'coefficients').mkdir();start=time.monotonic()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads((a.cache/'COMPLETE.json').read_text());hh={k.replace('\\','/'):v for k,v in receipt['hashes'].items()}
    prev=json.loads((a.previous/'COMPLETE.json').read_text());ph={k.replace('\\','/'):v for k,v in prev['hashes'].items()}
    target_dict=json.loads((a.previous/'protocol.json').read_text())['targets']
    targets=[(s,int(i),int(t)) for s,rr in target_dict.items() for i,t in rr]
    write_json(out/'protocol.json',dict(experiment='S031_LABEL_SEMANTICS',targets=targets,network_training=False,
        arms=['original','native_overlap','native_independent','raw_coco'],
        intervention='ONLY per-target binary supervision label changes. Same512sample positions and nativeGTbox factor, '
        'sameFP64RMS-scaledLBFGS120 andlambda.01 fromoriginal. Nativeoverlap coefficients savedS030. '
        'Nativeindependent actual Format overlap=False; rawCOCO annToMask nearest-exact into same640geometry.',
        motivation='Posthoc S030 catastrophic cases improved native-label fit while rawCOCO coverage collapsed. '
        'Test overlap encoding separately from polygon rasterization on ALL160preexisting targets,not only selectedfailures.',
        checks='Rebuild nativeFormat masks and require all2048samplelabels equal cachedY; samegeometry/IDs/boxes. '
        'Allnativeinstances in image participate in overlap encoding,not only selectedtargets.',
        evaluation='Whole originalCOCO mask and conditional spatial decomposition, originalpredictedcrop. '
        'Samplemetrics evaluated against each fixedlabel target,do not compare absolute losses acrosslabel definitions.',
        limits='Both splits use ownGT;no inference method/AP/recall/no dense specificity with23high instances. '
        'Official overlap_mask=False alreadyexists; even positiveintervention is not novelty or historicaltraining replay.',
        sources=dict(cache=sha(a.cache/'COMPLETE.json'),previous=sha(a.previous/'COMPLETE.json'),script=sha(__file__))))
    shutil.copy2(__file__,out/Path(__file__).name)
    subset=a.cache/'conversion_input/instances_probe.json'
    if sha(subset)!=hh['conversion_input/instances_probe.json']:raise RuntimeError('Changed annotation subset')
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(subset))
    cfg=get_cfg(overrides=dict(task='segment',imgsz=640,mask_ratio=1,overlap_mask=True,rect=False,cache=False,workers=0,fraction=1.0))
    data=dict(names={k:gt.cats[c]['name'] for k,c in enumerate(sorted(gt.cats))},nc=80,channels=3)
    with contextlib.redirect_stdout(io.StringIO()):
        ds=build_yolo_dataset(cfg,str(a.cache/'converted/images/probe'),1,data,mode='val',rect=False)
    index={int(Path(l['im_file']).stem):k for k,l in enumerate(ds.labels)}
    rows=[];labelrows=[];solvers=[];pixelrows=[];witness=[];done=0
    for iid in sorted({t[1] for t in targets}):
        path=a.cache/'images'/f'{iid}.npz'
        if sha(path)!=hh[f'images/{iid}.npz']:raise RuntimeError('Changed NPZ')
        item=read_np(path);image_meta=json.loads((a.cache/'images'/f'{iid}.json').read_text())
        lab=ds.labels[index[iid]];labelpath=a.cache/'converted/labels/probe'/f'{iid:012d}.txt'
        if sha(labelpath)!=image_meta['label_sha256']:raise RuntimeError('Changed YOLO labels')
        raw=deepcopy(lab);raw.pop('shape',None);raw['img']=cv2.imread(lab['im_file'])
        h,w=raw['img'].shape[:2];shape=(h,w);ishape=tuple(map(int,item['input_shape']))
        if shape!=tuple(item['shape']):raise RuntimeError('Geometry changed')
        raw=ds.update_labels_info(raw);raw['ori_shape']=shape;raw['ratio_pad']=(1.,1.)
        raw=LetterBox(new_shape=ishape,auto=False,scaleup=True)(raw)
        ids=image_meta['native_ids'];over=format_native(raw,ids,1,True);ind=format_native(raw,ids,1,False)
        p=torch.tensor(item['proto'],device='cuda').double();gain=min(ishape[0]/h,ishape[1]/w)
        rh,rw=round(h*gain),round(w*gain);top=round((ishape[0]-rh)/2-.1);left=round((ishape[1]-rw)/2-.1)
        masks={r['id']:gt.annToMask(r).astype(bool) for r in gt.imgToAnns[iid]}
        crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        for ann in gt.imgToAnns[iid]:
            if ann.get('iscrowd',0):crowd|=masks[ann['id']]
            else:union|=masks[ann['id']]
        for split,_,aid in [t for t in targets if t[1]==iid]:
            k=int(np.flatnonzero(item['annotation_ids']==aid)[0]);j=int(item['prediction_indices'][k])
            yover=mask_of(over,aid).astype(bool);yind=mask_of(ind,aid).astype(bool)
            yc=torch.zeros(ishape,device='cuda',dtype=torch.bool)
            yc[top:top+rh,left:left+rw]=F.interpolate(torch.tensor(masks[aid],device='cuda').float()[None,None],(rh,rw),mode='nearest-exact')[0,0].bool()
            yraw=yc.cpu().numpy();positions=item['sample_positions'][k]
            xor=int(np.count_nonzero(yover.flatten()[positions]!=item['sample_y'][k]))
            if xor:raise RuntimeError(f'Cached native labels fail replay {iid}:{aid}:{xor}')
            norm=over['boxes'][over['position'][aid]]
            if not np.allclose(norm,ind['boxes'][ind['position'][aid]],rtol=1e-6,atol=1e-7):raise RuntimeError('Independent GTbox differs')
            pp=torch.tensor(item['sample_p'][k],device='cuda');c0=torch.tensor(item['coeff'][j],device='cuda')
            box=torch.tensor(item['boxes'][j:j+1],device='cuda');factor=float(item['loss_factor'][k])
            savedpath=a.previous/'coefficients'/f'{iid}_{aid}.npz'
            if sha(savedpath)!=ph[f'coefficients/{iid}_{aid}.npz']:raise RuntimeError('Saved coefficient changed')
            saved=read_np(savedpath);cs=dict(original=c0.double(),native_overlap=torch.tensor(saved['ridge_0.01'],device='cuda'))
            annotation=gt.anns[aid];base=dict(split=split,image_id=iid,annotation_id=aid,category_id=annotation['category_id'])
            labels=dict(native_overlap=yover,native_independent=yind,raw_coco=yraw)
            for mode in ['native_independent','raw_coco']:
                yy=torch.tensor(labels[mode].flatten()[positions[:512]],device='cuda').float()
                cc,info=solve_regularized(pp[:512],yy,c0,factor,.01);cs[mode]=cc
                solvers.append(dict(**base,mode=mode,**info))
            np.savez_compressed(out/'coefficients'/f'{iid}_{aid}.npz',**{m:c.cpu().numpy() for m,c in cs.items()})
            seen=set(map(int,positions[:512]));held=[]
            for t in range(512,len(positions)):
                pos=int(positions[t])
                if pos not in seen:held.append(t);seen.add(pos)
            binaries={}
            for mode,cc in cs.items():
                zz=pp.double()@cc
                for target_name,y in labels.items():
                    yy=torch.tensor(y.flatten()[positions],device='cuda').double()
                    for domain,ix in [('train512',slice(0,512)),('unused',held)]:
                        pixelrows.append(dict(**base,mode=mode,label=target_name,domain=domain,**numbers(zz[ix],yy[ix],factor)))
                dense=F.interpolate((cc@p.flatten(1)).reshape(1,1,*p.shape[-2:]),ishape,mode='bilinear',align_corners=False)[0,0]
                binaries[mode]=ops.crop_mask((dense>0).byte()[None],box)[0]
            binput={m:bb.cpu().numpy().astype(bool) for m,bb in binaries.items()}
            origmask=ops.scale_masks(torch.stack(list(binaries.values()))[:,None],shape)[:,0]>.5
            own=masks[aid]&~crowd;area=int(own.sum());same=np.zeros(shape,bool)
            for ann in gt.imgToAnns[iid]:
                if not ann.get('iscrowd',0) and ann['id']!=aid and ann['category_id']==annotation['category_id']:same|=masks[ann['id']]
            allnb=sum(1 for ann in gt.imgToAnns[iid] if not ann.get('iscrowd',0) and ann['id']!=aid)
            targetcrop=ops.crop_mask(torch.ones((1,*ishape),device='cuda'),box)[0].cpu().numpy().astype(bool)
            labelrows.append(dict(**base,other_instances=allnb,raw640_pixels=int(yraw.sum()),independent_pixels=int(yind.sum()),overlap_pixels=int(yover.sum()),
                overlap_vs_independent_iou=iou(yover,yind),independent_vs_raw640_iou=iou(yind,yraw),overlap_vs_raw640_iou=iou(yover,yraw),
                overlap_deleted_own_pixels=int((yind&~yover).sum()),raw_gt_rejected_by_overlap=int((yraw&~yover).sum()),
                raw_gt_rejected_by_independent=int((yraw&~yind).sum()),
                original_tp_rejected_by_overlap=int((binput['original']&yraw&~yover).sum()),
                original_tp_pixels=int((binput['original']&yraw).sum()),
                native_lost_tp_rejected_by_overlap=int((binput['original']&~binput['native_overlap']&yraw&~yover).sum()),
                native_lost_tp_pixels=int((binput['original']&~binput['native_overlap']&yraw).sum()),
                original_cropped_raw640_iou=iou(binput['original'],yraw),
                crop_raw_gt_coverage=float((targetcrop&yraw).sum()/max(1,yraw.sum()))))
            for t,mode in enumerate(binaries):
                pred=origmask[t].cpu().numpy();vp=pred&~crowd
                rows.append(dict(**base,mode=mode,coco_iou=iou(pred,masks[aid]),
                    coverage=float((vp&own).sum()/area) if area else None,
                    neighbor=float((vp&same&~own).sum()/area) if area else None,
                    background=float((vp&~union).sum()/area) if area else None,
                    full640_iou_raw=iou(binput[mode],yraw),full640_iou_overlap=iou(binput[mode],yover),
                    full640_iou_independent=iou(binput[mode],yind)))
            witness.append(dict(**base,cached_sample_label_xor=xor,overlap_subset_of_independent=not bool((yover&~yind).any())))
            done+=1
            if done%20==0:print(json.dumps(dict(stage='solve',targets=done,total=160,seconds=time.monotonic()-start)),flush=True)
    for name,value in [('spatial',rows),('pixels',pixelrows),('labels',labelrows),('solver',solvers),('witness',witness)]:csv_save(out/f'{name}.csv',value)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',targets=done,seconds=time.monotonic()-start,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(stage='COMPLETE',targets=done,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--out' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--out')+1])
            if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
