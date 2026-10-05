"""Exact source-index candidate lifecycle and frozen feature probes on COCO."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,io,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops,nms
from frozen_mechanism_probe import ROOT,Capture,ownership,sha,write_json,box_iou
from structure_features import probe_image

STAGES=['raw_geometry','argmax_class','score','nms','top300','nonempty','eval100']

def require(ok,message):
    if not ok:raise RuntimeError(message)

def save_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)

class TraceCapture(Capture):
    def postprocess(self,preds,img,orig_imgs):
        raw=preds[0][0] if isinstance(preds[0],tuple) else preds[0]
        self.dense=raw.detach().clone()
        return super().postprocess(preds,img,orig_imgs)
    def construct_result(self,pred,img,orig_img,img_path,proto):
        self.before_empty=pred.detach().clone()
        return super().construct_result(pred,img,orig_img,img_path,proto)

def matrix_iou(boxes,gtboxes):
    lt=np.maximum(gtboxes[:,None,:2],boxes[None,:,:2]);rb=np.minimum(gtboxes[:,None,2:],boxes[None,:,2:]);inter=np.maximum(rb-lt,0).prod(2)
    ga=np.maximum(gtboxes[:,2:]-gtboxes[:,:2],0).prod(1);ba=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)
    return inter/np.maximum(ga[:,None]+ba[None,:]-inter,1e-12)

def pair_exists(a,b):
    return bool(len(a) and len(b) and (len(a)>1 or len(b)>1 or a[0]!=b[0]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--images',type=int,default=300);ap.add_argument('--split',choices=['train','val'],default='val');ap.add_argument('--features',action='store_true');a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False);(out/'raw').mkdir()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ann_path=ROOT/f'data/annotations/instances_{a.split}2017.json'
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ann_path))
    meta_path=ROOT/('census/COCO_EVAL_INSTANCE_MANIFEST.csv' if a.split=='val' else 'census/train2017_instances.csv')
    if not meta_path.exists():
        require(a.split=='train','Missing eval manifest')
        # Same per-instance definition; train witness does not depend on a particular census schema.
        metadata={}
        for iid,aa in gt.imgToAnns.items():
            ordinary=[q for q in aa if not q.get('iscrowd',0)]
            for x in ordinary:
                bb=x['bbox'];total=0.
                for y in ordinary:
                    if y['id']==x['id'] or y['category_id']!=x['category_id']:continue
                    cc=y['bbox'];total+=max(0,min(bb[0]+bb[2],cc[0]+cc[2])-max(bb[0],cc[0]))*max(0,min(bb[1]+bb[3],cc[1]+cc[3])-max(bb[1],cc[1]))
                metadata[x['id']]={'ici_same':total/max(bb[2]*bb[3],1e-12)}
    else:
        with meta_path.open(encoding='utf-8-sig') as f:metadata={int(r['annotation_id']):r for r in csv.DictReader(f)}
    if a.split=='val':
        selection_path=ROOT/'diagnostics/local_coeff_eval_20260911/protocol.json';ids=json.loads(selection_path.read_text())['images'][:a.images]
        previous={}
        with (ROOT/'diagnostics/full_local_comparison_20260911/gt_recovery.csv').open() as f:
            for r in csv.DictReader(f):
                if r['arm']=='initial' and int(r['image_id']) in set(ids):previous[int(r['target_annotation'])]=r['mask_recovered75']=='True'
    else:
        selection_path=ROOT/'diagnostics/coefficient_pilot_cache_v4_20260911/selection.json';excluded=set(json.loads(selection_path.read_text())['train'])
        available_ids={int(p.stem) for p in (ROOT/'data/images/train2017').glob('*.jpg')}
        ids=sorted([iid for iid in gt.imgs if iid not in excluded and iid in available_ids],key=lambda x:hashlib.sha256(f'structure-train:20260911:{x}'.encode()).hexdigest())[:a.images];previous={}
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);head=model.model.model[-1]
    require(not head.end2end,'Unexpected end-to-end runtime branch')
    protocol=dict(images=ids,split=a.split,training=False,feature_probes=a.features,stages=STAGES,weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),annotations_sha256=sha(ann_path),selection_source_sha256=sha(selection_path),
        script_sha256=sha(__file__),feature_script_sha256=sha(Path(__file__).with_name('structure_features.py')),nms_source_sha256=sha(nms.__file__),helper_sha256=sha(Path(__file__).with_name('frozen_mechanism_probe.py')),
        candidate_scope='8400 raw grid candidates at 640; class argmax, conf>.001; official class-aware NMS IoU .7; top300; original-mask nonempty; per-category top100. Each GT availability is nonexclusive; pair feasibility requires TWO DIFFERENT candidates. Neither is AP. Official initial mask R75 hits reused only after exact full output cache replay.',
        features='GT-informed within-image oracle probe; disjoint normalized 4x4 checkerboard spatial pixel blocks. 8 minimum and 64 maximum pixels per class/fold, train-only standardization, fixed ridge .1, <=32D Gaussian projection with seeds 0/1/2. Coordinate-only and shuffled train-label controls. RF overlap and geometry confounding remain. Feature controls are not yet category/area/distance matched.',
        pixel_mapping='Original GT pixel centers transformed by actual rounded letterbox resize; bilinear feature sampling. Pair exclusive GT masks, crowd excluded. Mixed cells measured by area resize/pooling.',
        limitations='Exploratory dense-enriched 300 val images reused by project. Train witness only checks execution, no YOLO weight training. Classification similarity does not imply instance confusion.')
    if a.split=='val':protocol['prior_mask_recovery_sha256']=sha(ROOT/'diagnostics/full_local_comparison_20260911/gt_recovery.csv')
    else:protocol['train_available_scope']=dict(available_images=len(available_ids),excluded_adaptation_images=len(excluded),selection='Hash rank of physically available train JPEGs excluding adaptation training; mounted train pool is not full COCO train2017. Fixed execution witness only.')
    write_json(out/'protocol.json',protocol)
    maps={};hooks=[]
    if a.features:
        for index in [4,6,10,16,19,22]:
            def hook(module,args,output,index=index):maps[f'layer{index}']=output.detach()
            hooks.append(model.model.model[index].register_forward_hook(hook))
        for branchname,branch in [('cls',head.cv3),('box',head.cv2),('mask',head.cv4)]:
            for level,item in enumerate(branch):
                def hook(module,args,output,branchname=branchname,level=level):maps[f'{branchname}_s{8*2**level}']=args[0].detach()
                hooks.append(item[-1].register_forward_hook(hook))
    rows=[];pairrows=[];featurerows=[];skips=[];witness=[];start=time.monotonic()
    for number,iid in enumerate(ids,1):
        maps.clear()
        with torch.inference_mode():
            model.predict(str(ROOT/f'data/images/{a.split}2017'/gt.imgs[iid]['file_name']),predictor=TraceCapture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
            predictor=model.predictor;raw=predictor.dense;cap=predictor.capture;pre=predictor.before_empty
            require(raw.ndim==3 and raw.shape[1]==116,'Unexpected raw shape')
            allnms,indices=nms.non_max_suppression(raw.clone(),conf_thres=.001,iou_thres=.7,nc=80,max_det=raw.shape[-1],return_idxs=True,end2end=False)
            keep_all=indices[0].flatten();keep300=keep_all[:300]
            require(torch.equal(allnms[0][:300],pre),'NMS/source index replay differs')
            replay=ops.process_mask(cap['proto'],pre[:,6:],pre[:,:4],cap['input_shape'],upsample=True)
            nonempty=replay.flatten(1).any(1).bool();keep=keep300[nonempty]
            require(torch.equal(raw[0,84:,keep].T,cap['coeff']),'Direct raw index coefficient mismatch')
            rawboxes=ops.xywh2xyxy(raw[0,:4].T.clone());scaled=ops.scale_boxes(cap['input_shape'],rawboxes.clone(),cap['shape'])
            require(torch.equal(scaled[keep],cap['detections'][:,:4]),'Source-index box replay mismatch')
            cache_error=0.
            if a.split=='val':
                with np.load(ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz') as cached:
                    for key in ['coeff','boxes','detections','proto']:
                        value=cap[key].cpu().numpy();require(value.shape==cached[key].shape,'Cache shape mismatch');error=float(np.abs(value-cached[key]).max()) if value.size else 0.;cache_error=max(cache_error,error)
                    require(cache_error==0.,f'Original cache mismatch {iid}: {cache_error}')
                    mapping=dict(zip(map(int,cached['mapping_gt']),map(int,cached['mapping_pred'])))
            else:mapping=ownership(gt,iid,cap['detections'])
            scores=raw[0,4:84].T.cpu().numpy();cls=scores.argmax(1);confidence=scores.max(1);nc=len(cls)
            pools=dict(raw_geometry=np.arange(nc),argmax_class=np.arange(nc),score=np.flatnonzero(confidence>.001),nms=keep_all.cpu().numpy(),top300=keep300.cpu().numpy(),nonempty=keep.cpu().numpy())
            filtered=[]
            for label in range(80):filtered.extend([int(j) for j in pools['nonempty'] if cls[j]==label][:100])
            pools['eval100']=np.array(filtered,dtype=int)
            ordinary=[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0)];categories=sorted(gt.cats);catindex={v:k for k,v in enumerate(categories)}
            gtboxes=np.array([[q['bbox'][0],q['bbox'][1],q['bbox'][0]+q['bbox'][2],q['bbox'][1]+q['bbox'][3]] for q in ordinary]).reshape(-1,4)
            overlaps=matrix_iou(scaled.cpu().numpy(),gtboxes);available={};anchors=[set() for _ in ordinary]
            for g,ann in enumerate(ordinary):
                label=catindex[ann['category_id']];edges={}
                for stage,pool in pools.items():edges[stage]=pool[(overlaps[g,pool]>=.5)&((cls[pool]==label) if stage!='raw_geometry' else True)]
                available[ann['id']]=edges;flags={stage:bool(len(edges[stage])) for stage in STAGES}
                for earlier,later in zip(STAGES,STAGES[1:]):require(not flags[later] or flags[earlier],'Non-nested availability')
                best=int(overlaps[g].argmax());validgeo=overlaps[g]>=.5
                row=dict(image_id=iid,annotation_id=ann['id'],category_id=ann['category_id'],area=ann['area'],ici_same=float(metadata[ann['id']]['ici_same']),best_raw_box_iou=float(overlaps[g,best]),best_raw_source_index=best,best_geometry_true_class_score=float(scores[validgeo,label].max()) if validgeo.any() else 0.,official_bbox50_matched=ann['id'] in mapping,
                    **{f'{stage}_available':flags[stage] for stage in STAGES},**{f'{stage}_count':len(edges[stage]) for stage in STAGES},first_unavailable=next((stage for stage in STAGES if not flags[stage]),'all_stages_available'))
                if a.split=='val':row['official_mask75_recovered']=previous[ann['id']]
                rows.append(row)
            for j,x in enumerate(ordinary):
                for y in ordinary[j+1:]:
                    if x['category_id']!=y['category_id'] or box_iou(x['bbox'],y['bbox'])<=.05:continue
                    pair=dict(image_id=iid,annotation_a=x['id'],annotation_b=y['id'],ici_max=max(float(metadata[q['id']]['ici_same']) for q in [x,y]),gt_box_iou=box_iou(x['bbox'],y['bbox']))
                    for stage in STAGES:pair[stage+'_two_distinct']=pair_exists(available[x['id']][stage],available[y['id']][stage])
                    pair['official_bbox50_both']=x['id'] in mapping and y['id'] in mapping
                    if a.split=='val':pair['official_mask75_both']=previous[x['id']] and previous[y['id']]
                    pairrows.append(pair)
            np.savez_compressed(out/'raw'/f'{iid}.npz',boxes_input=rawboxes.cpu().numpy(),boxes_original=scaled.cpu().numpy(),class_scores=scores,coefficients=raw[0,84:].T.cpu().numpy(),nms_indices=pools['nms'],top300_indices=pools['top300'],nonempty_indices=pools['nonempty'],eval100_indices=pools['eval100'],annotation_ids=np.array([q['id'] for q in ordinary]),bbox_iou=overlaps,mapping_gt=np.array(list(mapping)),mapping_pred=np.array(list(mapping.values())))
            if a.features:
                fr,ss=probe_image(gt,iid,maps,cap,mapping,metadata);featurerows.extend(fr);skips.extend(ss)
            witness.append(dict(image_id=iid,raw_candidates=nc,nms_candidates=len(keep_all),top300=len(keep300),nonempty=len(keep),raw_index_replay=True,cache_max_error=cache_error))
        if number%10==0 or number==len(ids):
            progress=dict(completed=number,total=len(ids),feature_rows=len(featurerows),seconds=round(time.monotonic()-start,1));write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    for h in hooks:h.remove()
    save_csv(out/'instances.csv',rows);save_csv(out/'pairs.csv',pairrows);save_csv(out/'features.csv',featurerows);save_csv(out/'feature_skipped.csv',skips);save_csv(out/'witness.csv',witness)
    summary=[]
    for group in ['all','high','low']:
        selected=[r for r in rows if group=='all' or (r['ici_same']>.5+1e-10)==(group=='high')]
        if not selected:continue
        summary.append(dict(group=group,gt=len(selected),**{stage:float(np.mean([r[stage+'_available'] for r in selected])) for stage in STAGES},official_bbox50_matched=float(np.mean([r['official_bbox50_matched'] for r in selected])),**({'official_mask75_recovered':float(np.mean([r['official_mask75_recovered'] for r in selected]))} if a.split=='val' else {})))
    save_csv(out/'stage_summary.csv',summary)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',training=False,images=len(ids),gt=len(rows),same_adjacent_pairs=len(pairrows),feature_rows=len(featurerows),seconds=time.monotonic()-start,script_sha256=sha(__file__),hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
