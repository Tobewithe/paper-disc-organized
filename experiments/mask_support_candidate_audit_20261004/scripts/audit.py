import argparse, json, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# Use the vendored/frozen runtime. The remote launcher sets PYTHONPATH to 8.4.100.
from ultralytics import YOLO
from ultralytics.utils import ops


def xyxy_iou(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)),dtype=np.float32)
    lt=np.maximum(a[:,None,:2],b[None,:,:2]); rb=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.prod(np.maximum(0,rb-lt),axis=-1)
    aa=np.prod(np.maximum(0,a[:,2:]-a[:,:2]),axis=1); bb=np.prod(np.maximum(0,b[:,2:]-b[:,:2]),axis=1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)


def rle_mask(coco, ann, h, w):
    seg=ann.get('segmentation')
    if isinstance(seg, list):
        rles=mask_utils.frPyObjects(seg,h,w); rle=mask_utils.merge(rles)
    elif isinstance(seg, dict):
        rle=seg
    else:
        return np.zeros((h,w),np.uint8)
    return mask_utils.decode(rle).astype(np.uint8)


def run_one(model, image_path, image_id, coco, max_det=300, imgsz=640):
    bgr=cv2.imread(str(image_path));
    if bgr is None: raise RuntimeError(f'cannot read {image_path}')
    h,w=bgr.shape[:2]
    head=model.model.model[-1]
    captured=[]
    hook=head.register_forward_hook(lambda m,i,o: captured.append(o))
    try:
        results=model.predict(source=str(image_path), imgsz=imgsz, conf=0.001, max_det=max_det, retina_masks=True, verbose=False, device=0)
    finally: hook.remove()
    if not captured: raise RuntimeError('head hook empty')
    out=captured[-1]
    # Segment26 eval returns ((decoded, proto), branch_dict)
    if not (isinstance(out, tuple) and len(out)>=2 and isinstance(out[1],dict)):
        raise RuntimeError(f'unexpected head output type {type(out)}')
    one=out[1].get('one2one',out[1])
    proto=one.get('proto', None)
    if proto is None:
        # inference proto is in first tuple element for some builds
        first=out[0]; proto=first[1] if isinstance(first,tuple) else None
    if proto is None: raise RuntimeError('prototype missing')
    boxes_in=head._get_decode_boxes(one)[0].T
    scores_all=one['scores'][0].sigmoid().T
    score_vals, cls_all=scores_all.max(dim=1)
    coeff=one['mask_coefficient'][0].T
    # decode input-coordinate boxes to original pixels
    input_h=int(one['feats'][0].shape[-2]*head.stride[0]); input_w=int(one['feats'][0].shape[-1]*head.stride[0])
    boxes=ops.scale_boxes((input_h,input_w), boxes_in.clone(), (h,w)).detach().cpu().numpy().astype(np.float32)
    score_vals=score_vals.detach().cpu().numpy().astype(np.float32); cls_all=cls_all.detach().cpu().numpy().astype(np.int64)
    coeff=coeff.detach(); proto=proto[0] if proto.ndim==4 else proto
    n=len(boxes)
    # top-k by max class score is the official one2one candidate budget
    top_idx=np.argsort(-score_vals,kind='stable')[:min(max_det,n)]
    top_set=set(int(x) for x in top_idx)
    # decode top candidates in one batch
    with torch.no_grad():
        masks=ops.process_mask_native(proto, coeff[top_idx], torch.as_tensor(boxes[top_idx],device=coeff.device), (h,w)).detach().cpu().numpy().astype(np.uint8)
    anns=coco.loadAnns(coco.getAnnIds(imgIds=[image_id],iscrowd=False))
    if image_id == 9:
        print('DBG_NAMES', type(model.names), list(model.names.items())[:3] if hasattr(model.names,'items') else list(model.names)[:3], 'ann0', coco.cats[int(anns[0]['category_id'])]['name'] if anns else None, 'clsuniq', np.unique(cls_all)[:10], 'topcls', np.unique(cls_all[top_idx]), flush=True)
    rows=[]; feature_rows=[]
    # Candidate-to-candidate descriptors are computed once per image.
    top_cls=cls_all[top_idx]; top_masks=masks.astype(bool); flat=top_masks.reshape(len(top_masks),-1).astype(np.float32)
    area=flat.sum(1); mi=(flat@flat.T); mui=area[:,None]+area[None,:]-mi; mask_iou_mat=mi/np.maximum(mui,1.0); np.fill_diagonal(mask_iou_mat,0.0)
    top_co=coeff[top_idx].detach().cpu().numpy(); cn=np.linalg.norm(top_co,axis=1,keepdims=True); cos_mat=(top_co@top_co.T)/np.maximum(cn@cn.T,1e-8); np.fill_diagonal(cos_mat,0.0)
    box_iou_mat=xyxy_iou(boxes[top_idx],boxes[top_idx]); np.fill_diagonal(box_iou_mat,0.0)
    for ann in anns:
        gtcls=int(coco.cats[int(ann['category_id'])]['id'])
        # COCO category id is not contiguous; model classes use category mapping.
        # convert via model.names lookup.
        catname=coco.cats[int(ann['category_id'])]['name']
        name_to_cls={str(v):int(k) for k,v in (model.names.items() if hasattr(model.names,'items') else enumerate(model.names))}
        model_cls=name_to_cls.get(catname,-1)
        cls_candidates=np.flatnonzero(cls_all==model_cls)
        cls_candidates=np.asarray([x for x in cls_candidates if int(x) in top_set],dtype=np.int64)
        if len(cls_candidates)==0: continue
        gt=rle_mask(coco,ann,h,w)
        gt_area=float(gt.sum());
        cand_pos=np.asarray([np.flatnonzero(top_idx==x)[0] for x in cls_candidates],dtype=np.int64)
        cm=masks[cand_pos]
        inter=(cm & gt[None]).sum((1,2)); union=(cm | gt[None]).sum((1,2)); miou=inter/np.maximum(union,1)
        biou=xyxy_iou(boxes[cls_candidates],np.asarray([ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]],np.float32)[None])[:,0]
        ccoef=coeff[cls_candidates].detach().cpu().numpy()
        for q,(cid,mv,bv) in enumerate(zip(cls_candidates.tolist(),miou.tolist(),biou.tolist())):
            if bv < 0.5: continue
            bb=boxes[cid]; bw=max(float(bb[2]-bb[0]),1e-6); bh=max(float(bb[3]-bb[1]),1e-6)
            m=masks[cand_pos[q]].astype(bool); ma=float(m.sum())
            x0=max(0,int(np.floor(bb[0]))); y0=max(0,int(np.floor(bb[1]))); x1=min(w,int(np.ceil(bb[2]))); y1=min(h,int(np.ceil(bb[3])))
            crop=m[y0:y1,x0:x1]; fill=float(crop.mean()) if crop.size else 0.0
            if crop.size:
                edge=np.zeros_like(crop,dtype=bool); edge[:min(3,edge.shape[0])]=True; edge[-min(3,edge.shape[0]):]=True; edge[:,:min(3,edge.shape[1])]=True; edge[:,-min(3,edge.shape[1]):]=True; edge_frac=float((crop & edge).sum()/max(crop.sum(),1))
            else: edge_frac=0.0
            cp=int(cand_pos[q]); samepos=np.flatnonzero(top_cls==cls_all[cid]); samepos=samepos[samepos!=cp]
            mask_overlap_max=float(mask_iou_mat[cp,samepos].max()) if len(samepos) else 0.0
            coef_cos_max=float(cos_mat[cp,samepos].max()) if len(samepos) else 0.0
            box_neighbor_count=int((box_iou_mat[cp,samepos]>0.1).sum()) if len(samepos) else 0
            feature_rows.append(dict(image_id=int(image_id),annotation_id=int(ann['id']),category=catname,target_mask_iou=float(mv),box_iou_label=float(bv),score=float(score_vals[cid]),global_rank=int(np.where(np.argsort(-score_vals,kind='stable')==cid)[0][0]+1),box_area_frac=float((bw*bh)/(w*h)),box_aspect=float(np.log(bw/bh)),center_x=float((bb[0]+bb[2])/(2*w)),center_y=float((bb[1]+bb[3])/(2*h)),mask_area_frac=float(ma/(w*h)),mask_box_fill=fill,mask_edge_frac=edge_frac,coef_norm=float(np.linalg.norm(ccoef[q])),coef_std=float(np.std(ccoef[q])),mask_overlap_max=mask_overlap_max,coef_cos_max=coef_cos_max,box_neighbor_count=box_neighbor_count))
        # score winner and mask winner
        sorder=np.lexsort((cls_candidates,-score_vals[cls_candidates])); si=int(sorder[0]); bi=int(np.argmax(miou));
        good=np.flatnonzero(biou>=0.5)
        if len(good):
            bgi=int(good[np.argmax(miou[good])]); sgi=int(good[np.argmax(score_vals[cls_candidates[good]])])
            best_good_iou=float(miou[bgi]); best_good_box_iou=float(biou[bgi]); best_good_score=float(score_vals[cls_candidates[bgi]]); best_good_raw=int(cls_candidates[bgi])
            score_good_iou=float(miou[sgi]); score_good_box_iou=float(biou[sgi]); score_good_score=float(score_vals[cls_candidates[sgi]]); score_good_raw=int(cls_candidates[sgi])
        else:
            bgi=-1; sgi=-1; best_good_iou=float('nan'); best_good_box_iou=float('nan'); best_good_score=float('nan'); best_good_raw=-1; score_good_iou=float('nan'); score_good_box_iou=float('nan'); score_good_score=float('nan'); score_good_raw=-1
        rows.append(dict(image_id=int(image_id),annotation_id=int(ann['id']),category=catname,gt_area=gt_area,neighbor_count=0,
            score_iou=float(miou[si]),best_iou=float(miou[bi]),mask_regret=float(miou[bi]-miou[si]),
            best_good_iou=best_good_iou,score_good_iou=score_good_iou,good_regret=(best_good_iou-score_good_iou) if len(good) else float('nan'), good_candidate_count=int(len(good)),
            score_box_iou=float(biou[si]),best_box_iou=float(biou[bi]),score=float(score_vals[cls_candidates[si]]),best_score=float(score_vals[cls_candidates[bi]]),
            best_good_box_iou=best_good_box_iou,best_good_score=best_good_score,score_good_box_iou=score_good_box_iou,score_good_score=score_good_score,
            score_raw_id=int(cls_candidates[si]),best_raw_id=int(cls_candidates[bi]),score_rank=int(np.where(np.argsort(-score_vals,kind='stable')==cls_candidates[si])[0][0]+1),
            best_rank=int(np.where(np.argsort(-score_vals,kind='stable')==cls_candidates[bi])[0][0]+1),best_good_raw_id=best_good_raw,score_good_raw_id=score_good_raw,candidate_count=int(len(cls_candidates))))
    # neighbor counts among same-category GT boxes with IoU > 0.1
    if rows:
        rb=np.asarray([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns],np.float32)
        rc=[coco.cats[int(a['category_id'])]['name'] for a in anns];
        for row in rows:
            j=next(i for i,a in enumerate(anns) if int(a['id'])==row['annotation_id']);
            same=[i for i,c in enumerate(rc) if c==row['category'] and i!=j]
            row['neighbor_count']=int((xyxy_iou(rb[j:j+1],rb[same])[0]>0.1).sum()) if same else 0
    return rows, feature_rows, dict(raw_count=int(n),top_count=int(len(top_idx)),gt_count=len(anns),pred_count=len(results[0].boxes) if results and results[0].boxes is not None else 0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--images',type=int,default=50); ap.add_argument('--start',type=int,default=0); ap.add_argument('--imgsz',type=int,default=640); args=ap.parse_args()
    root=Path(args.root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    split=os.environ.get('COCO_SPLIT','val2017'); ann_path=root/'data'/'annotations'/f'instances_{split}.json'; imgroot=root/'data'/'images'/split; weight=root/'models'/'yolo26m-seg.pt'
    coco=COCO(str(ann_path)); ids=sorted(coco.getImgIds())[args.start:args.start+args.images]
    model=YOLO(str(weight)); model.model.float().cuda().eval()
    rows=[]; feature_rows=[]; stats=[]; t=time.time()
    for k,iid in enumerate(ids,1):
        info=coco.loadImgs([iid])[0]; r,f,s=run_one(model,imgroot/info['file_name'],iid,coco,max_det=300,imgsz=args.imgsz); rows.extend(r); feature_rows.extend(f); s['image_id']=int(iid); stats.append(s)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if k==1 or k%10==0 or k==len(ids): print(json.dumps({'progress':k,'total':len(ids),'rows':len(rows),'elapsed_s':round(time.time()-t,1)}),flush=True)
    import pandas as pd
    df=pd.DataFrame(rows); df.to_csv(out/'rows.csv',index=False); pd.DataFrame(feature_rows).to_csv(out/'quality_features.csv',index=False)
    summary={'status':'completed','images':len(ids),'gt_with_candidate':int(len(df)),'raw_mean':float(np.mean([x['raw_count'] for x in stats])) if stats else 0,'top_mean':float(np.mean([x['top_count'] for x in stats])) if stats else 0}
    if len(df):
        for key in ('score_iou','best_iou','mask_regret','best_good_iou','score_good_iou','good_regret','score_box_iou','best_box_iou','score_good_box_iou','candidate_count','good_candidate_count','neighbor_count'):
            summary[key+'_mean']=float(df[key].mean())
        summary['regret_positive_01']=float((df.mask_regret>0.01).mean()); summary['regret_positive_02']=float((df.mask_regret>0.02).mean()); summary['neighbor_regret_mean']=float(df.loc[df.neighbor_count>0,'mask_regret'].mean()) if (df.neighbor_count>0).any() else None
        summary['nonneighbor_regret_mean']=float(df.loc[df.neighbor_count==0,'mask_regret'].mean()) if (df.neighbor_count==0).any() else None
        good=df[df.good_candidate_count>0]; summary['good_rows']=int(len(good)); summary['good_regret_positive_01']=float((good.good_regret>0.01).mean()) if len(good) else None; summary['good_regret_positive_02']=float((good.good_regret>0.02).mean()) if len(good) else None; summary['good_neighbor_regret_mean']=float(good.loc[good.neighbor_count>0,'good_regret'].mean()) if (good.neighbor_count>0).any() else None; summary['good_nonneighbor_regret_mean']=float(good.loc[good.neighbor_count==0,'good_regret'].mean()) if (good.neighbor_count==0).any() else None
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'COMPLETE.json').write_text(json.dumps({'completed':True,'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__': main()


