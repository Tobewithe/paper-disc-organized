"""Frozen native two-branch candidate and TAL replay, with paired crop controls."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--sample",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args()
    sys.path.insert(0,str(args.root/"shared/vendor/ultralytics_8_4_100"))
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.metrics import box_iou
    from pycocotools.coco import COCO
    assert ultralytics.__version__=="8.4.100"
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    args.out.mkdir(parents=True,exist_ok=True)
    sample=json.loads(args.sample.read_text(encoding="utf-8"))
    if args.limit:
        sample=sample[:args.limit]
    coco=COCO(str(args.root/"assets/datasets/coco/annotations/instances_val2017.json"))
    yolo=YOLO(str(args.root/"assets/models/coco_clean_20260911/yolo26m-seg.pt"))
    model=yolo.model.cuda().float().eval()
    head=model.model[-1]
    assert head.end2end and head.nc==80
    cat_name_to_id={v["name"]:k for k,v in coco.cats.items()}
    cls_to_cat={k:cat_name_to_id[name] for k,name in model.names.items()}
    cat_to_cls={v:k for k,v in cls_to_cat.items()}
    criteria={"one2one":v8DetectionLoss(model,tal_topk=7,tal_topk2=1),"one2many":v8DetectionLoss(model,tal_topk=10)}
    transform=LetterBox((640,640),auto=True,stride=32)
    results=[]
    candidates_handle=(args.out/"candidates.jsonl").open("w",encoding="utf-8")
    start=time.perf_counter()

    def mask_stats(proto,coeff,boxes,shape,original_shape,gt):
        stats=[]
        for first in range(0,len(coeff),8):
            binary=ops.process_mask(proto,coeff[first:first+8],boxes[first:first+8],shape,upsample=True)
            masks=ops.scale_masks(binary[None],original_shape)[0].byte().bool()
            tp=(masks & gt).sum((1,2)).float()
            pa=masks.sum((1,2)).float()
            ga=gt.sum().float()
            values=torch.stack((tp/(pa+ga-tp).clamp(min=1),tp/ga.clamp(min=1),tp/pa.clamp(min=1)),1)
            stats.extend(values.cpu().tolist())
        return np.asarray(stats)

    with torch.inference_mode():
        for number,target in enumerate(sample):
            image_id=target["image_id"]
            im=cv2.imread(str(args.root/"assets/datasets/coco/images/val2017"/coco.imgs[image_id]["file_name"]))
            assert im is not None
            params=transform.get_params({"img":im})
            resized=transform.apply_image({"img":im},params)["img"]
            tensor=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            shape=tuple(tensor.shape[2:])
            model_output,raw=model(tensor)
            assert set(raw)=={"one2one","one2many"}
            anns=[a for a in coco.imgToAnns[image_id] if not a.get("iscrowd",0)]
            gj=next(i for i,a in enumerate(anns) if a["id"]==target["annotation_id"])
            gtboxes=torch.tensor([a["bbox"] for a in anns],device="cuda",dtype=torch.float32)
            gtboxes[:,2:]+=gtboxes[:,:2]
            original_gtboxes=gtboxes.clone()
            gtboxes*=torch.tensor(params["ratio"]*2,device="cuda")
            gtboxes+=torch.tensor([params["left"],params["top"]]*2,device="cuda")
            labels=torch.tensor([cat_to_cls[a["category_id"]] for a in anns],device="cuda").float()[None,:,None]
            valid=torch.ones((1,len(anns),1),device="cuda",dtype=torch.bool)
            gt=torch.from_numpy(coco.annToMask(anns[gj]).astype(bool)).cuda()
            gtcls=cat_to_cls[target["category_id"]]
            primary=raw["one2one"]
            primary_scores=primary["scores"].permute(0,2,1).sigmoid()
            top_score,top_label,top_index=head.get_topk_index(primary_scores,300)
            keep=top_score[0,:,0]>.001
            top_index=top_index[0,:,0][keep]
            top_label=top_label[0,:,0][keep].long()
            primary_boxes=head._get_decode_boxes(primary)[0].T
            # Reconstruct diagnostic slots on THIS forward; cache indices are not
            # identities across fused/unfused devices or nearly tied top-k scores.
            top_boxes=ops.scale_boxes(shape,primary_boxes[top_index].clone(),im.shape[:2])
            top_ious=box_iou(top_boxes,original_gtboxes).cpu().numpy()
            used=set(); current_slots={}
            for slot,raw_id in enumerate(top_index.cpu().tolist()):
                cat=cls_to_cat[int(top_label[slot])]
                available=[j for j,a in enumerate(anns) if j not in used and a["category_id"]==cat and top_ious[slot,j]>=.5]
                if not available:
                    continue
                check_mask=ops.process_mask(primary["proto"][0],primary["mask_coefficient"][0,:,raw_id][None],primary_boxes[raw_id][None],shape,upsample=True)
                exported=ops.scale_masks(check_mask[None],im.shape[:2])[0].byte()
                if not bool(exported.any()):
                    continue
                match=max(available,key=lambda j:top_ious[slot,j])
                used.add(match); current_slots[match]=(slot,raw_id)
            if gj not in current_slots:
                raise RuntimeError(f"Selected GT {target['annotation_id']} lacks a live fixed Box50 slot")
            live_slot,retained=current_slots[gj]
            fixed_box=primary_boxes[retained].clone()
            kept_gt_indices=set(int(x) for x in top_index[top_label==gtcls].cpu().tolist())
            for branch,pred in raw.items():
                criterion=criteria[branch]
                anchors,strides=make_anchors(pred["feats"],head.stride,.5)
                boxes=criterion.bbox_decode(anchors,pred["boxes"].permute(0,2,1))[0]*strides
                scores=pred["scores"].permute(0,2,1).sigmoid()
                _,_,target_scores,foreground,owner=criterion.assigner(scores,boxes[None],anchors*strides,labels,gtboxes[None],valid)
                own=foreground[0].bool() & (owner[0]==gj)
                own_ids=torch.where(own)[0].cpu().tolist()
                original_boxes=ops.scale_boxes(shape,boxes.clone(),im.shape[:2])
                biou=box_iou(original_boxes,original_gtboxes[gj:gj+1])[:,0]
                gt_scores=scores[0,:,gtcls]
                qualified=torch.where(biou>=.5)[0]
                best_box_ids=biou.topk(min(32,len(biou))).indices.cpu().tolist()
                score_ids=qualified[gt_scores[qualified].topk(min(32,len(qualified))).indices].cpu().tolist() if len(qualified) else []
                reference=retained if branch=="one2one" else (int(qualified[gt_scores[qualified].argmax()]) if len(qualified) else int(biou.argmax()))
                pool=sorted(set(best_box_ids+score_ids+own_ids+[reference]))
                ids=torch.tensor(pool,device="cuda")
                coeff=pred["mask_coefficient"][0].T[ids]
                proto=pred["proto"][0]
                native=mask_stats(proto,coeff,boxes[ids],shape,im.shape[:2],gt)
                fixed=mask_stats(proto,coeff,fixed_box[None].expand(len(ids),4),shape,im.shape[:2],gt)
                ref=pool.index(reference)
                b=biou[ids].cpu().numpy(); s=gt_scores[ids].cpu().numpy()
                eligible=np.flatnonzero((b>=.5)&(s>=.001))
                if not len(eligible):
                    eligible=np.asarray([ref])
                best_native=int(eligible[native[eligible,0].argmax()])
                best_fixed=int(eligible[fixed[eligible,0].argmax()])
                # Score/geometry eligibility must be identical for the box and
                # mask selectors. An unrestricted best-box is a different oracle.
                best_box=int(eligible[b[eligible].argmax()])
                geometry=np.flatnonzero(b>=.5)
                geometric_native=int(geometry[native[geometry,0].argmax()])
                geometric_fixed=int(geometry[fixed[geometry,0].argmax()])
                own_pos=[pool.index(i) for i in own_ids]
                own_native=max((float(native[k,0]) for k in own_pos),default=None)
                own_fixed=max((float(fixed[k,0]) for k in own_pos),default=None)
                owner_array=owner[0,ids].cpu().numpy()
                fg_array=foreground[0,ids].bool().cpu().numpy()
                scales=[]
                for stride,feature in zip(head.stride.tolist(),pred["feats"]):
                    scales.extend([int(stride)]*(feature.shape[-2]*feature.shape[-1]))
                def ownership(k):
                    return "own_gt" if fg_array[k] and owner_array[k]==gj else ("other_gt" if fg_array[k] else "background")
                row={**target,"branch":branch,"reference_kind":"retained_output" if branch=="one2one" else "highest_gt_score_box50_raw",
                    "live_slot":live_slot,"raw_candidates":len(boxes),"box50_raw_candidates":len(qualified),"pool_n":len(pool),"eligible_pool_n":len(eligible),
                    "own_assigned_n":len(own_ids),"reference_raw_index":reference,
                    "reference_box_iou":float(b[ref]),"reference_mask_iou":float(native[ref,0]),"reference_fixed_iou":float(fixed[ref,0]),
                    "reference_ownership":ownership(ref),"reference_stride":scales[reference],
                    "best_box_mask_iou":float(native[best_box,0]),"best_native_iou":float(native[best_native,0]),
                    "geometric_best_native_iou":float(native[geometric_native,0]),"geometric_best_fixed_iou":float(fixed[geometric_fixed,0]),
                    "geometric_best_native_gt_score":float(s[geometric_native]),
                    "best_fixed_iou":float(fixed[best_fixed,0]),"best_own_native_iou":own_native,"best_own_fixed_iou":own_fixed,
                    "best_native_raw_index":pool[best_native],"best_fixed_raw_index":pool[best_fixed],
                    "best_native_ownership":ownership(best_native),"best_fixed_ownership":ownership(best_fixed),
                    "best_native_stride":scales[pool[best_native]],"best_fixed_stride":scales[pool[best_fixed]],
                    "best_native_is_sameclass_retained":pool[best_native] in kept_gt_indices if branch=="one2one" else None,
                    "best_native_box_iou":float(b[best_native]),"best_native_gt_score":float(s[best_native]),
                    "reference_gt_score":float(s[ref]),
                    "cache_mask_iou_delta":float(native[ref,0]-target["mask_iou"]) if branch=="one2one" else None,
                    "cache_box_iou_delta":float(b[ref]-target["box_iou"]) if branch=="one2one" else None}
                results.append(row)
                for k,index in enumerate(pool):
                    candidates_handle.write(json.dumps(dict(image_id=image_id,annotation_id=target["annotation_id"],cohort=target["cohort"],branch=branch,
                        raw_index=index,stride=scales[index],box_iou=float(b[k]),gt_score=float(s[k]),ownership=ownership(k),
                        owner_annotation_id=int(anns[int(owner_array[k])]["id"]) if fg_array[k] else None,
                        target_score=float(target_scores[0,index,gtcls]),native_mask_iou=float(native[k,0]),fixed_mask_iou=float(fixed[k,0]),
                        native_coverage=float(native[k,1]),native_purity=float(native[k,2]),fixed_coverage=float(fixed[k,1]),fixed_purity=float(fixed[k,2])))+"\n")
            dump(args.out/"instances_partial.json",results)
            candidates_handle.flush()
            print(json.dumps({"images":number+1,"total":len(sample),"elapsed_s":round(time.perf_counter()-start,1),"image_id":image_id}),flush=True)
    candidates_handle.close()
    dump(args.out/"instances.json",results)
    summary={"images":len(sample),"rows":len(results),"version":ultralytics.__version__,"package":ultralytics.__file__,
             "eval_mode":True,"training":False,"gpu_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"groups":{}}
    rng=np.random.default_rng(16)
    for branch in ("one2one","one2many"):
        for cohort in ("failure","control"):
            group=[r for r in results if r["branch"]==branch and r["cohort"]==cohort]
            if not group:
                continue
            metrics={}
            for key in ("reference_mask_iou","best_box_mask_iou","best_native_iou","reference_fixed_iou","best_fixed_iou","best_own_native_iou","best_own_fixed_iou"):
                values=[r[key] for r in group if r[key] is not None]
                metrics[key]={"n":len(values),"mean":float(np.mean(values)) if values else None,"mask75":sum(v>=.75 for v in values)}
            for name,a,bkey in (("native_gain","best_native_iou","reference_mask_iou"),("fixed_gain","best_fixed_iou","reference_fixed_iou")):
                values=np.array([r[a]-r[bkey] for r in group])
                means=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
                metrics[name]={"mean":float(values.mean()),"bootstrap95":list(map(float,np.quantile(means,[.025,.975])))}
            metrics["best_native_ownership"]=dict(Counter(r["best_native_ownership"] for r in group))
            metrics["best_fixed_ownership"]=dict(Counter(r["best_fixed_ownership"] for r in group))
            metrics["native75_rescued"]=sum(r["reference_mask_iou"]<.75<=r["best_native_iou"] for r in group)
            metrics["fixed75_rescued"]=sum(r["reference_fixed_iou"]<.75<=r["best_fixed_iou"] for r in group)
            metrics["assignment_native75_opportunity"]=sum(r["best_own_native_iou"] is not None and r["best_own_native_iou"]<.75<=r["best_native_iou"] for r in group)
            metrics["assignment_fixed75_opportunity"]=sum(r["best_own_fixed_iou"] is not None and r["best_own_fixed_iou"]<.75<=r["best_fixed_iou"] for r in group)
            summary["groups"][branch+"/"+cohort]=metrics
    primary=[r for r in results if r["branch"]=="one2one"]
    summary["cache_replay_max_abs_mask_iou_delta"]=max(abs(r["cache_mask_iou_delta"]) for r in primary)
    summary["cache_replay_max_abs_box_iou_delta"]=max(abs(r["cache_box_iou_delta"]) for r in primary)
    summary["limitations"]=["32 failures and size/class-matched controls; exploratory, not independent confirmation", "Raw-TAL replay on original COCO boxes, no historical augmentation or gradients", "Candidate best is bounded GT-assisted oracle and can conflict across GT", "Both native branch heads retained: unfused eval; cache uses deployment model", "O2M reference is raw highest-GT-score Box50 candidate, not final NMS output"]
    dump(args.out/"SUMMARY.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
