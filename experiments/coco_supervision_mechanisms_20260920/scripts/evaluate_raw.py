"""Score-independent raw geometry, supervision replay, and normal-output export.

All GT/candidate maxima and Mask50/75 edges are exact for the stated decoder.
Pruning uses a conservative spatial-support envelope, never a score threshold.
"""
import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def cardinality(edges):
    """Maximum distinct-candidate coverage; rows=candidates, columns=GT."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if not edges.size:
        return 0
    return int((maximum_bipartite_matching(csr_matrix(edges.T), perm_type="column") >= 0).sum())


def state(box, mask, joint):
    if joint:
        return "joint_good"
    if box >= .75 and mask >= .75:
        return "separate_good_no_joint"
    if box >= .75:
        return "box_good_mask_unavailable"
    if mask >= .75:
        return "mask_good_box_unavailable"
    return "neither_good"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--split", choices=["train2017", "val2017"], default="train2017")
    ap.add_argument("--image-ids", type=Path, required=True)
    ap.add_argument("--exhaustive", action="store_true")
    ap.add_argument("--check-exhaustive", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(a.root / "shared/vendor/ultralytics_8_4_100"))
    import cv2
    import torch
    import torch.nn.functional as F
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.tal import make_anchors
    from pycocotools.coco import COCO
    from pycocotools import mask as mu

    assert ultralytics.__version__ == "8.4.100"
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    a.out.mkdir(parents=True, exist_ok=True)
    shards = a.out / "images"
    shards.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(a.root / f"assets/datasets/coco/annotations/instances_{a.split}.json"))
    ids = json.loads(a.image_ids.read_text())[a.offset:a.offset+a.limit]
    assert len(ids) == len(set(ids)) and set(ids) <= set(coco.imgs)
    dump(a.out / "image_ids.json", ids)
    yolo = YOLO(str(a.root / "assets/models/coco_clean_20260911/yolo26m-seg.pt"))
    model = yolo.model.cuda().float().eval()
    head = model.model[-1]
    assert head.end2end
    names = {v["name"]: k for k, v in coco.cats.items()}
    cls_cat = {k: names[v] for k, v in model.names.items()}
    cat_cls = {v: k for k, v in cls_cat.items()}
    criteria = {"one2one": v8DetectionLoss(model, tal_topk=7, tal_topk2=1),
                "one2many": v8DetectionLoss(model, tal_topk=10)}
    transform = LetterBox((640, 640), auto=True, stride=32)
    start = time.monotonic()
    total_gt = total_decoded = total_raw = 0
    rows_out = (a.out / "instances.jsonl").open("w", encoding="utf-8")
    images_out = (a.out / "image_summary.jsonl").open("w", encoding="utf-8")

    def masks_for(p, boxes, indices, shape, orig):
        idx = torch.as_tensor(indices, device="cuda", dtype=torch.long)
        # Fix the FP32 dot products once per image: changing GEMM batch shapes can
        # move nearly-zero logits across the binary threshold by a few pixels.
        logits = F.interpolate(p["_mask_logits"][idx][None], shape, mode="bilinear")[0]
        binary = ops.crop_mask(logits, boxes[idx]).gt_(0).byte()
        return ops.scale_masks(binary[None], orig)[0].byte()

    def support_envelope(boxes, shape, orig):
        # scale_masks removes padding before linear interpolation. Strictly-positive
        # 1-D support encloses the final byte-mask support, including edge rounding.
        ih, iw = shape
        oh, ow = orig
        gain = min(ih / oh, iw / ow)
        pw, ph = (iw - round(ow * gain)) / 2, (ih - round(oh * gain)) / 2
        left, top = round(pw-.1), round(ph-.1)
        right, bottom = iw-round(pw+.1), ih-round(ph+.1)
        extents = []
        for first, last, lo, hi, size in ((left,right,0,2,ow),(top,bottom,1,3,oh)):
            coords = torch.arange(first,last,device="cuda")[None]
            support = ((coords >= boxes[:,lo,None]) & (coords < boxes[:,hi,None])).float()
            expanded = F.interpolate(support[:,None], size=size, mode="linear", align_corners=False)[:,0] > 0
            valid = expanded.any(1)
            begin = expanded.int().argmax(1)
            end = size-expanded.flip(1).int().argmax(1)
            begin[~valid] = 0
            end[~valid] = 0
            extents.append((begin.cpu().numpy(),end.cpu().numpy()))
        return np.stack((extents[0][0],extents[1][0],extents[0][1],extents[1][1]),1)

    def boundary(mask, radius):
        padded = cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
        eroded = cv2.erode(padded,np.ones((3,3),np.uint8),iterations=radius)[1:-1,1:-1]
        return mask.astype(bool) & ~eroded.astype(bool)

    with torch.inference_mode():
        for number, image_id in enumerate(ids):
            image_start = time.monotonic()
            im = cv2.imread(str(a.root / f"assets/datasets/coco/images/{a.split}" / coco.imgs[image_id]["file_name"]))
            if im is None:
                raise RuntimeError(f"Missing image {image_id}")
            orig = im.shape[:2]
            params = transform.get_params({"img":im})
            resized = transform.apply_image({"img":im}, params)["img"]
            x = torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            shape = tuple(x.shape[2:])
            output, raw = model(x)
            p = raw["one2one"]
            proto = p["proto"][0]
            p["_mask_logits"] = (p["mask_coefficient"][0].T @ proto.float().flatten(1)).reshape(-1,*proto.shape[-2:])
            boxes = head._get_decode_boxes(p)[0].T
            orig_boxes = ops.scale_boxes(shape, boxes.clone(), orig)
            scores = p["scores"].permute(0,2,1).sigmoid()
            n = len(boxes)
            anns = sorted((v for v in coco.imgToAnns[image_id] if not v.get("iscrowd",0) and not v.get("ignore",0)),key=lambda v:v["id"])
            g = len(anns)
            gm = np.stack([coco.annToMask(v) for v in anns]) if g else np.zeros((0,*orig),np.uint8)
            areas = gm.sum((1,2)).astype(np.float64)
            if g and not (areas>0).all():
                raise RuntimeError("Empty ordinary GT mask requires explicit handling")
            gt_gpu = torch.from_numpy(gm.reshape(g,-1)).cuda().float() if g else torch.zeros((0,orig[0]*orig[1]),device="cuda")
            gt_boxes = torch.tensor([v["bbox"] for v in anns],device="cuda",dtype=torch.float32).reshape(-1,4)
            gt_boxes[:,2:] += gt_boxes[:,:2]
            b = box_iou(orig_boxes,gt_boxes).cpu().numpy()
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all() or not np.isfinite(b).all():
                raise RuntimeError("Nonfinite raw output")
            env = support_envelope(boxes,shape,orig)
            ub = np.zeros((n,g),np.float32)
            for gi in range(g):
                integral = cv2.integral(gm[gi],sdepth=cv2.CV_32S)
                x1,y1,x2,y2 = env.T
                ub[:,gi] = (integral[y2,x2]-integral[y1,x2]-integral[y2,x1]+integral[y1,x1])/areas[gi]
            m = np.full((n,g),np.nan,np.float32)
            pa = np.full(n,-1,np.int64)
            decoded = np.zeros(n,bool)

            def decode(indices):
                for k in range(0,len(indices),a.chunk):
                    ix = np.asarray(indices[k:k+a.chunk],dtype=np.int64)
                    masks = masks_for(p,boxes,ix,shape,orig)
                    flat = masks.flatten(1).float()
                    pixel_area = flat.sum(1)
                    intersections = flat @ gt_gpu.T
                    union = pixel_area[:,None] + torch.as_tensor(areas,device="cuda",dtype=torch.float32)[None]-intersections
                    values = (intersections/union.clamp(min=1)).cpu().numpy()
                    if g and (values > ub[ix]+2e-6).any():
                        raise RuntimeError("Crop envelope failed to upper-bound actual IoU")
                    m[ix] = values
                    pa[ix] = pixel_area.long().cpu().numpy()
                    decoded[ix] = True

            seed_ids = {0}
            if g:
                for gi in range(g):
                    seed_ids.update(np.argsort(-b[:,gi],kind="stable")[:8].tolist())
            decode(sorted(seed_ids))
            if a.exhaustive:
                decode(np.flatnonzero(~decoded))
            elif g:
                best = np.nanmax(m,axis=0)
                best_b = np.max(np.where((b>=.75)&np.isfinite(m),m,-1),axis=0)
                need = (ub>=.5-2e-6).any(1) | ((ub>0)&(ub+2e-6>best)).any(1)
                need |= ((b>=.75)&(ub>0)&(ub+2e-6>best_b)).any(1)
                # Process by spatial overlap, never classification scores.
                queue = np.flatnonzero(need&~decoded)
                queue = queue[np.argsort(-ub[queue].max(1),kind="stable")]
                for k in range(0,len(queue),a.chunk):
                    ix=queue[k:k+a.chunk]
                    keep=(ub[ix]>=.5-2e-6).any(1)|((ub[ix]>0)&(ub[ix]+2e-6>best)).any(1)
                    keep|=((b[ix]>=.75)&(ub[ix]>0)&(ub[ix]+2e-6>best_b)).any(1)
                    ix=ix[keep]
                    decode(ix)
                    if len(ix):
                        best=np.maximum(best,m[ix].max(0))
                        best_b=np.maximum(best_b,np.max(np.where(b[ix]>=.75,m[ix],-1),axis=0))
            geometry_decoded = int(decoded.sum())
            maxima_before = np.nanmax(m,axis=0) if g else np.zeros(0)
            constrained_before = np.max(np.where((b>=.75)&np.isfinite(m),m,-1),axis=0) if g else np.zeros(0)
            edges_before = np.nan_to_num(m,nan=-1)>=.5
            if a.check_exhaustive:
                decode(np.flatnonzero(~decoded))
                assert np.allclose(maxima_before,m.max(0),atol=1e-7)
                assert np.allclose(constrained_before,np.where(b>=.75,m,-1).max(0),atol=1e-7)
                assert np.array_equal(edges_before,m>=.5)

            assignments = {}
            primary_owner = np.zeros(n,np.int64)
            primary_fg = np.zeros(n,bool)
            if g:
                transformed = gt_boxes.clone()*torch.tensor(params["ratio"]*2,device="cuda")
                transformed += torch.tensor([params["left"],params["top"]]*2,device="cuda")
                labels = torch.tensor([cat_cls[v["category_id"]] for v in anns],device="cuda").float()[None,:,None]
                for branch in ("one2one","one2many"):
                    q=raw[branch]
                    anchors,strides=make_anchors(q["feats"],head.stride,.5)
                    qb=criteria[branch].bbox_decode(anchors,q["boxes"].permute(0,2,1))*strides
                    _,_,weights,fg,owner=criteria[branch].assigner(q["scores"].permute(0,2,1).sigmoid(),qb,anchors*strides,labels,transformed[None],torch.ones((1,g,1),device="cuda",dtype=torch.bool))
                    fg=fg[0].cpu().numpy().astype(bool); owner=owner[0].cpu().numpy()
                    w=weights[0].sum(1).cpu().numpy()
                    assignments[branch]=dict(foreground=fg,owner=owner,weight=w)
                    if branch=="one2one":
                        primary_fg,primary_owner=fg,owner
                        decode(np.flatnonzero(fg&~decoded))

            top_scores,top_labels,top_ids=head.get_topk_index(scores,300)
            ti=top_ids[0,:,0].cpu().numpy(); ts=top_scores[0,:,0].cpu().numpy(); tc=top_labels[0,:,0].long().cpu().numpy()
            assert np.allclose(output[0][0,:,:4].cpu().numpy(),boxes[ti].cpu().numpy(),atol=1e-4)
            retained=ti[ts>.001]
            decode(np.array(sorted(set(retained.tolist())-set(np.flatnonzero(decoded))),dtype=np.int64))
            # Additional supervised/output decoding must not change certified geometry.
            if g:
                assert np.allclose(maxima_before,np.nanmax(m,axis=0),atol=2e-6)
                assert np.allclose(constrained_before,np.where((b>=.75)&np.isfinite(m),m,-1).max(0),atol=2e-6)
            normal=[]
            mask_by_id={}
            for k in range(0,len(set(retained.tolist())),a.chunk):
                ix=np.array(sorted(set(retained.tolist()))[k:k+a.chunk])
                masks=masks_for(p,boxes,ix,shape,orig).cpu().numpy()
                for ri,mask in zip(ix,masks):
                    if mask.any():
                        rle=mu.encode(np.asfortranarray(mask));rle["counts"]=rle["counts"].decode("ascii")
                        mask_by_id[int(ri)]=rle
            ob=orig_boxes.cpu().numpy()
            for rank,(ri,score,ci) in enumerate(zip(ti,ts,tc)):
                if score<=.001 or int(ri) not in mask_by_id:
                    continue
                bb=ob[ri].copy();bb[2:]-=bb[:2]
                normal.append(dict(image_id=image_id,category_id=cls_cat[int(ci)],score=float(score),bbox=bb.tolist(),segmentation=mask_by_id[int(ri)],raw_id=int(ri),output_rank=rank))

            union_gt=gm.astype(bool).any(0) if g else np.zeros(orig,bool)
            crowd=np.zeros(orig,bool)
            for ann in coco.imgToAnns[image_id]:
                if ann.get("iscrowd",0) or ann.get("ignore",0): crowd|=coco.annToMask(ann).astype(bool)
            witnesses={}
            for gi in range(g):
                # Unexamined zero-upper-bound candidates have exact zero IoU.
                values=np.nan_to_num(m[:,gi],nan=0)
                mask_id=int(np.argmax(values));box_id=int(np.argmax(b[:,gi]))
                eligible=np.flatnonzero(b[:,gi]>=.75)
                joint_id=int(eligible[np.argmax(values[eligible])]) if len(eligible) else None
                for role,ri in (("best_mask",mask_id),("best_box",box_id),("best_mask_with_box75",joint_id)):
                    if ri is not None: witnesses.setdefault(ri,[]).append((gi,role))
            witness_stats={}
            radius=max(1,round(.02*math.hypot(*orig)))
            gt_boundary=[boundary(v,radius) for v in gm]
            witness_ids=sorted(witnesses)
            decode(np.array([i for i in witness_ids if not decoded[i]],dtype=np.int64))
            for k in range(0,len(witness_ids),a.chunk):
                ix=witness_ids[k:k+a.chunk]
                masks=masks_for(p,boxes,ix,shape,orig).cpu().numpy()
                ones=ops.process_mask(torch.ones_like(p["proto"][0,:1]),torch.ones((len(ix),1),device="cuda"),boxes[ix],shape,upsample=True)
                support=ops.scale_masks(ones[None],orig)[0].byte().cpu().numpy().astype(bool)
                for ri,mask,sp in zip(ix,masks,support):
                    mb=mask.astype(bool);pb=boundary(mask,radius)
                    for gi,role in witnesses[ri]:
                        gt=gm[gi].astype(bool);tp=int((mb&gt).sum());fp=int((mb&~gt).sum());fn=int((~mb&gt).sum())
                        gb=gt_boundary[gi]
                        stats=dict(raw_id=int(ri),box_iou=float(b[ri,gi]),mask_iou=tp/max(tp+fp+fn,1),tp=tp,fp=fp,fn=fn,
                            coverage=tp/int(areas[gi]),purity=tp/(tp+fp) if tp+fp else None,
                            crop_support=float((sp&gt).sum()/areas[gi]),boundary_iou=float((pb&gb).sum()/max((pb|gb).sum(),1)),
                            neighbor_fp=int((mb&~gt&union_gt).sum()),background_fp=int((mb&~union_gt&~crowd).sum()),crowd_fp=int((mb&~union_gt&crowd).sum()),
                            predicted_class=cls_cat[int(scores[0,ri].argmax())],gt_class_score=float(scores[0,ri,cat_cls[anns[gi]["category_id"]]]),
                            foreground=bool(primary_fg[ri]),owner_annotation_id=int(anns[int(primary_owner[ri])]["id"]) if primary_fg[ri] else None)
                        assert stats["mask_iou"]<=stats["crop_support"]+1e-6
                        if abs(stats["mask_iou"]-m[ri,gi])>=2e-6:
                            dump(a.out/"decode_consistency_failure.json",dict(image_id=image_id,raw_id=int(ri),gt_id=anns[gi]["id"],direct=stats,matrix_iou=float(m[ri,gi])))
                            raise RuntimeError("Repeated decoding changed the same raw mask")
                        witness_stats[(gi,role)]=stats

            instance_rows=[]
            for gi,ann in enumerate(anns):
                vals=np.nan_to_num(m[:,gi],nan=0)
                best_box=witness_stats[(gi,"best_box")];best_mask=witness_stats[(gi,"best_mask")]
                joint=witness_stats.get((gi,"best_mask_with_box75"))
                row=dict(image_id=image_id,annotation_id=ann["id"],category_id=ann["category_id"],area=ann["area"],mask_area=int(areas[gi]),image_gt_count=g,
                         box_max=float(b[:,gi].max()),mask_max=float(vals.max()),mask_given_box75=joint["mask_iou"] if joint else None,
                         geometry_state=state(float(b[:,gi].max()),float(vals.max()),bool(joint and joint["mask_iou"]>=.75)),
                         best_box=best_box,best_mask=best_mask,best_mask_with_box75=joint)
                for branch,ass in assignments.items():
                    own=ass["foreground"]&(ass["owner"]==gi)
                    row[branch+"_positive_count"]=int(own.sum())
                    row[branch+"_positive_weight"]=float(ass["weight"][own].sum())
                    if branch=="one2one":
                        row["assigned_mask_max"]=float(vals[own].max()) if own.any() else None
                        row["assigned_raw_ids"]=np.flatnonzero(own).tolist()
                instance_rows.append(row)
                rows_out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            matched={}
            for threshold in (.5,.75):
                key=str(threshold)
                me=np.nan_to_num(m,nan=-1)>=threshold
                matched[key]=dict(box=cardinality(b>=threshold),mask=cardinality(me),joint=cardinality(me&(b>=threshold)))
            np.savez_compressed(shards/f"{image_id:012d}.npz",boxes_input=boxes.cpu().numpy(),boxes_original=ob,
                scores=scores[0].cpu().numpy(),coefficients=p["mask_coefficient"][0].cpu().numpy(),proto=p["proto"][0].cpu().numpy(),
                box_ious=b,mask_ious=m,support_upper_bound=ub,decoded=decoded,mask_areas=pa,gt_ids=np.array([v["id"] for v in anns]),
                input_shape=np.array(shape),original_shape=np.array(orig),top_ids=ti,top_scores=ts,top_classes=tc,
                o2o_foreground=primary_fg,o2o_owner=primary_owner)
            dump(shards/f"{image_id:012d}.json",dict(image_id=image_id,predictions=normal,gt_rows=instance_rows,
                 matching=matched,letterbox=params,raw_count=n,geometry_decoded=geometry_decoded,all_decoded=int(decoded.sum()),
                 exhaustive_checked=a.check_exhaustive))
            total_gt+=g;total_raw+=n;total_decoded+=geometry_decoded
            progress=dict(images=number+1,total=len(ids),gt=total_gt,raw_candidates=total_raw,geometry_decoded=total_decoded,
                          elapsed_s=round(time.monotonic()-start,2),last_image=image_id,last_image_s=round(time.monotonic()-image_start,2),
                          gpu_peak_mib=round(torch.cuda.max_memory_allocated()/2**20,1))
            images_out.write(json.dumps({**progress,"matching":matched})+"\n");images_out.flush();rows_out.flush()
            dump(a.out/"progress.json",progress)
            print(json.dumps(progress),flush=True)
    rows_out.close();images_out.close()
    dump(a.out/"COMPLETE.json",{**progress,"version":ultralytics.__version__,"device":torch.cuda.get_device_name(),"exhaustive_check":a.check_exhaustive})


if __name__ == "__main__":
    main()
