"""S070: exact decoder support and GT-crop intervention for joint failures.

Unlike S069's intersection of final masks, this reruns the existing decoder
with its original raw mask response and a GT tight crop box. No coefficients,
features, logits, classes, scores or candidate selection are optimized.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "local_readout_runtime_20260912"
sys.path.insert(0, str(RUNTIME / "vendor"))
os.environ["YOLO_CONFIG_DIR"] = str(RUNTIME / "settings")
if os.name == "nt":
    os.environ["PATH"] = str(Path(sys.prefix) / "Library/bin") + os.pathsep + os.environ.get("PATH", "")
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(key, "4")

import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import shutil
import time
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops
from structure_candidate_trace import TraceCapture
from mask_error_ap_probe import (COCO, mu, ANNOTATION, CENSUS, dump, read_gz,
    save_gz, save_csv, encode, sha, evaluate, paired_stats)

GOLD = ROOT / "diagnostics/frozen_readouts_fullval_20260912_v2"
PRIOR = ROOT / "diagnostics/mask_error_ap_20260913"
RESIDUAL = ROOT / "diagnostics/failure_dimension_analysis_20260913/joint_failure_residuals.csv"


def rank(aid):
    return hashlib.sha256(f"S070:20260913:{aid}".encode()).hexdigest()


def restore(binary, shape):
    return (ops.scale_masks(binary[:, None], shape)[:, 0] > .5).cpu().numpy()


def tight_input_box(own, gain, left, top, device):
    yy, xx = np.nonzero(own)
    assert len(xx)
    b = [xx.min(), yy.min(), xx.max()+1, yy.max()+1]
    return torch.tensor(b, device=device, dtype=torch.float32)*gain + torch.tensor([left,top,left,top],device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-images", type=int, default=0)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ["patches", "tensor_cache", "source"]:
        (out/name).mkdir()
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with CENSUS.open(encoding="utf-8-sig") as f:
        metadata = {int(r["annotation_id"]): r for r in csv.DictReader(f)}
    rd = pd.read_csv(RESIDUAL).set_index("annotation_id")
    targets = {aid:r for aid,r in metadata.items() if r["state"] in ["box_bad__mask_bad", "box_bad__mask_good"]}
    assert len(targets) == 7690
    groups = {}
    for aid,r in targets.items():
        residual = str(rd.loc[aid,"residual_state"]) if aid in rd.index else "mask_good_control"
        groups[aid] = f"{r['mask_density']}:{residual}"
    # Fixed, outcome-stratified tensor sample, frozen before this intervention.
    # Prior S069 state may define a diagnostic stratum; no S070 outcome is used.
    sample = []
    for density in ["low", "high"]:
        for state in ["requires_target_pixel_recovery", "sufficient_true_pixels_but_residual_false_pixels", "mask_good_control"]:
            eligible=[aid for aid,r in targets.items() if groups[aid]==f"{density}:{state}"
                      and (state=="mask_good_control" or (r["official_mask75"]=="False" and r["any_kept_mask75"]=="False"))]
            sample.extend(sorted(eligible,key=rank)[:24])
    sample=set(sample)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANNOTATION))
    byimage = defaultdict(list)
    for aid,r in targets.items(): byimage[int(r["image_id"])].append(aid)
    image_ids=sorted(byimage)[:a.limit_images or None]
    selected=set(image_ids)
    sample={aid for aid in sample if int(targets[aid]["image_id"]) in selected}
    dump(out/"tensor_manifest.json",[dict(annotation_id=aid,image_id=int(targets[aid]["image_id"]),group=groups[aid]) for aid in sorted(sample)])
    weight=ROOT/"weights/yolo26m-seg.pt"
    assert sha(weight)=="16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
    assert ultralytics.__version__=="8.4.143"
    dump(out/"protocol.json",dict(experiment="S070_EXACT_DECODER_JOINT_FAILURE",training=False,
        question="Does correcting the actual input-resolution crop recover missing own pixels and task performance in joint box/mask failures? What residual errors and source-cell positions remain?",
        decision="Strong crop-only recovery prioritizes support prediction; weak recovery plus raw missing positives prioritizes response formation; source-cell ownership is associative triage, not proof.",
        incremental_to_S069="S069 intersected final original-size binary masks and could never add true pixels. S070 holds raw c/P fixed and changes actual decoder crop, allowing originally cut positive response to return.",
        selected_images=len(image_ids), limit_images=a.limit_images, joint_failure_targets=6584, poor_box_good_mask_controls=1106,
        base_branch="one-to-many, end2endFalse, original frozen S036 candidate slots",
        inference=dict(imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,half=False,retina_masks=False,tf32=False),
        gt_crop="tight integer bounding rectangle of original COCO raster mask -> input coordinates using same gain and integer letterbox pad; no padding tuned against IoU",
        decoder="Fixed full-batch c @ P; bilinear640, >0, crop, binary scale_masks to original, >.5. All source c/P and original RLE must replay exactly.",
        support="All-ones positive 640 binary mask through identical crop+scale path gives exact output support. Loss of target TP due crop = raw_uncropped_TP-original_TP; rest = ownGT-raw_uncropped_TP.",
        source_cell="Read actual head.anchors*head.strides. Assign center and stride-square footprint in nearest-exact GT640; footprint is NOT receptive field or training assigner identity.",
        tensor_sample="24 per low/high x prior FN/FP/mask-good stratum, SHA-ranked annotation IDs; no new outcome selection. Failures without any retained Mask75 only for saved tensor subset, all7690 measured.",
        statistics="GT-assisted diagnostic, explored val; no novelty/method/causal-density claim. One deterministic pretrained checkpoint; no training-seed experiment.",
        runtime=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,ultralytics_path=ultralytics.__file__,gpu=torch.cuda.get_device_name()),
        fingerprints=dict(script=sha(__file__),ops=sha(Path(ops.__file__)),census=sha(CENSUS),weight=sha(weight))))
    for file in [Path(__file__),Path(ops.__file__),ROOT/"structure_candidate_trace.py",ROOT/"frozen_mechanism_probe.py",ROOT/"mask_error_ap_probe.py"]:
        shutil.copy2(file,out/"source"/file.name)
    model=YOLO(str(weight));model.model.eval().requires_grad_(False)
    head=model.model.model[-1]
    assert not head.end2end
    feature_maps,handles={},[]
    for level,branch in enumerate(head.cv4):
        def capture(module,args,output,level=level): feature_maps[level]=args[0].detach()
        handles.append(branch[-1].register_forward_hook(capture))
    categories=sorted(gt.cats)
    rows,witness=[],[]
    for number,iid in enumerate(image_ids,1):
        feature_maps.clear()
        image=RUNTIME/"data/images/val2017"/gt.imgs[iid]["file_name"]
        with torch.inference_mode():
            model.predict(str(image),predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,
                max_det=300,half=False,retina_masks=False,device=0,verbose=False)
            cap,raw=model.predictor.capture,model.predictor.dense
            # Predictor setup deep-copies the model in this vendored version.
            # Read anchors from the model that actually executed this forward,
            # not from the weight wrapper's stale serialized anchor buffers.
            live_head=model.predictor.model.backend.model.model[-1]
            assert live_head.anchors.shape[1]==raw.shape[-1]==8400
            golden=read_gz(GOLD/"images"/f"{iid}.json.gz")
            assert cap["input_shape"]==(640,640)
            c,proto,boxes=cap["coeff"],cap["proto"].float(),cap["boxes"]
            assert len(c)==len(golden["original"])
            # Whole original batch retains the original GEMM numerical path.
            z=F.interpolate((c@proto.flatten(1)).reshape(1,len(c),*proto.shape[-2:]),(640,640),mode="bilinear",align_corners=False)[0]
            raw_binary=(z>0).byte()
            original_binary=ops.crop_mask(raw_binary.clone(),boxes)
            shape=cap["shape"]
            gain=min(640/shape[0],640/shape[1]);nh,nw=round(shape[0]*gain),round(shape[1]*gain)
            top,left=round((640-nh)/2-.1),round((640-nw)/2-.1)
            masks={ann["id"]:gt.annToMask(ann).astype(bool) for ann in gt.imgToAnns[iid]}
            mask_input={}
            for ann in gt.imgToAnns[iid]:
                canvas=np.zeros((640,640),bool)
                canvas[top:top+nh,left:left+nw]=cv2.resize(masks[ann["id"]].astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST_EXACT)>0
                mask_input[ann["id"]]=canvas
            patches={}
            cache_aids=[];cache_c=[];cache_b=[];cache_gb=[];cache_h=[];cache_source=[]
            for aid in byimage[iid]:
                r=targets[aid];slot,source=int(r["prediction_slot"]),int(r["source_index"])
                pred=golden["original"][slot]
                assert golden["witness"]["source_indices"][slot]==source
                assert torch.equal(raw[0,84:,source],c[slot])
                assert int(r["category_id"])==categories[int(cap["detections"][slot,5])]
                assert abs(float(cap["detections"][slot,4])-pred["score"])<1e-12
                original=restore(original_binary[slot:slot+1],shape)[0]
                assert np.array_equal(original,mu.decode(pred["segmentation"]).astype(bool)),(iid,aid,"baseline RLE differs")
                own=masks[aid];area=int(own.sum());assert area
                gtbox=tight_input_box(own,gain,left,top,c.device)
                changed_binary=ops.crop_mask(raw_binary[slot:slot+1].clone(),gtbox[None])
                changed=restore(changed_binary,shape)[0]
                raw_mask=restore(raw_binary[slot:slot+1],shape)[0]
                support=restore(ops.crop_mask(torch.ones((1,640,640),device=c.device,dtype=torch.uint8),boxes[slot:slot+1]),shape)[0]
                gt_support=restore(ops.crop_mask(torch.ones((1,640,640),device=c.device,dtype=torch.uint8),gtbox[None]),shape)[0]
                assert not (original & ~support).any() and not (original & ~raw_mask).any()
                tp=int((original&own).sum());rtp=int((raw_mask&own).sum());ntp=int((changed&own).sum())
                original_iou=tp/max(int((original|own).sum()),1)
                assert abs(original_iou-float(r["mask_iou"]))<1e-10
                crop_fn=rtp-tp;response_fn=area-rtp
                assert crop_fn>=0 and crop_fn+response_fn==area-tp
                stride=float(live_head.strides[0,source]);center=(live_head.anchors[:,source]*stride).cpu().numpy()
                ax,ay=map(float,center);px,py=int(np.floor(ax)),int(np.floor(ay))
                x0,x1=int(round(ax-stride/2)),int(round(ax+stride/2));y0,y1=int(round(ay-stride/2)),int(round(ay+stride/2))
                assert 0<=x0<x1<=640 and 0<=y0<y1<=640
                own_in=mask_input[aid]
                same=np.zeros((640,640),bool);other=np.zeros_like(same);crowd=np.zeros_like(same)
                for ann in gt.imgToAnns[iid]:
                    if ann.get("iscrowd",0) or ann.get("ignore",0):crowd|=mask_input[ann["id"]]
                    elif ann["id"]!=aid:
                        if ann["category_id"]==int(r["category_id"]):same|=mask_input[ann["id"]]
                        else:other|=mask_input[ann["id"]]
                location="own" if own_in[py,px] else "same_neighbor" if same[py,px] else "other_class" if other[py,px] else "crowd" if crowd[py,px] else "background"
                new_iou=ntp/max(int((changed|own).sum()),1)
                rows.append(dict(annotation_id=aid,image_id=iid,slot=slot,source_index=source,
                    category_id=int(r["category_id"]),area=float(r["area"]),area_bin=r["area_bin"],density=r["mask_density"],
                    original_state=r["state"],prior_residual=groups[aid].split(":",1)[1],official_mask75=r["official_mask75"]=="True",any_retained_mask75=r["any_kept_mask75"]=="True",
                    box_iou=float(r["box_iou"]),original_iou=original_iou,gt_crop_iou=new_iou,delta_iou=new_iou-original_iou,
                    original75=original_iou>=.75,gt_crop75=new_iou>=.75,
                    gt_area=area,original_tp=tp,raw_tp=rtp,gt_crop_tp=ntp,exact_support_coverage=float((support&own).sum()/area),gt_support_coverage=float((gt_support&own).sum()/area),
                    crop_fn=crop_fn,response_fn=response_fn,crop_fn_fraction=crop_fn/area,response_fn_fraction=response_fn/area,
                    source_stride=stride,source_x=ax,source_y=ay,source_center_location=location,
                    source_cell_own_fraction=float(own_in[y0:y1,x0:x1].mean()),
                    source_cell_same_fraction=float((same&~own_in)[y0:y1,x0:x1].mean()),
                    source_cell_other_fraction=float((other&~own_in&~same)[y0:y1,x0:x1].mean())))
                if r["state"]=="box_bad__mask_bad":patches[str(slot)]=encode(changed)
                if aid in sample:
                    level={8:0,16:1,32:2}[int(stride)];offset=[0,6400,8000][level];idx=source-offset
                    hidden=feature_maps[level][0].flatten(1).T[idx]
                    cache_aids.append(aid);cache_c.append(c[slot].cpu().numpy());cache_b.append(boxes[slot].cpu().numpy());cache_gb.append(gtbox.cpu().numpy());cache_h.append(hidden.cpu().numpy());cache_source.append(source)
            if patches:save_gz(out/"patches"/f"{iid}.json.gz",patches)
            if cache_aids:
                np.savez_compressed(out/"tensor_cache"/f"{iid}.npz",annotation_ids=np.array(cache_aids),source_indices=np.array(cache_source),
                    coeff=np.array(cache_c),boxes=np.array(cache_b),gt_boxes=np.array(cache_gb),h=np.array(cache_h),proto=proto.cpu().numpy(),shape=np.array(shape),input_shape=np.array([640,640]))
            witness.append(dict(image_id=iid,targets=len(byimage[iid]),baseline_exact=True,source_exact=True,tp_partition_exact=True))
        if number%100==0 or number==len(image_ids):
            status=dict(stage="DECODE",images=number,total=len(image_ids),targets=len(rows),seconds=round(time.monotonic()-start,2),pid=os.getpid())
            dump(out/"progress.json",status);print(json.dumps(status),flush=True)
            save_csv(out/"instance_effects.csv",rows)
    for handle in handles:handle.remove()
    del model,cap,raw,feature_maps
    torch.cuda.empty_cache()
    save_csv(out/"instance_effects.csv",rows);save_csv(out/"replay_witness.csv",witness)
    if not a.limit_images:
        assert len(rows)==7690
        allids=sorted(gt.imgs);predictions=[]
        for iid in allids:
            pp=read_gz(PRIOR/"images"/f"{iid}.json.gz")["original"]
            path=out/"patches"/f"{iid}.json.gz";patch=read_gz(path) if path.exists() else {}
            predictions.extend(dict(pred,segmentation=patch[str(j)]) if str(j) in patch else pred for j,pred in enumerate(pp))
        assert len(predictions)==446097
        dump(out/"progress.json",dict(stage="COCO_EVALUATE",seconds=time.monotonic()-start,pid=os.getpid()))
        summary,records=evaluate(gt,predictions,allids,metadata,"joint_failure_actual_gt_crop")
        ref=json.loads((PRIOR/"original.json").read_text())["summary"]
        summary["delta_ap_points"]=100*(summary["mask_ap"]-ref["mask_ap"])
        summary["delta_gap_points"]=100*(summary["gap_low_high"]-ref["gap_low_high"])
        dump(out/"task_summary.json",summary);save_csv(out/"gt_recovery.csv",records)
        del predictions;gc.collect()
        base=pd.read_csv(PRIOR/"original_gt.csv").to_dict("records")
        dump(out/"paired_stats.json",paired_stats(base,records))
        print(json.dumps(summary),flush=True)
    dump(out/"COMPLETE.json",dict(status="COMPLETE",images=len(image_ids),targets=len(rows),cached_targets=len(sample),
        seconds=time.monotonic()-start,full_val_task_evaluated=not bool(a.limit_images),instance_effects_sha256=sha(out/"instance_effects.csv")))
    dump(out/"progress.json",dict(stage="COMPLETE",seconds=time.monotonic()-start))


if __name__=="__main__":main()
