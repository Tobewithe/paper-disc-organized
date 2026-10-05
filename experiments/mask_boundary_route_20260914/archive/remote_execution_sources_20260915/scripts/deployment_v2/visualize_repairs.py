"""Deterministic development examples: median repair and median damage, with GT.

GT selects illustrations after evaluation; it never selects a prediction threshold.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    for name in ("package-root","weights","images","annotations","records","output"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args()
    sys.path.insert(0,args.package_root)
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from mask_calibration import input_logits,export_masks,apply_threshold

    torch.set_num_threads(4)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    buckets={"Coverage-sufficient repair":[],"Other repair":[],"Damage":[]}
    with Path(args.records).open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if row["variant"]!="smooth_gated":
                continue
            before,after=float(row["baseline_iou"]),float(row["iou"])
            if before<.75<=after:
                mechanism=float(row["box_iou"])>=.75 and float(row["baseline_recall"])>=.95
                buckets["Coverage-sufficient repair" if mechanism else "Other repair"].append(row)
            elif after<.75<=before:
                buckets["Damage"].append(row)
    chosen=[]
    for title,rows in buckets.items():
        rows.sort(key=lambda r:(abs(float(r["iou"])-float(r["baseline_iou"])),int(r["image_id"]),int(r["annotation_id"])))
        if rows:
            chosen.append((title,rows[len(rows)//2]))
    assert chosen,"No illustrative transitions"
    coco=COCO(args.annotations)
    holder={}

    class Capture(SegmentationPredictor):
        def construct_result(self,pred,img,orig_img,img_path,proto):
            holder.update(pred=pred.detach(),shape=img.shape[2:],proto=proto.detach(),orig_img=orig_img)
            return Results(orig_img,path=img_path,names=self.model.names,boxes=pred[:,:6])

    model=YOLO(args.weights)
    model.model.model[-1].end2end=True
    fig,axes=plt.subplots(len(chosen),4,figsize=(12,3.8*len(chosen)),squeeze=False,layout="constrained")
    records=[]
    for line,(title,row) in enumerate(chosen):
        ann=coco.anns[int(row["annotation_id"])]
        file=Path(args.images)/coco.imgs[ann["image_id"]]["file_name"]
        model.predict(str(file),predictor=Capture,device=0,imgsz=640,conf=.001,max_det=300,half=False,batch=1,verbose=False,save=False)
        index=int(row["candidate_index"])
        pred=holder["pred"]
        # Use the same chunk boundaries as the recorded experiment.
        first=(index//24)*24
        group=pred[first:first+24]
        with torch.inference_mode():
            logits=input_logits(holder["proto"],group[:,6:],group[:,:4],holder["shape"])
            original_shape=holder["orig_img"].shape[:2]
            baseline=export_masks((logits>0).byte(),original_shape)
            area=baseline.sum((1,2)).float()
            method=export_masks(apply_threshold(logits,area,"smooth"),original_shape)
            b=baseline[index-first].cpu().numpy().astype(bool)
            m=method[index-first].cpu().numpy().astype(bool)
        gt=coco.annToMask(ann).astype(bool)
        def metrics(mask):
            tp=int((mask & gt).sum())
            return dict(iou=tp/max(int((mask|gt).sum()),1),recall=tp/max(int(gt.sum()),1),purity=tp/max(int(mask.sum()),1))
        bm,mm=metrics(b),metrics(m)
        assert abs(bm["iou"]-float(row["baseline_iou"]))<1e-6,"Baseline candidate differs from recorded evaluation"
        assert abs(mm["iou"]-float(row["iou"]))<1e-6,"Calibrated candidate differs from recorded evaluation"
        image=holder["orig_img"][:,:,::-1].copy()
        x,y,w,h=ann["bbox"]
        padx,pady=max(w*.35,12),max(h*.35,12)
        x1,y1=max(0,int(x-padx)),max(0,int(y-pady))
        x2,y2=min(image.shape[1],int(x+w+padx)),min(image.shape[0],int(y+h+pady))
        def overlay(mask):
            rgb=image.astype(float)
            for region,color in (((mask & gt),(35,170,105)),((mask & ~gt),(225,65,45)),((gt & ~mask),(40,110,235))):
                rgb[region]=.35*rgb[region]+.65*np.asarray(color)
            return rgb.astype(np.uint8)
        panels=[image,overlay(gt),overlay(b),overlay(m)]
        names=[f"{title}\nImage {ann['image_id']}, GT {ann['id']}","Target GT",
               f"Baseline\nIoU {bm['iou']:.3f}, recall {bm['recall']:.3f}",
               f"Smooth scale\nIoU {mm['iou']:.3f}, recall {mm['recall']:.3f}"]
        for ax,panel,name in zip(axes[line],panels,names):
            ax.imshow(panel[y1:y2,x1:x2]); ax.set_title(name,fontsize=10); ax.axis("off")
        records.append(dict(selection_group=title,eligible=len(buckets[title]),image_id=ann["image_id"],
            annotation_id=ann["id"],candidate_index=index,baseline=bm,method=mm,crop_xyxy=[x1,y1,x2,y2]))
    fig.suptitle("Median-change development examples | green: TP, red: FP, blue: FN",fontsize=13)
    for ext in ("png","pdf"):
        fig.savefig(out/f"repair_damage_examples.{ext}",dpi=180)
    plt.close(fig)
    (out/"SUMMARY.json").write_text(json.dumps(dict(run_id=os.environ.get("RESEARCH_RUN_ID"),
        selection="Median absolute IoU change within each development transition group; tie break image and annotation ID",
        records=records,limitations=["Illustrative examples, not a population estimate", "GT used only for post-evaluation selection and pixel diagnostics"]),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
