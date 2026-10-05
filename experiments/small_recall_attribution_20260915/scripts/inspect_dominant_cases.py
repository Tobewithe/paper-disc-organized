"""Inspect two highest positive AR-contribution GTs in immutable exports."""
import argparse,contextlib,csv,gzip,io,json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu
from PIL import Image,ImageDraw,ImageFont

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--study",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();cfg=json.loads((args.study/"protocol.json").read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(cfg["annotation"])
    selected=[1983608,1128490]
    anns=[gt.anns[i] for i in selected]
    retained={};details=[];nearby=[]
    for arm,root in cfg["source_evaluations"].items():
        with gzip.open(Path(root)/"exports/one2one/predictions.json.gz","rt") as f:pred=json.load(f)
        for ann in anns:
            rows=[r for r in pred if r["image_id"]==ann["image_id"]]
            same=sorted([r for r in rows if r["category_id"]==ann["category_id"]],key=lambda r:-r["score"])[:100]
            rle=gt.annToRLE(ann)
            box=np.asarray(mu.iou([r["bbox"] for r in rows],[ann["bbox"]],[0])).reshape(-1) if rows else np.array([])
            masks=np.asarray(mu.iou([r["segmentation"] for r in rows],[rle],[0])).reshape(-1) if rows else np.array([])
            same_b=np.asarray(mu.iou([r["bbox"] for r in same],[ann["bbox"]],[0])).reshape(-1) if same else np.array([])
            same_m=np.asarray(mu.iou([r["segmentation"] for r in same],[rle],[0])).reshape(-1) if same else np.array([])
            chosen=same[int(same_m.argmax())] if same else None
            best_any_mask=int(masks.argmax()) if rows else None
            shown=chosen if chosen else (rows[best_any_mask] if rows and masks[best_any_mask]>0 else None)
            retained[(arm,ann["id"])]=shown
            best_any=int(box.argmax()) if rows else None
            row=dict(arm=arm,annotation_id=ann["id"],image_id=ann["image_id"],category=gt.cats[ann["category_id"]]["name"],
                     gt_area=ann["area"],same_class_exported_count=len(same),image_predictions=len(rows),
                     mask_iou=float(same_m.max()) if same else 0.,best_box_iou=float(same_b.max()) if same else 0.,
                     score=float(chosen["score"]) if chosen else 0.,
                     any_class_best_box_iou=float(box.max()) if rows else 0.,
                     any_class_best_box_category=gt.cats[rows[best_any]["category_id"]]["name"] if rows else "",
                     any_class_best_box_score=float(rows[best_any]["score"]) if rows else 0.,
                     any_class_best_mask_iou=float(masks.max()) if rows else 0.,
                     any_class_best_mask_category=gt.cats[rows[best_any_mask]["category_id"]]["name"] if rows else "",
                     any_class_best_mask_score=float(rows[best_any_mask]["score"]) if rows else 0.,
                     displayed_category=gt.cats[shown["category_id"]]["name"] if shown else "",
                     displayed_mask_iou=float(mu.iou([shown["segmentation"]],[rle],[0])[0,0]) if shown else 0.,
                     displayed_score=float(shown["score"]) if shown else 0.)
            details.append(row)
            for k in np.argsort(-box)[:5]:
                nearby.append(dict(arm=arm,annotation_id=ann["id"],category=gt.cats[rows[k]["category_id"]]["name"],
                                   score=rows[k]["score"],box_iou=float(box[k]),mask_iou=float(masks[k]),bbox=rows[k]["bbox"]))
        del pred
    for name,rows in [("dominant_cases.csv",details),("nearby_predictions.csv",nearby)]:
        with (args.out/name).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    canvas=Image.new("RGB",(1320,880),(246,248,252));draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",16)
    except OSError:font=ImageFont.load_default()
    draw.text((15,10),"Largest positive contributors to COCO small Mask AR (one-to-one, last)",fill=(20,30,45),font=font)
    for ri,ann in enumerate(anns):
        info=gt.imgs[ann["image_id"]]
        image=Image.open(Path(cfg["annotation"]).parents[1]/"images/val2017"/info["file_name"]).convert("RGB")
        own=mu.decode(gt.annToRLE(ann)).astype(bool)
        x,y,w,h=ann["bbox"];side=max(w,h)*4
        crop=(max(0,int(x+w/2-side/2)),max(0,int(y+h/2-side/2)),
              min(info["width"],int(x+w/2+side/2)),min(info["height"],int(y+h/2+side/2)))
        label=f"{gt.cats[ann['category_id']]['name']} | GT {ann['id']} | image {ann['image_id']} | area {ann['area']:.1f}"
        top=50+ri*410;draw.text((15,top),label,fill=(20,30,45),font=font)
        for ci,arm in enumerate(["GT","baseline","reg_only","reg_scheduled"]):
            base=np.array(image).astype(float)
            overlay=own if arm=="GT" else (mu.decode(retained[(arm,ann["id"])]["segmentation"]).astype(bool) if retained[(arm,ann["id"])] else np.zeros_like(own))
            base[overlay]=.6*base[overlay]+.4*np.array([35,190,90] if arm=="GT" else [250,105,40])
            im=Image.fromarray(base.astype(np.uint8))
            d=ImageDraw.Draw(im);d.rectangle([x,y,x+w,y+h],outline=(30,230,110),width=1)
            im=im.crop(crop).resize((310,310),Image.Resampling.NEAREST)
            left=15+ci*327;canvas.paste(im,(left,top+48))
            draw.text((left,top+25),arm,fill=(20,30,45),font=font)
            if arm=="GT":description="Green: GT; shared crop"
            else:
                stat=next(r for r in details if r["arm"]==arm and r["annotation_id"]==ann["id"])
                description=f"IoU {stat['displayed_mask_iou']:.3f} | score {stat['displayed_score']:.5f}"
                if not stat["same_class_exported_count"]:
                    description="No overlapping mask" if not stat["displayed_category"] else "Wrong class: "+stat["displayed_category"]
                draw.text((left,top+386),f"shown: {stat['displayed_category'] or 'none'}",fill=(20,30,45),font=font)
            draw.text((left,top+365),description,fill=(20,30,45),font=font)
        image.save(args.out/(str(ann["image_id"])+"_source.jpg"))
    canvas.save(args.out/"dominant_cases.png")
    (args.out/"COMPLETE.json").write_text(json.dumps(dict(status="complete",selected_annotation_ids=selected,
                  selection="Two largest positive single-GT contributions to one2one small AR versus baseline; post hoc descriptive examples. Followup measures class-agnostic masks to distinguish category-conditioned absence from geometry absence.",
                  sources=cfg["source_evaluations"],training=False,inference=False,
                  limitation="Saved predictions only; missing class cannot distinguish below-confidence, below-top-k, wrong class or raw geometry absence."),indent=2))
    print(json.dumps(details),flush=True)

if __name__=="__main__":main()
