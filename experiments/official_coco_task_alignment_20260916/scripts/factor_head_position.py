"""Controlled head x source-position coefficient replacement, fixed prototype/crop."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.root/"shared/vendor/ultralytics_8_4_100"))
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    from pycocotools.coco import COCO
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    coco=COCO(str(a.root/"assets/datasets/coco/annotations/instances_val2017.json"))
    official={int(r["annotation_id"]):r["matched"] for r in json.loads((a.cache/"official_gt_mask75.json").read_text()) if not r["ignore"]}
    instances=json.loads((a.probe/"instances.json").read_text())
    subjects=[r for r in instances if r["branch"]=="one2one"]
    cp={}
    with (a.probe/"candidates.jsonl").open() as f:
        for line in f:
            r=json.loads(line)
            if r["branch"]=="one2many" and r["ownership"]=="own_gt":
                cp.setdefault(r["annotation_id"],[]).append(r)
    model=YOLO(str(a.root/"assets/models/coco_clean_20260911/yolo26m-seg.pt")).model.cuda().float().eval()
    head=model.model[-1]; lb=LetterBox((640,640),auto=True,stride=32)
    rows=[];fig_examples=[]
    with torch.inference_mode():
        for s in subjects:
            im=cv2.imread(str(a.root/"assets/datasets/coco/images/val2017"/coco.imgs[s["image_id"]]["file_name"]))
            image=lb(image=im)
            tensor=torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            _,raw=model(tensor)
            o=raw["one2one"];m=raw["one2many"]
            ref=s["reference_raw_index"]
            source=max(cp[s["annotation_id"]],key=lambda x:x["fixed_mask_iou"])
            src=source["raw_index"]
            fixed_box=head._get_decode_boxes(o)[0,:,ref][None].expand(4,4)
            coeff=torch.stack((o["mask_coefficient"][0,:,ref],m["mask_coefficient"][0,:,ref],
                               o["mask_coefficient"][0,:,src],m["mask_coefficient"][0,:,src]))
            binary=ops.process_mask(o["proto"][0],coeff,fixed_box,tensor.shape[2:],upsample=True)
            masks=ops.scale_masks(binary[None],im.shape[:2])[0].byte().bool()
            gt=torch.from_numpy(coco.annToMask(coco.anns[s["annotation_id"]]).astype(bool)).cuda()
            tp=(masks & gt).sum((1,2)).float();pa=masks.sum((1,2)).float();ga=gt.sum().float()
            ious=(tp/(pa+ga-tp)).cpu().tolist()
            assert abs(ious[0]-s["reference_mask_iou"])<.003
            assert abs(ious[3]-source["fixed_mask_iou"])<.003
            row=dict(annotation_id=s["annotation_id"],image_id=s["image_id"],cohort=s["cohort"],pair_id=s["pair_id"],
                     official_mask75_matched=official[s["annotation_id"]],reference_raw_index=ref,source_raw_index=src,
                     source_stride=source["stride"],reference_stride=s["reference_stride"],
                     A_o2o_original=ious[0],B_o2m_original=ious[1],C_o2o_teacher_position=ious[2],D_o2m_teacher_position=ious[3],
                     head_at_original=ious[1]-ious[0],position_in_o2o=ious[2]-ious[0],
                     head_at_teacher=ious[3]-ious[2],interaction=ious[3]-ious[2]-ious[1]+ious[0],
                     coverage=(tp/ga).cpu().tolist(),purity=(tp/pa.clamp(min=1)).cpu().tolist())
            rows.append(row)
            if s["cohort"]=="failure" and not official[s["annotation_id"]] and ious[0]<.75<=ious[3]:
                fig_examples.append((s,im,gt.cpu().numpy(),masks.cpu().numpy(),ious))
    rng=np.random.default_rng(16);groups={}
    metrics=["A_o2o_original","B_o2m_original","C_o2o_teacher_position","D_o2m_teacher_position"]
    for name,group in (("all_failures",[r for r in rows if r["cohort"]=="failure"]),
                       ("official_task_failures",[r for r in rows if r["cohort"]=="failure" and not r["official_mask75_matched"]]),
                       ("controls",[r for r in rows if r["cohort"]=="control"])):
        g={"n":len(group)}
        for key in metrics+["head_at_original","position_in_o2o","head_at_teacher","interaction"]:
            v=np.array([r[key] for r in group]);means=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)
            g[key]={"mean":float(v.mean()),"bootstrap95":list(map(float,np.quantile(means,[.025,.975])))}
            if key in metrics:
                g[key].update(mask75=int((v>=.75).sum()),repaired=sum(r["A_o2o_original"]<.75<=r[key] for r in group),
                              broken=sum(r[key]<.75<=r["A_o2o_original"] for r in group))
        groups[name]=g
    result=dict(groups=groups,instances=rows,
                controls="Same original image, shared prototype, original O2O crop, same GT; cross branch head and source position only",
                limitation="Source position selected by GT among native O2M positives; selection favors D; no historical training-cause claim or learned method claim")
    (a.out/"SUMMARY.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if fig_examples:
        metadata=[]
        for row,(s,im,gt,masks,ious) in enumerate(fig_examples):
            x,y,w,h=coco.anns[s["annotation_id"]]["bbox"];pad=max(w,h)*.35+8
            x1=max(0,int(x-pad));x2=min(im.shape[1],int(x+w+pad));y1=max(0,int(y-pad));y2=min(im.shape[0],int(y+h+pad))
            np.savez_compressed(a.out/f"example_{row}.npz",image=im[y1:y2,x1:x2,::-1],gt=gt[y1:y2,x1:x2],masks=masks[:,y1:y2,x1:x2])
            metadata.append(dict(file=f"example_{row}.npz",annotation_id=s["annotation_id"],ious=ious))
        (a.out/"examples.json").write_text(json.dumps(metadata),encoding="utf-8")
    print(json.dumps(groups),flush=True)


if __name__=="__main__":main()
