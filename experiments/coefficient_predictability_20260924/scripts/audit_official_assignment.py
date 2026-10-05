"""Replay official loss assignment on frozen images and audit geometric mapping."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
import ultralytics


def main(a):
    assert ultralytics.__version__ == "8.4.100"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    coco=COCO(str(a.data/"annotations/instances_val2017.json"))
    cat_to_cls={cat:i for i,cat in enumerate(sorted(coco.cats))}
    wrapper=YOLO(str(a.weights))
    model=wrapper.model.cuda().float().eval()
    model.args=SimpleNamespace(**wrapper.ckpt["train_args"])
    criteria=model.init_criterion()
    branch_criteria={"one2one":criteria.one2one,"one2many":criteria.one2many}
    transform=LetterBox((640,640),auto=False,stride=32)
    split=json.loads(a.split.read_text())
    captured={}
    handles=[]
    for name,criterion in branch_criteria.items():
        handles.append(criterion.assigner.register_forward_hook(
            lambda _, inp, result, name=name:captured.__setitem__(name,result)))
    counts=Counter()
    comparisons=[]
    assignment_rows=[]
    for number,iid in enumerate(split["val"],1):
        image=torch.load(a.bank/"images"/f"{iid:012d}.pt",weights_only=False,map_location="cpu")
        im=cv2.imread(str(a.data/"images/val2017"/coco.imgs[iid]["file_name"]))
        shaped=transform(image=im)
        x=torch.from_numpy(np.ascontiguousarray(shaped[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
        with torch.no_grad():
            _,raw=model(x)
        delta=float((raw["one2one"]["mask_coefficient"][0].T.cpu()-image["coeff"]).abs().max())
        assert delta<1e-5,(iid,delta)
        anns=[r for r in coco.imgToAnns[iid] if not r.get("iscrowd",0) and not r.get("ignore",0)]
        counts["images"]+=1
        counts["gt"]+=len(anns)
        if not anns:
            continue
        boxes=torch.tensor([r["bbox"] for r in anns],device="cuda",dtype=torch.float32)
        boxes[:,2:]+=boxes[:,:2]
        boxes*=image["gain"]
        boxes+=torch.tensor([image["left"],image["top"]]*2,device="cuda")
        batch=dict(batch_idx=torch.zeros(len(anns),device="cuda"),
                   cls=torch.tensor([cat_to_cls[r["category_id"]] for r in anns],device="cuda"),
                   bboxes=ops.xyxy2xywh(boxes)/640)
        annotation_index={r["id"]:i for i,r in enumerate(anns)}
        for name,criterion in branch_criteria.items():
            with torch.no_grad():
                (fg,owner,_,_,_),_,_=criterion.get_assigned_targets_and_loss(raw[name],batch)
            scores=captured[name][2][0].sum(1)
            fg=fg[0].cpu()
            owner=owner[0].cpu()
            foreground=torch.where(fg)[0]
            counts[name+"_positive_candidates"]+=len(foreground)
            owners=set(owner[foreground].tolist())
            counts[name+"_gt_with_positive"]+=len(owners)
            for raw_id in foreground.tolist():
                assignment_rows.append(dict(image_id=iid,branch=name,raw_id=raw_id,
                    annotation_id=anns[int(owner[raw_id])]["id"],target_score=float(scores[raw_id])))
            if name!="one2one":
                continue
            for row in image["rows"]:
                rid=row["raw_id"]
                gt_index=annotation_index[row["annotation_id"]]
                same=bool(fg[rid]) and int(owner[rid])==gt_index
                state="same_gt_positive" if same else ("different_gt_positive" if bool(fg[rid]) else "not_positive")
                counts["geometric_"+state]+=1
                alternatives=foreground[owner[foreground]==gt_index].tolist()
                comparisons.append({**row,"assignment_state":state,"official_same_gt_raw_ids":alternatives})
        if number%25==0:
            print(json.dumps(dict(images=number,total=len(split["val"]))),flush=True)
    for handle in handles:
        handle.remove()
    summary=dict(counts=counts,assigners={name:{k:getattr(c.assigner,k,None) for k in ("topk","topk2","alpha","beta")}
                                        for name,c in branch_criteria.items()},
        replay="Direct official get_assigned_targets_and_loss, both branches, same frozen square input and original COCO boxes/classes; no private reimplementation",
        limits="Frozen unaugmented checkpoint replay, not pretrained training history or polygon-converted/overlap-rasterized full loader reproduction.",
        loss_correction="At checkpoint mask_ratio=1, official segmentation loss upsamples prototype to mask size. Prior native160 objective is a separate diagnostic.")
    (a.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2))
    (a.out/"COMPARISONS.json").write_text(json.dumps(comparisons,indent=2))
    (a.out/"OFFICIAL_ASSIGNMENTS.json").write_text(json.dumps(assignment_rows,indent=2))
    (a.out/"COMPLETE.json").write_text(json.dumps(dict(images=len(split["val"]),comparisons=len(comparisons))))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("data","weights","bank","split","out"):
        p.add_argument("--"+key,type=Path,required=True)
    main(p.parse_args())
