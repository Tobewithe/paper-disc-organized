"""S073: current official TAL assignment of S071 original and donor queries.

No optimizer, backward pass, BN updates or reconstruction of historical
training. All ordinary GT compete simultaneously, using unchanged raw scores
and boxes. Official one-to-many assigner defaults are used verbatim.
"""
from pathlib import Path
import argparse
import contextlib
import io
import json
import shutil
import time
from joint_failure_decoder_probe import (ROOT,RUNTIME,GOLD,ANNOTATION,COCO,YOLO,TraceCapture,
    ops,mu,torch,np,pd,dump,save_csv,sha)
from source_location_coefficient_probe import gt_input
from mask_error_ap_probe import read_gz
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils import tal,loss


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/"source").mkdir();(a.out/"assignments").mkdir()
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    selection=pd.read_csv(a.source/"selection.csv")
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ANNOTATION))
    dump(a.out/"protocol.json",dict(experiment="S073_SOURCE_QUERY_CURRENT_ASSIGNMENT",training=False,optimizer_steps=0,
        question="Were S071 own-interior donors assigned to the same GT by the current official one-to-many assigner?",
        decision="Many unassigned/wrong-GT donors mean S071 rejects naive transplantation only; an own-positive subset can test whether effect survives this confound, without pretending to prove historical supervision.",
        source_selection_sha256=sha(a.source/"selection.csv"),assigner=dict(topk=10,alpha=.5,beta=6,stride=[8,16,32],topk2=None),
        GT="All original ordinary COCO annotations, one annotation per instance, original xywh boxes transformed by the actual integer letterbox pad and gain; no polygons or segmentation labels needed by this box/class assigner.",
        predictions="Same frozen raw class probabilities, decoded boxes, actual executed head anchors; eval BN, no augmentation.",
        warning="Current frozen-checkpoint assignment is not historical positive assignment, training-gradient audit or proof that donor features lack information. Different candidate scores/box geometry mediate assignment.",
        code_sha256=sha(__file__),tal_sha256=sha(Path(tal.__file__)),loss_sha256=sha(Path(loss.__file__))))
    for path in [Path(__file__),Path(tal.__file__),Path(loss.__file__)]:shutil.copy2(path,a.out/"source"/path.name)
    model=YOLO(str(ROOT/"weights/yolo26m-seg.pt"));model.model.eval().requires_grad_(False)
    assigner=TaskAlignedAssigner(topk=10,num_classes=80,alpha=.5,beta=6.,stride=[8,16,32],topk2=None)
    categories=sorted(gt.cats);rows=[];start=time.monotonic()
    for number,row in enumerate(selection.to_dict("records"),1):
        iid,aid=int(row["image_id"]),int(row["annotation_id"])
        saved=np.load(a.source/"tensors"/f"{aid}.npz")
        with torch.inference_mode():
            model.predict(str(RUNTIME/"data/images/val2017"/gt.imgs[iid]["file_name"]),predictor=TraceCapture,
                imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,quantize=32,retina_masks=False,device=0,verbose=False)
            raw=model.predictor.dense
            head=model.predictor.model.backend.model.model[-1]
            anchors=(head.anchors*head.strides).T
            assert anchors.shape==(8400,2)
            _,_,gain,left,top=gt_input(gt,iid,aid)
            anns=[x for x in gt.imgToAnns[iid] if not x.get("iscrowd",0) and not x.get("ignore",0)]
            ann_ids=np.array([x["id"] for x in anns])
            box=torch.tensor([x["bbox"] for x in anns],device=raw.device,dtype=torch.float32)
            box[:,2:]+=box[:,:2]
            box=box*gain+torch.tensor([left,top,left,top],device=raw.device)
            label=torch.tensor([categories.index(x["category_id"]) for x in anns],device=raw.device,dtype=torch.float32)[None,:,None]
            prob=raw[:,4:84,:].permute(0,2,1)
            physical=ops.xywh2xyxy(raw[:,:4,:].permute(0,2,1))
            _,tbox,tscore,fg,tidx=assigner(prob,physical,anchors,label,box[None],torch.ones_like(label,dtype=torch.bool))
            target_index=int(np.flatnonzero(ann_ids==aid)[0])
            for arm,source,coef in zip(saved["arms"],saved["sources"],saved["coeff"]):
                if source<0:continue
                source=int(source)
                assert np.array_equal(raw[0,84:,source].cpu().numpy(),coef),(aid,str(arm),"Coefficient source changed")
                isfg=bool(fg[0,source]);assigned=int(ann_ids[int(tidx[0,source])]) if isfg else -1
                status="own_positive" if assigned==aid else "other_positive" if isfg else "not_positive"
                rows.append(dict(annotation_id=aid,image_id=iid,pair_id=int(row["pair_id"]),density=row["density"],
                    prior_residual=row["prior_residual"],arm=str(arm),source=source,assignment=status,assigned_annotation=assigned,
                    true_class_score=float(prob[0,source,categories.index(int(row["category_id"]))]),
                    assigned_target_weight=float(tscore[0,source].sum()),coeff_norm=float(raw[0,84:,source].norm()),
                    own_positive_count=int((fg[0]&(tidx[0]==target_index)).sum())))
            np.savez_compressed(a.out/"assignments"/f"{aid}.npz",annotation_ids=ann_ids,fg=fg[0].cpu().numpy(),
                target_gt_index=tidx[0].cpu().numpy(),anchors=anchors.cpu().numpy())
        if number%24==0 or number==len(selection):
            dump(a.out/"progress.json",dict(completed=number,total=len(selection),seconds=time.monotonic()-start))
    save_csv(a.out/"assignments.csv",rows)
    d=pd.DataFrame(rows)
    summary=d.groupby(["density","arm","assignment"]).size().reset_index(name="n")
    summary.to_csv(a.out/"summary.csv",index=False)
    effects=pd.read_csv(a.source/"effects.csv")
    statuses=d.pivot(index="annotation_id",columns="arm",values="assignment")
    both=statuses[(statuses.original=="own_positive")&(statuses.own_query=="own_positive")].index
    stats=[]
    for (crop,density),g in effects[effects.annotation_id.isin(both)].groupby(["crop","density"]):
        w=g.pivot(index="annotation_id",columns="arm",values="iou")
        delta=(w.own_query-w.original).to_numpy()
        rng=np.random.default_rng(20260913);boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)*100
        ci=np.quantile(boot,[.025,.975])
        stats.append(dict(crop=crop,density=density,n=len(delta),delta_iou_points=float(delta.mean()*100),ci_low=float(ci[0]),ci_high=float(ci[1])))
    save_csv(a.out/"both_own_positive_contrasts.csv",stats)
    dump(a.out/"COMPLETE.json",dict(status="COMPLETE",targets=len(selection),rows=len(rows),seconds=time.monotonic()-start,
        interpretation="Original and donor verified identical to S071; all statuses refer only to current checkpoint/unaugmented official assignment."))
    print(summary.to_string(index=False));print(pd.DataFrame(stats).round(4).to_string(index=False))


if __name__=="__main__":main()
