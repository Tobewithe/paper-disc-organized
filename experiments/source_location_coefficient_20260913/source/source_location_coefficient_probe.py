"""S071: spatial ownership-controlled coefficient query, no optimization.

Nearest own-GT interior query and its reflected non-own query are equally far
from the original source, same FPN scale, both inside the GT bounding rectangle.
Their selection never reads candidate mask IoU. Prediction-interior and local
mean controls use no GT to select a coefficient, but this is a GT-selected
diagnostic population, not an unconditional method/AP evaluation.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import shutil
import time
from collections import Counter
from joint_failure_decoder_probe import (ROOT,RUNTIME,GOLD,ANNOTATION,COCO,YOLO,TraceCapture,
    ops,mu,torch,F,np,pd,cv2,dump,save_csv,sha,restore,tight_input_box)

PRIOR=ROOT/"diagnostics/joint_failure_decoder_20260913_v2"
STATES=["requires_target_pixel_recovery","sufficient_true_pixels_but_residual_false_pixels"]
ARMS=["original","own_query","reflected_nonown","local_mean","predicted_interior"]


def rank(*values):return hashlib.sha256(("S071:20260913:"+":".join(map(str,values))).encode()).hexdigest()


def gt_input(gt,iid,aid):
    own=gt.annToMask(gt.anns[aid]).astype(bool)
    sh=own.shape;gain=min(640/sh[0],640/sh[1]);nh,nw=round(sh[0]*gain),round(sh[1]*gain)
    top,left=round((640-nh)/2-.1),round((640-nw)/2-.1)
    inp=np.zeros((640,640),bool)
    inp[top:top+nh,left:left+nw]=cv2.resize(own.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST_EXACT)>0
    return own,inp,gain,left,top


def offsets(radius=2):
    return [(dx,dy) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1) if dx*dx+dy*dy<=radius*radius]


def choose(gt,df,out):
    eligible=[];rejected=Counter()
    cand=df[df.density.isin(["low","high"]) & df.prior_residual.isin(STATES)
            & ~df.official_mask75 & ~df.any_retained_mask75 & df.source_center_location.ne("own")].copy()
    for row in cand.to_dict("records"):
        aid,iid=int(row["annotation_id"]),int(row["image_id"])
        own,inp,gain,left,top=gt_input(gt,iid,aid)
        px,py=int(row["source_x"]),int(row["source_y"]);stride=int(row["source_stride"])
        assert not inp[py,px]
        yy,xx=np.nonzero(own);box=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])*gain+np.array([left,top,left,top])
        crowd=np.zeros_like(inp)
        for ann in gt.imgToAnns[iid]:
            if ann.get("iscrowd",0):crowd|=gt_input(gt,iid,ann["id"])[1]
        choices=[]
        for dx,dy in offsets():
            if dx==dy==0:continue
            ox,oy=px+dx*stride,py+dy*stride
            mx,my=px-dx*stride,py-dy*stride
            if not all([0<=ox<640,0<=oy<640,0<=mx<640,0<=my<640]):continue
            if not (box[0]<=ox<box[2] and box[1]<=oy<box[3] and box[0]<=mx<box[2] and box[1]<=my<box[3]):continue
            if inp[oy,ox] and not inp[my,mx] and not crowd[my,mx] and not crowd[oy,ox]:
                choices.append((dx*dx+dy*dy,rank(aid,dx,dy),dx,dy))
        if not choices:
            rejected["no_own_and_reflected_nonown_inside_same_gt_box_radius2"]+=1;continue
        _,_,dx,dy=min(choices)
        width=640//stride;offset={8:0,16:6400,32:8000}[stride]
        idx=int(row["source_index"])
        assert offset+(py//stride)*width+(px//stride)==idx
        own_idx=idx+dy*width+dx;mirror_idx=idx-dy*width-dx
        row.update(own_source=own_idx,mirror_source=mirror_idx,dx=dx,dy=dy,source_distance_cells=float(np.hypot(dx,dy)))
        eligible.append(row)
    ed=pd.DataFrame(eligible)
    save_csv(out/"eligible.csv",eligible)
    pairs=[];used_images=set()
    # Match density groups on residual type, category, area, FPN stride and
    # within .05 Box IoU. No image is used twice across all pairs.
    for state in STATES:
        highs=ed[(ed.density=="high") & (ed.prior_residual==state)].to_dict("records")
        lows=ed[(ed.density=="low") & (ed.prior_residual==state)].to_dict("records")
        count=0
        for h in sorted(highs,key=lambda r:rank(r["annotation_id"])):
            if int(h["image_id"]) in used_images:continue
            options=[l for l in lows if int(l["image_id"]) not in used_images and int(l["image_id"])!=int(h["image_id"])
                and l["category_id"]==h["category_id"] and l["area_bin"]==h["area_bin"]
                and l["source_stride"]==h["source_stride"] and abs(l["box_iou"]-h["box_iou"])<=.05]
            if not options:continue
            l=min(options,key=lambda r:(abs(r["box_iou"]-h["box_iou"]),rank(r["annotation_id"])))
            pair_id=len(pairs)//2
            for row in [h,l]:
                row["pair_id"]=pair_id;pairs.append(row);used_images.add(int(row["image_id"]))
            count+=1
            if count==64:break
    assert len(used_images)==len(pairs)
    assert len(pairs)>0
    save_csv(out/"selection.csv",pairs)
    dump(out/"selection_summary.json",dict(initial_candidates=len(cand),eligible=len(eligible),rejected=dict(rejected),
        selected_targets=len(pairs),pairs=len(pairs)//2,per_state=dict(Counter(r["prior_residual"] for r in pairs)),
        quota_per_state=64,scope="Up to64 matched high/low pairs per residual state; common support may yield fewer. Each image appears once."))
    return pairs


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/"source").mkdir();(out/"tensors").mkdir()
    start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    df=pd.read_csv(PRIOR/"instances_analyzed.csv")
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ANNOTATION))
    pairs=choose(gt,df,out)
    dump(out/"protocol.json",dict(experiment="S071_SOURCE_OWNERSHIP_COEFFICIENT_QUERY",training=False,
        question="Do same-scale nearby own-interior features provide a better coefficient than equally displaced non-own features, at fixed P and crop?",
        incremental="S014 selected best coefficients by candidate mask IoU. This selection uses GT spatial ownership only, with mirrored-distance control, and matches high/low category/area/stride/boxIoU.",
        decision_positive="If own beats both original and mirror and simple mean, investigate spatial query/readout alignment, then test an automatic or trained selector on new images.",
        decision_negative="If own fails to beat controls, stop interpreting outside-own source position as a sufficient explanation and do not add mask-aware query modules based on association alone.",
        arms=ARMS,crops=["original_predicted_box","same_gt_box"],radius_cells=2,
        selection="Fixed selection.csv before intervention; no candidate mask IoU used. Prior GT residual selection and exact matched common-support sample are diagnostic conditions, not population AP.",
        main="own_query minus original/reflected_nonown; all targets incl harm kept. GT crops are shared among arms, not optimized separately.",
        controls="local_mean averages 3x3 same-level cells. predicted_interior selects max distance-to-current-predicted-mask-boundary among radius2 lattice centers, ties nearest original then source ID; retains original if no positive candidate.",
        limitations="Own and mirror use GT. Prediction-only controls are still measured on a GT-selected cohort. No task AP or method claim. One frozen pretrained model; no training seeds.",
        protocol_selection_sha256=sha(out/"selection.csv"),source_sha256=sha(__file__)))
    for path in [Path(__file__),Path(ops.__file__),ROOT/"joint_failure_decoder_probe.py",ROOT/"structure_candidate_trace.py",ROOT/"frozen_mechanism_probe.py"]:
        shutil.copy2(path,out/"source"/path.name)
    model=YOLO(str(ROOT/"weights/yolo26m-seg.pt"));model.model.eval().requires_grad_(False)
    results=[]
    for number,row in enumerate(pairs,1):
        iid,aid,slot,source=[int(row[k]) for k in ["image_id","annotation_id","slot","source_index"]]
        with torch.inference_mode():
            model.predict(str(RUNTIME/"data/images/val2017"/gt.imgs[iid]["file_name"]),predictor=TraceCapture,
                imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,quantize=32,retina_masks=False,device=0,verbose=False)
            raw,cap=model.predictor.dense,model.predictor.capture
            c,proto,boxes=cap["coeff"],cap["proto"].float(),cap["boxes"]
            assert torch.equal(raw[0,84:,source],c[slot])
            old=read_golden=read_gz(GOLD/"images"/f"{iid}.json.gz")
            assert len(c)==len(old["original"]) and old["witness"]["source_indices"][slot]==source
            own,inp,gain,left,top=gt_input(gt,iid,aid)
            shape=own.shape;gtbox=tight_input_box(own,gain,left,top,c.device)
            z=F.interpolate((c@proto.flatten(1)).reshape(1,len(c),*proto.shape[-2:]),(640,640),mode="bilinear",align_corners=False)[0]
            original_binary=ops.crop_mask((z>0).byte(),boxes)
            baseline=restore(original_binary[slot:slot+1],shape)[0]
            assert np.array_equal(baseline,mu.decode(old["original"][slot]["segmentation"]).astype(bool))
            stride=int(row["source_stride"]);width=640//stride;offset={8:0,16:6400,32:8000}[stride]
            py,px=int(row["source_y"]),int(row["source_x"])
            neighbours=[];local=[]
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    cx,cy=px+dx*stride,py+dy*stride
                    if 0<=cx<640 and 0<=cy<640:local.append(offset+(cy//stride)*width+(cx//stride))
            binary=original_binary[slot].cpu().numpy()
            distance=cv2.distanceTransform(binary,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            for dx,dy in offsets():
                cx,cy=px+dx*stride,py+dy*stride
                if 0<=cx<640 and 0<=cy<640 and binary[cy,cx]:
                    idx=offset+(cy//stride)*width+(cx//stride)
                    neighbours.append((-float(distance[cy,cx]),dx*dx+dy*dy,idx))
            inferred=min(neighbours)[2] if neighbours else source
            sources=dict(original=source,own_query=int(row["own_source"]),reflected_nonown=int(row["mirror_source"]),predicted_interior=inferred)
            coefficients={k:raw[0,84:,idx] for k,idx in sources.items()}
            coefficients["local_mean"]=raw[0,84:,local].mean(1)
            assert coefficients["local_mean"].shape==(32,)
            arr=[]
            for arm in ARMS:
                cc=c.clone();cc[slot]=coefficients[arm]
                # Full candidate GEMM matches the reference numerical layout.
                logits=(cc@proto.flatten(1)).reshape(len(c),*proto.shape[-2:])
                zz=F.interpolate(logits[None],(640,640),mode="bilinear",align_corners=False)[0,slot:slot+1]
                masks=[]
                for crop,box in [("original_predicted_box",boxes[slot:slot+1]),("same_gt_box",gtbox[None])]:
                    mask=restore(ops.crop_mask((zz>0).byte(),box),shape)[0]
                    if arm=="original" and crop=="original_predicted_box":assert np.array_equal(mask,baseline)
                    tp=int((mask&own).sum());fp=int((mask&~own).sum());area=int(own.sum())
                    results.append(dict(pair_id=int(row["pair_id"]),annotation_id=aid,image_id=iid,density=row["density"],
                        prior_residual=row["prior_residual"],category_id=int(row["category_id"]),area_bin=row["area_bin"],stride=stride,
                        arm=arm,crop=crop,iou=tp/(area+fp),coverage=tp/area,fp_per_gt=fp/area,mask75=tp/(area+fp)>=.75,
                        selected_source=sources.get(arm,-1),source_distance_cells=row["source_distance_cells"]))
                    masks.append(encode(mask))
                arr.append(coefficients[arm].cpu().numpy())
            cls_idx=sorted(gt.cats).index(int(row["category_id"]))
            np.savez_compressed(out/"tensors"/f"{aid}.npz",arms=np.array(ARMS),coeff=np.stack(arr),
                sources=np.array([sources.get(k,-1) for k in ARMS]),source_class_scores=np.array([float(raw[0,4+cls_idx,sources[k]]) if k in sources else np.nan for k in ARMS]),
                proto=proto.cpu().numpy(),pred_box=boxes[slot].cpu().numpy(),gt_box=gtbox.cpu().numpy(),shape=np.array(shape))
        if number%20==0 or number==len(pairs):
            status=dict(stage="QUERY_PROBE",completed=number,total=len(pairs),seconds=time.monotonic()-start)
            dump(out/"progress.json",status);save_csv(out/"effects.csv",results);print(json.dumps(status),flush=True)
    dump(out/"COMPLETE.json",dict(status="COMPLETE",targets=len(pairs),pairs=len(pairs)//2,rows=len(results),
        seconds=time.monotonic()-start,effects_sha256=sha(out/"effects.csv")))
    dump(out/"progress.json",dict(stage="COMPLETE"))


if __name__=="__main__":
    from mask_error_ap_probe import read_gz,encode
    main()
