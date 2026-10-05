"""7R: fixed-prediction spatial decomposition, no training or target refit."""
import argparse,json,time,uuid,traceback,math
from datetime import datetime,timezone
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops
from local_replay import Q,ROOT,DATA,MODEL,load,preprocess,bounds,roi

RADII=(1,3,5)
REGIONS=("boundary","interior","neighbor","background")
MODES=("h_coefficient","p_coefficient","p_spatial","wrong_instance_spatial","wrong_image_spatial")
NAMES=tuple(f"{m}_s{s}" for m in MODES for s in (0,1))
RMET=("cos","nmse","amplitude","sign_agree_energy","mean_delta_positive","mean_delta_negative",
      "repair_fn","repair_fp","damage_tp","damage_tn","pred_energy","target_energy","pixels")
GMET=("pixels","positive","orig_fn","orig_fp","target_energy","oracle_repair_fn","oracle_repair_fp",
      "oracle_damage_tp","oracle_damage_tn","target_mean","target_rms")
SMET=("pixels","target_energy","oracle_repair","oracle_damage")
SUBREGIONS=("inner_boundary","outer_boundary_neighbor","outer_boundary_background","crowd_external")
def write(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
def now():return datetime.now(timezone.utc).isoformat()
def decode640(logits,box,shape,rp):
    m=ops.crop_mask(logits.clone(),box[None].expand(len(logits),-1).clone())>0
    return ops.scale_masks(m.float()[None],shape,ratio_pad=rp)[0]>.5
def morph(g,r):
    kernel=np.ones((2*r+1,2*r+1),np.uint8)
    return (cv2.erode(g.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool),
            cv2.dilate(g.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool))
def measure(t,p,y,b,om):
    """All arguments restricted to one region; p has arms x pixels."""
    n=len(t);a=len(p)
    if n==0:return np.full((a,len(RMET)),np.nan),np.full(len(GMET),np.nan)
    pos=y;neg=~y;pm=b[None]+p>0;bm=b>0
    te=float(np.sum(t*t));pe=np.sum(p*p,axis=1);dot=np.sum(p*t,axis=1)
    if te>1e-10:
        cos=dot/np.sqrt(pe*te).clip(1e-12)
        cos[pe<1e-10]=np.nan
        nmse=np.sum((p-t)**2,axis=1)/te
        amp=dot/te
        agree=np.sum((p*t>0)*t**2,axis=1)/te
    else:cos=nmse=amp=agree=np.full(a,np.nan)
    def avg(q):return p[:,q].mean(1) if q.any() else np.full(a,np.nan)
    fn=pos&~bm;fp=neg&bm;tp=pos&bm;tn=neg&~bm
    ps=np.stack((cos,nmse,amp,agree,avg(pos),avg(neg),
        (pm&fn).sum(1),(~pm&fp).sum(1),(~pm&tp).sum(1),(pm&tn).sum(1),
        pe,np.repeat(te,a),np.repeat(n,a)),1)
    gs=np.array((n,pos.sum(),fn.sum(),fp.sum(),te,(om&fn).sum(),(~om&fp).sum(),
        (~om&tp).sum(),(om&tn).sum(),t.mean(),np.sqrt(te/n)))
    return ps,gs

@torch.no_grad()
def main(a):
    import ultralytics
    assert ultralytics.__version__=="8.4.100"
    torch.set_num_threads(6);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    write(out/"run.json",dict(run_id=out.name,started_at=now(),status="running",stage="7R_decomposition",
        scope="frozen 7Q predictions, original COCO mask regions, local native-P replay"))
    t=load(Q/"runs/RUN_011491022c594a86aabaebd9d9d3a2a1/TEST.pt")
    pred=load(Q/"runs/RUN_e37f1bf4d7e6432187a451f23933c1e6/PREDICTIONS.pt")
    old=np.load(Q/"runs/RUN_e37f1bf4d7e6432187a451f23933c1e6/METRICS.npz")
    assert list(map(tuple,t["keys"]))==list(map(tuple,pred["keys"]))==list(map(tuple,old["keys"]))
    group=defaultdict(list)
    for i,k in enumerate(t["keys"]):group[int(k[0])].append(i)
    if a.limit:group=dict(list(group.items())[:a.limit])
    indices=[i for ix in group.values() for i in ix];n=len(indices);mapping={i:j for j,i in enumerate(indices)}
    rm=np.full((n,3,4,10,len(RMET)),np.nan,np.float32)
    gm=np.full((n,3,4,len(GMET)),np.nan,np.float32)
    hy=np.full((n,3,4,10),np.nan,np.float32)
    only=np.full((n,3,4),np.nan,np.float32);leave=np.full_like(only,np.nan)
    subs=np.full((n,4,10,len(RMET)),np.nan,np.float32);subg=np.full((n,4,len(GMET)),np.nan,np.float32)
    ious640=np.full((n,12),np.nan,np.float32);coverage=np.full_like(ious640,np.nan)
    # original,oracle,p_coefficient seeds,p_spatial seeds
    exact_names=("original","oracle","p_coefficient_s0","p_coefficient_s1","p_spatial_s0","p_spatial_s1")
    exact=np.full((n,6),np.nan,np.float32)
    errors=np.zeros((n,3),np.float32);gtarea=np.zeros(n);fullarea=np.zeros(n)
    source=COCO(str(DATA/"annotations/instances_val2017.json"))
    model=YOLO(str(MODEL)).model.cuda().float().eval()
    pp={name:pred["predictions"][name].cuda() for name in NAMES}
    (out/"native_proto").mkdir(exist_ok=True)
    begin=time.time();maxp=maxc=maxb=0.
    for step,(iid,ix) in enumerate(group.items(),1):
        x,shape,rp,hw=preprocess(iid);_,raw=model(x);r=raw["one2one"]
        proto=r["proto"][0];cc=r["mask_coefficient"][0].T
        bb=model.model[-1]._get_decode_boxes(r)[0].T
        rids=[t["keys"][i][2] for i in ix]
        ec=float((cc[rids].cpu()-t["c0"][ix]).abs().max())
        eb=float((bb[rids].cpu()-t["box"][ix]).abs().max())
        ep=float((torch.stack([roi(proto,t["box"][i]) for i in ix]).cpu()-t["p"][ix]).abs().max())
        assert ec<.001 and eb<.01 and ep<.001,(iid,ec,eb,ep)
        maxc=max(maxc,ec);maxb=max(maxb,eb);maxp=max(maxp,ep)
        torch.save(dict(proto=proto.cpu(),original_shape=shape,ratio_pad=rp,hw_resized=hw),
                   out/"native_proto"/f"{iid:012d}.pt")
        anns=source.imgToAnns[iid];masks={};origmasks={};union=np.zeros((640,640),bool);crowd=union.copy()
        left,top=rp[1];hh,ww=hw
        for ann in anns:
            rawmask=source.annToMask(ann).astype(np.uint8);origmasks[ann["id"]]=rawmask
            m=np.zeros((640,640),bool)
            m[top:top+hh,left:left+ww]=cv2.resize(rawmask,(ww,hh),interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[ann["id"]]=m;union|=m
            if ann.get("iscrowd"):crowd|=m
        for i in ix:
            k=mapping[i];aid=t["keys"][i][1]
            box=t["box"][i].cuda();c=t["c0"][i].cuda();d=t["delta"][i].cuda()
            z=(c@proto.flatten(1)).reshape(160,160);target=(d@proto.flatten(1)).reshape(160,160)
            bx1,by1,bx2,by2=bounds(box)
            effects=[torch.zeros_like(z),target]
            for name in NAMES:
                v=pp[name][i]
                if "coefficient" in name:e=(v@proto.flatten(1)).reshape(160,160)
                else:
                    e=torch.zeros_like(z)
                    e[by1:by2,bx1:bx2]=F.interpolate(v.reshape(1,1,16,16),(by2-by1,bx2-bx1),
                        mode="bilinear",align_corners=False)[0,0]
                effects.append(e)
            eff=F.interpolate(torch.stack(effects)[None],(640,640),mode="bilinear",align_corners=False)[0]
            base=F.interpolate(z[None,None],(640,640),mode="bilinear",align_corners=False)[0,0]
            exact_ix=[0,1,4,5,6,7]
            binary=decode640(base[None]+eff[exact_ix],box,shape,rp)
            original_gt=torch.from_numpy(origmasks[aid].astype(bool)).cuda()
            inter=(binary&original_gt).sum((1,2));un=(binary|original_gt).sum((1,2))
            exact[k]=(inter/un.clamp_min(1)).cpu().numpy()
            oldix=[list(old["names"]).index(name) for name in exact_names]
            errors[k,0]=ec;errors[k,1]=ep;errors[k,2]=np.max(np.abs(exact[k]-old["values"][i,oldix,0]))
            x1=max(left,0,math.ceil(float(box[0])));y1=max(top,0,math.ceil(float(box[1])))
            x2=min(left+ww,640,math.ceil(float(box[2])));y2=min(top+hh,640,math.ceil(float(box[3])))
            gt=masks[aid];gtarea[k]=gt.sum()
            if x2<=x1 or y2<=y1:ious640[k]=0;coverage[k]=0;continue
            sl=np.s_[y1:y2,x1:x2]
            y=gt[sl].flatten();other=(union&~gt)[sl];bgcrowd=(crowd&~gt)[sl]
            b=base[sl].cpu().numpy().flatten();e=eff[:,y1:y2,x1:x2].cpu().numpy().reshape(12,-1)
            truth=e[1];predictions=e[2:];fullarea[k]=len(y)
            bm=b>0;allm=b[None]+e>0
            tp=(allm&y).sum(1);fp=(allm&~y).sum(1);total=gtarea[k]
            ious640[k]=tp/np.maximum(total+fp,1);coverage[k]=tp/max(total,1)
            for ir,radius in enumerate(RADII):
                er,di=morph(gt,radius)
                br=(di&~er)[sl];ins=er[sl]
                ne=(other&~di[sl]);ba=~(br|ins|ne)
                rs=[br,ins,ne,ba]
                assert np.all(sum(rr.astype(np.uint8) for rr in rs)==1)
                for jr,reg in enumerate(rs):
                    q=reg.flatten()
                    vals,geom=measure(truth[q],predictions[:,q],y[q],b[q],allm[1,q])
                    rm[k,ir,jr]=vals;gm[k,ir,jr]=geom
                    tpr=(allm[:,q]&y[q]).sum(1);fpr=(allm[:,q]&~y[q]).sum(1)
                    hy[k,ir,jr]=(tp[2:]+tpr[1]-tpr[2:])/np.maximum(total+fp[2:]+fpr[1]-fpr[2:],1)
                    only[k,ir,jr]=(tp[0]+tpr[1]-tpr[0])/max(total+fp[0]+fpr[1]-fpr[0],1)
                    leave[k,ir,jr]=(tp[1]-tpr[1]+tpr[0])/max(total+fp[1]-fpr[1]+fpr[0],1)
                if radius==3:
                    sr=[br&gt[sl],br&~gt[sl]&other,br&~gt[sl]&~other,bgcrowd]
                    for si,reg in enumerate(sr):
                        q=reg.flatten()
                        subs[k,si],subg[k,si]=measure(truth[q],predictions[:,q],y[q],b[q],allm[1,q])
            if a.limit and k<3:
                # Exact count formula agrees with materialized GT-assisted region replacement.
                q=rs[0].flatten();v=allm[6].copy();v[q]=allm[1,q]
                direct=(v&y).sum()/max(total+(v&~y).sum(),1)
                assert abs(direct-float(hy[k,2,0,4]))<1e-6
        if step%10==0 or step==len(group):
            status=dict(images=step,total_images=len(group),candidates=sum(len(v) for v in list(group.values())[:step]),
                elapsed_seconds=time.time()-begin,max_c_error=maxc,max_box_error=maxb,max_p_error=maxp)
            write(out/"PROGRESS.json",status);print(json.dumps(status),flush=True)
    np.savez_compressed(out/"DECOMPOSITION.npz",keys=np.asarray(t["keys"])[indices],names=NAMES,
        radii=RADII,regions=REGIONS,regional=rm,geometry=gm,regional_metrics=RMET,geometry_metrics=GMET,
        hybrid_iou=hy,oracle_only_iou=only,oracle_leaveout_iou=leave,
        subregions=SUBREGIONS,subregional=subs,subgeometry=subg,
        iou640=ious640,coverage640=coverage,original_iou=old["values"][indices,0,0],
        box_iou=old["box_iou"][indices],exact_iou=exact,exact_names=exact_names,
        replay_errors=errors,gtarea=gtarea,supportarea=fullarea)
    summary=dict(images=len(group),instances=n,status="completed",elapsed_seconds=time.time()-begin,
        max_c_error=maxc,max_box_error=maxb,max_p_error=maxp,
        original_resolution_iou_max_error=float(errors[:,2].max()),
        original_resolution_iou_error_above_1e_3=int((errors[:,2]>.001).sum()),
        no_support=int((fullarea==0).sum()),no_resized_gt=int((gtarea==0).sum()),
        notes="Reused 7Q targets and predictions. Original-mask regions at input640; not COCO AP.")
    write(out/"COMPLETE.json",summary)
    record=json.loads((out/"run.json").read_text(encoding="utf-8-sig"))
    write(out/"run.json",dict(record,status="completed",finished_at=now()))
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--limit",type=int,default=0)
    a=p.parse_args()
    try:main(a)
    except Exception:
        if a.out.exists():
            write(a.out/"FAILED.json",dict(time=now(),traceback=traceback.format_exc()))
        raise


