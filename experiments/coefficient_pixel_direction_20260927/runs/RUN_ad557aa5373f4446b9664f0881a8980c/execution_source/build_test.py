"""Rebuild the fixed 2000-image test bank and finite targets; training locked."""
import argparse,json,sys
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.augment as augment_module
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
import ultralytics
from common import *

def main(a):
    setup()
    assert ultralytics.__version__=="8.4.100"
    assert (ROOT/"TRAINING_COMPLETE.json").exists()
    sys.path.insert(0,str(OFFICIAL/"scripts"));sys.path.insert(0,str(ORACLE/"scripts"))
    from official_pipeline import target_rois
    from finite_oracle import solve
    a.out.mkdir(parents=True,exist_ok=True);(a.out/"images").mkdir(exist_ok=True)
    manifest=json.loads((PRIOR/"MANIFEST.json").read_text())
    selected=manifest["independent_test_images"]
    assert len(selected)==2000
    assert set(selected).isdisjoint(manifest["train_images"])
    assert set(selected).isdisjoint(manifest["old_development_images"])
    identities=json.loads((TEST_LABELS/"IDENTITIES.json").read_text())
    wrapper=YOLO(str(WEIGHTS));model=wrapper.model.cuda().float().eval()
    model.args=SimpleNamespace(**wrapper.ckpt["train_args"])
    for parameter in model.parameters():parameter.requires_grad_(False)
    criterion=model.init_criterion().one2one;head=model.model[-1]
    maps={}
    handles=[head.one2one_cv4[l][-1].register_forward_pre_hook(
        lambda _,x,l=l:maps.__setitem__(l,x[0].detach())) for l in range(3)]
    order={};original=augment_module.polygons2masks_overlap
    def raster(*args,**kwargs):
        mask,indices=original(*args,**kwargs);order["indices"]=indices.copy()
        return mask,indices
    augment_module.polygons2masks_overlap=raster
    data=TEST_LABELS/"official_data"
    ds=YOLODataset(img_path=str(data/"test.txt"),imgsz=640,batch_size=1,
        augment=False,hyp=deepcopy(model.args),rect=False,cache=False,stride=32,
        data={"names":wrapper.names,"nc":80,"channels":3},task="segment")
    by_id={int(Path(p).stem):j for j,p in enumerate(ds.im_files)}
    assert set(by_id)==set(selected)
    coco=COCO(str(TEST_LABELS/"conversion_input/instances_val2017.json"))
    rows=[];max_stationary=0.;index=[]
    try:
        for step,iid in enumerate(selected,1):
            iid=int(iid);j=by_id[iid];label=ds.labels[j]
            path=data/"labels/val2017"/f"{iid:012d}.txt"
            lines=path.read_text().strip().splitlines() if path.exists() else []
            source=[(np.asarray(line.split(),dtype=np.float32),aid)
                for line,aid in zip(lines,identities[str(iid)])]
            pre=[]
            for cls,polygon in zip(label["cls"].flatten(),label["segments"]):
                key=np.r_[cls,polygon.flatten()].astype(np.float32)
                matched=[aid for value,aid in source if np.array_equal(key,value)]
                assert len(matched)==1,(iid,matched);pre.append(matched[0])
            order.clear();sample=ds[j]
            annids=np.asarray(pre)[order["indices"]].tolist() if pre else []
            assert len(annids)==len(sample["cls"])
            batch=YOLODataset.collate_fn([sample])
            batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            with torch.no_grad():
                _,raw=model(batch["img"].float()/255)
                assignment,_,_=criterion.get_assigned_targets_and_loss(raw["one2one"],batch)
                fg,owners,target_boxes=(t[0] for t in assignment[:3])
                ids=torch.where(fg)[0];ownerids=owners[ids]
                assert len(set(ownerids.tolist()))==len(ownerids)
                pred=raw["one2one"];proto=pred["proto"][0]
                coeff=pred["mask_coefficient"][0].T
                boxes=head._get_decode_boxes(pred)[0].T
                h=torch.cat([maps[l][0].flatten(1).T for l in range(3)])
                levels=torch.cat([torch.full((maps[l].shape[-2]*maps[l].shape[-1],),l) for l in range(3)])
            selected_rows=[]
            scaled=ops.scale_boxes((640,640),boxes[ids].clone(),sample["ori_shape"],ratio_pad=sample["ratio_pad"])
            for k,(rid,owner) in enumerate(zip(ids.tolist(),ownerids.tolist())):
                ann=coco.anns[int(annids[owner])]
                gb=torch.tensor(ann["bbox"],device="cuda").float();gb[2:]+=gb[:2]
                selected_rows.append(dict(image_id=iid,annotation_id=int(ann["id"]),raw_id=rid,
                    level=int(levels[rid]),box_iou=float(box_iou(gb[None],scaled[k:k+1])[0,0])))
            image=dict(proto=proto.cpu(),coeff=coeff[ids].cpu(),boxes=boxes[ids].cpu(),
                masks=sample["masks"][0].cpu(),owners=ownerids.cpu(),target_boxes=target_boxes[ids].cpu(),
                rows=selected_rows,original_shape=sample["ori_shape"],ratio_pad=sample["ratio_pad"])
            if len(ids):
                targets=target_rois(image)
                delta=[]
                for k,(r,target) in enumerate(zip(selected_rows,targets)):
                    c0=coeff[r["raw_id"]]
                    d,state=solve(c0,target,.003,120)
                    if not torch.isfinite(d).all() or state["stationary_norm"]>1e-3:
                        d,state=solve(c0,target,.003,240)
                    assert torch.isfinite(d).all() and state["stationary_norm"]<=1e-3,(r,state)
                    max_stationary=max(max_stationary,state["stationary_norm"])
                    delta.append(d.cpu())
                    rows.append(dict(r,h=h[r["raw_id"]].cpu(),c0=c0.cpu(),delta=d.cpu(),
                        p=roi(proto,boxes[r["raw_id"]]).cpu(),box=boxes[r["raw_id"]].cpu()))
                image["delta"]=torch.stack(delta)
                # Evaluation AUC is explicitly on official ownership labels at native resolution.
                image["owner160"]=F.interpolate(image.pop("masks")[None,None].float(),
                    (160,160),mode="nearest")[0,0].short()
                torch.save(image,a.out/"images"/f"{iid:012d}.pt")
            index.append(dict(image_id=iid,instances=len(ids)))
            if step%25==0 or step==len(selected):
                status=dict(images=step,total=len(selected),instances=len(rows),max_stationary=max_stationary)
                write(a.out/"PROGRESS.json",status);print(json.dumps(status),flush=True)
        torch.save(assemble(rows),a.out/"TEST.pt")
        write(a.out/"INDEX.json",index)
        write(a.out/"COMPLETE.json",dict(images=len(selected),instances=len(rows),max_stationary=max_stationary,
            reused_diagnostic_test=True,auc_gt="official owner label, nearest resize to native160",
            final_iou_gt="original COCO annToMask"))
    finally:
        augment_module.polygons2masks_overlap=original
        for handle in handles:handle.remove()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args())

