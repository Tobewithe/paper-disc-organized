"""Build 7Q fit/dev representations using frozen exact FP32 prototypes."""
import argparse,json,sys
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
import torch
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics
from common import *

def main(a):
    setup()
    assert ultralytics.__version__=="8.4.100"
    a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(PRIOR/"scripts"))
    from build_oracle_features import decompressed
    manifest=json.loads((PRIOR/"MANIFEST.json").read_text())
    target={}
    for path in sorted((TARGETS/"shards").glob("PART_*.pt")):
        for r in load(path):
            key=(int(r["image_id"]),int(r["annotation_id"]),int(r["original_raw_id"]))
            assert key not in target
            target[key]=r
    assert len(target)==71269
    wrapper=YOLO(str(WEIGHTS))
    model=wrapper.model.cuda().float().eval()
    model.args=SimpleNamespace(**wrapper.ckpt["train_args"])
    assert model.args.mask_ratio==1 and model.args.overlap_mask
    for parameter in model.parameters(): parameter.requires_grad_(False)
    head=model.model[-1]
    maps={}
    hooks=[head.one2one_cv4[l][-1].register_forward_pre_hook(
        lambda _,x,l=l:maps.__setitem__(l,x[0].detach())) for l in range(3)]
    ds=YOLODataset(img_path=str(BANK/"official_data/fit.txt"),imgsz=640,batch_size=1,
        augment=False,hyp=deepcopy(model.args),rect=False,cache=False,stride=32,
        data={"names":wrapper.names,"nc":80,"channels":3},task="segment")
    by_id={int(Path(p).stem):j for j,p in enumerate(ds.im_files)}
    selected=manifest["train_images"][:a.count] if a.count else manifest["train_images"]
    target_images={k[0] for k in target}
    n=0;block=[];max_h=max_c=max_box=0.
    shard=a.out/"fit";shard.mkdir(exist_ok=True)
    for step,iid in enumerate(selected,1):
        path=BANK/"images"/f"{iid:012d}.pt"
        if path.exists(): cache=decompressed(load(path))
        else:
            assert iid not in target_images, f"Missing positive cache: {iid}"
            cache={"rows":[]}
        if cache["rows"]:
            sample=ds[by_id[iid]]
            assert torch.equal(sample["masks"][0].long(),cache["masks"]),iid
            with torch.no_grad():
                _,raw=model(sample["img"].cuda().float()[None]/255)
                pred=raw["one2one"]
                proto=pred["proto"][0]
                coeff=pred["mask_coefficient"][0].T
                h=torch.cat([maps[l][0].flatten(1).T for l in range(3)])
                boxes=head._get_decode_boxes(pred)[0].T
            for r in cache["rows"]:
                rawid=int(r["original_raw_id"])
                key=(int(iid),int(r["annotation_id"]),rawid)
                t=target[key]
                max_h=max(max_h,float((h[rawid].cpu()-t["h"]).abs().max()))
                max_c=max(max_c,float((coeff[rawid].cpu()-t["c0"]).abs().max()))
                max_box=max(max_box,float((boxes[rawid].cpu()-t["predicted_box"]).abs().max()))
                block.append(dict(image_id=iid,annotation_id=key[1],raw_id=rawid,
                    level=int(t["level"]),h=t["h"],c0=t["c0"],delta=t["delta"],
                    p=roi(proto,boxes[rawid]).cpu(),box=boxes[rawid].cpu()))
                n+=1
        if step%250==0 or step==len(selected):
            if block: torch.save(assemble(block),shard/f"PART_{step:05d}.pt")
            block=[]
            status=dict(images=step,total=len(selected),instances=n,max_h_error=max_h,
                max_c_error=max_c,max_box_error=max_box)
            write(a.out/"PROGRESS.json",status);print(json.dumps(status),flush=True)
    assert max_h<1e-4 and max_c<1e-4 and max_box<1e-3
    for handle in hooks:handle.remove()
    del model,wrapper,ds,target
    dev_rows=load(DEV_TARGETS)
    grouped=defaultdict(list)
    for r in dev_rows:grouped[int(r["image_id"])].append(r)
    dev=[]
    for iid,rows in sorted(grouped.items()):
        cache=load(OLD_BANK/"images"/f"{iid:012d}.pt")
        assert cache["coeff"].shape==(8400,32)
        for r in rows:
            rawid=int(r["raw_id"])
            assert torch.allclose(cache["coeff"][rawid].float(),r["c0"],atol=1e-5,rtol=1e-5)
            dev.append(dict(image_id=iid,annotation_id=int(r["annotation_id"]),raw_id=rawid,
                level=int(cache["levels"][rawid]),h=r["h"],c0=r["c0"],delta=r["delta"],
                p=roi(cache["proto"].float(),cache["boxes"][rawid]),box=cache["boxes"][rawid]))
    assert set(selected).isdisjoint(grouped)
    torch.save(assemble(dev),a.out/"DEV.pt")
    write(a.out/"COMPLETE.json",dict(fit_images=len(selected),fit_candidates=n,
        dev_images=len(grouped),dev_candidates=len(dev),max_h_error=max_h,
        max_c_error=max_c,max_box_error=max_box,train_test_image_disjoint=True))
    print("COMPLETE",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    p.add_argument("--count",type=int,default=0);main(p.parse_args())

