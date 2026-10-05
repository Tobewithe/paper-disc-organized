"""Nested train-size curve for the frozen h-only coefficient head."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time
import zlib

import numpy as np
from pycocotools.coco import COCO
import torch


def main(a):
    sys.path.insert(0, str(a.source / "scripts"))
    from official_pipeline import setup, load, target_rois, roi_losses, decode, write
    from feature_probes import Probe
    setup()
    training_bank = a.train_bank or a.bank
    index = json.loads((training_bank / "INDEX.json").read_text())
    if a.order:
        order = json.loads(a.order.read_text())["images"]
        anchor_ids = order[:16]
    else:
        anchor_ids = json.loads((a.source / "runs/RUN_acd0d817ff7c478982746a91452fd68e/SELECTION.json").read_text())["images"]
        remaining = sorted({r["image_id"] for r in index["fit"]}-set(anchor_ids))
        random.Random(20260924).shuffle(remaining)
        order = anchor_ids + remaining
    fit = {r["image_id"]:r for r in index["fit"]}
    assert len(anchor_ids) == 16 and all(i in fit for i in anchor_ids)
    selected = order[:a.images]
    assert len(selected) == a.images and all(i in fit for i in selected)
    def training_image(iid):
        image=load(training_bank/"images"/f"{iid:012d}.pt")
        if "mask_zlib" in image:
            mask=np.frombuffer(zlib.decompress(image.pop("mask_zlib")),dtype="<u2").copy()
            image["masks"]=torch.from_numpy(mask.reshape(image.pop("mask_shape")).astype(np.int64))
            image["proto"]=image["proto"].float()
        return image
    write(a.out/"SELECTION.json",dict(images=selected,anchor_images=anchor_ids,
        source="Official frozen cached fit images, deterministic nested order, positives only"))
    hs,cs=[],[]
    for iid in selected:
        if not fit[iid]["n"]:continue
        image=training_image(iid)
        ids=[r["raw_id"] for r in image["rows"]]
        if not ids:continue
        hs.append(image["h"][ids]);cs.append(image["coeff"][ids])
    h,c=torch.cat(hs),torch.cat(cs)
    nullp=torch.zeros(512)
    net=Probe("mlp_large",h.mean(0),h.std(0).clamp_min(.01),nullp,torch.ones_like(nullp),
              c.std(0).clamp_min(.1)).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    def loss_on_image(image):
        ids=torch.tensor([r["raw_id"] for r in image["rows"]])
        if not len(ids):
            return None,0
        features=image["h"][ids].cuda()
        coefficients=image["coeff"][ids].cuda()
        levels=image["levels"][ids].cuda()
        predicted=coefficients+net(features,torch.zeros(len(ids),512,device="cuda"),levels)
        return roi_losses(predicted[None],target_rois(image))[0],len(ids)
    def mean_loss(ids, training=False, dataset="train"):
        total,n=0.0,0
        if training:
            sequence=[i for i in ids if fit[i]["n"]>0]
            random.Random(20260924+epoch).shuffle(sequence)
            for start in range(0,len(sequence),8):
                batch=sequence[start:start+8]
                count=sum(fit[i]["n"] for i in batch)
                optimizer.zero_grad(set_to_none=True)
                for iid in batch:
                    image=training_image(iid)
                    value,k=loss_on_image(image)
                    (value*image["segmentation_gain"]/count).backward()
                    total+=float(value.detach());n+=k
                torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True)
                optimizer.step()
            return total/n
        with torch.no_grad():
            for iid in ids:
                image=training_image(iid) if dataset=="train" else load(a.bank/"images"/f"{iid:012d}.pt")
                value,k=loss_on_image(image)
                if not k:continue
                total+=float(value);n+=k
        return total/n
    eval_index=json.loads((a.bank/"INDEX.json").read_text())
    dev=[r["image_id"] for r in eval_index["dev"] if r["n"]>0]
    def original_mean(ids):
        total,n=0.0,0
        with torch.no_grad():
            for iid in ids:
                image=load(a.bank/"images"/f"{iid:012d}.pt")
                chosen=torch.tensor([r["raw_id"] for r in image["rows"]])
                if not len(chosen):continue
                coeff=image["coeff"][chosen].cuda()
                total+=float(roi_losses(coeff[None],target_rois(image))[0])
                n+=len(chosen)
        return total/n
    dev_original=original_mean(dev)
    start=time.monotonic()
    best=dev_original
    chosen_epoch=0
    history=[]
    stale=0
    for epoch in range(1,a.epochs+1):
        net.train()
        train=mean_loss(selected,training=True)
        net.eval()
        dev_loss=mean_loss(dev,dataset="eval")
        row=dict(epoch=epoch,train_loss=train,dev_loss=dev_loss,dev_original=dev_original,seconds=time.monotonic()-start)
        history.append(row)
        write(a.out/"HISTORY.json",history)
        write(a.out/"PROGRESS.json",row)
        print(json.dumps(row),flush=True)
        torch.save(dict(state_dict=net.state_dict(),epoch=epoch,dev_loss=dev_loss),a.out/f"epoch{epoch}.pt")
        if dev_loss < best:
            best=dev_loss;chosen_epoch=epoch;stale=0
            torch.save(dict(state_dict=net.state_dict(),epoch=epoch,dev_loss=dev_loss),a.out/"best.pt")
        else:stale+=1
        if stale >= a.patience and epoch>=a.min_epochs:break
    last_epoch=history[-1]["epoch"]
    last_state={name:value.detach().cpu().clone() for name,value in net.state_dict().items()}
    if chosen_epoch:
        net.load_state_dict(load(a.out/"best.pt")["state_dict"])
    net.eval()
    anchor_path=a.anchor_oracle or a.source/"runs/RUN_acd0d817ff7c478982746a91452fd68e/ORACLE_ROWS.json"
    anchor_oracle=json.loads(anchor_path.read_text())
    val_oracle=json.loads((a.source/"runs/RUN_0c62bc2c9d8a48d78af75436f0d0eef0/ROWS.json").read_text())
    def gap(rows, dataset):
        by_image=defaultdict(list)
        for row in rows:by_image[row["image_id"]].append(row)
        before=sum(row["initial_loss"] if "initial_loss" in row else row["original_loss"] for row in rows)
        optimum=sum(row["oracle_loss"] for row in rows)
        predicted=0.0
        with torch.no_grad():
            for iid,group in by_image.items():
                image=training_image(iid) if dataset=="train" else load(a.bank/"images"/f"{iid:012d}.pt")
                indices={r["annotation_id"]:j for j,r in enumerate(image["rows"])}
                ids=[indices[r["annotation_id"]] for r in group]
                if chosen_epoch:
                    selected_rows=[image["rows"][j] for j in ids]
                    features=image["h"][[r["raw_id"] for r in selected_rows]].cuda()
                    coeff=image["coeff"][[r["raw_id"] for r in selected_rows]].cuda()
                    levels=image["levels"][[r["raw_id"] for r in selected_rows]].cuda()
                    coeff=coeff+net(features,torch.zeros(len(ids),512,device="cuda"),levels)
                else:
                    coeff=image["coeff"][[image["rows"][j]["raw_id"] for j in ids]].cuda()
                predicted+=float(roi_losses(coeff[None],target_rois(image,ids))[0])
        return dict(n=len(rows),original_loss=before/len(rows),oracle_loss=optimum/len(rows),
                    selected_loss=predicted/len(rows),gap_closed=(before-predicted)/(before-optimum))
    gaps=dict(train_anchor=gap(anchor_oracle,"train"),val_oracle=gap(val_oracle,"eval"))
    net.load_state_dict(last_state)
    net.eval()
    chosen_before_last=chosen_epoch
    chosen_epoch=last_epoch
    final_gaps=dict(train_anchor=gap(anchor_oracle,"train"),val_oracle=gap(val_oracle,"eval"))
    chosen_epoch=chosen_before_last
    coco=COCO(str(a.source/"data/annotations/instances_val2017.json"))
    val_rows=[]
    if chosen_epoch:
        net.load_state_dict(load(a.out/"best.pt")["state_dict"])
    with torch.no_grad():
        for iid in [r["image_id"] for r in eval_index["val"]]:
            image=load(a.bank/"images"/f"{iid:012d}.pt")
            rows=image["rows"]
            if not rows:continue
            ids=torch.tensor([r["raw_id"] for r in rows])
            coeff=image["coeff"][ids].cuda()
            if chosen_epoch:
                candidate=coeff+net(image["h"][ids].cuda(),
                   torch.zeros(len(ids),512,device="cuda"),image["levels"][ids].cuda())
            else:candidate=coeff
            masks=decode(image,candidate,ids)
            for row,mask in zip(rows,masks):
                gt=coco.annToMask(coco.anns[row["annotation_id"]]).astype(bool)
                after=float((mask & gt).sum()/max((mask | gt).sum(),1))
                val_rows.append(dict(image_id=iid,annotation_id=row["annotation_id"],before=row["initial_iou"],after=after))
    before=np.asarray([r["before"] for r in val_rows])
    after=np.asarray([r["after"] for r in val_rows])
    output=dict(images=a.images,chosen_epoch=chosen_epoch,last_epoch=last_epoch,
        dev_original=dev_original,best_dev_loss=best,last_checkpoint_gaps=final_gaps,
        train_anchor=gaps["train_anchor"],val_oracle=gaps["val_oracle"],
        val_all=dict(n=len(val_rows),mean_iou_original=float(before.mean()),mean_iou_selected=float(after.mean()),
         mask75_original=int((before>=.75).sum()),mask75_selected=int((after>=.75).sum()),
         repairs=int(((before<.75)&(after>=.75)).sum()),damages=int(((before>=.75)&(after<.75)).sum())),
        scope="Single-seed frozen one2one candidate diagnostic, not COCO AP")
    write(a.out/"VAL_ROWS.json",val_rows)
    write(a.out/"SUMMARY.json",output)
    write(a.out/"COMPLETE.json",dict(images=a.images,epochs_run=len(history),chosen_epoch=chosen_epoch))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("source","bank","out"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--images",type=int,required=True)
    p.add_argument("--train-bank",type=Path)
    p.add_argument("--order",type=Path)
    p.add_argument("--anchor-oracle",type=Path)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--patience",type=int,default=10)
    p.add_argument("--min-epochs",type=int,default=1)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    main(args)
