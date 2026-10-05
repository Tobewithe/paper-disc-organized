"""Descriptive held-out training-image curve; does not change epoch8 protocol."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
from learn_refinement import Refiner,roi_correct,save
from repair_refinement import per_loss,hard_iou


def main():
    p=argparse.ArgumentParser()
    for k in ('bank','models','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    bank=torch.load(a.bank,map_location='cpu',weights_only=False,mmap=True);split=json.loads((a.models/'SPLIT.json').read_text())
    ids=set(split['selection_image_ids']);owners=defaultdict(set)
    for r in bank['records']:owners[(r['image_id'],r['raw_id'])].add(r['annotation_id'])
    seen=set();keep=[]
    for i,r in enumerate(bank['records']):
        key=(r['image_id'],r['raw_id'])
        if r['image_id'] in ids and len(owners[key])==1 and key not in seen:keep.append(i);seen.add(key)
    data=TensorDataset(*(bank[k][keep] for k in ('x','base','target')));result={}
    with torch.no_grad():
        for mode,kind in [('scalar','plain'),('coeff_local4','plain'),('coeff_local4','safe')]:
            key=f'{mode}_{kind}';result[key]=[]
            for epoch in range(1,9):
                net=Refiner(mode).cuda().eval();net.load_state_dict(torch.load(a.models/f'{key}_epoch{epoch}.pt',weights_only=False)['state_dict'])
                total=loss=delta=saturation=0.;n=damaged=repaired=loss_better_iou_worse=0
                for x,b,y in DataLoader(data,batch_size=128):
                    x=x.cuda().float();b=b.cuda().float();y=y.cuda().float();c=net(x);z=roi_correct(b,x,c,mode)
                    bi=hard_iou(b,y);zi=hard_iou(z,y);bl=per_loss(b,y);zl=per_loss(z,y)
                    n+=len(x);total+=float(zi.sum());loss+=float(zl.sum());delta+=float((z-b).abs().mean((1,2)).sum())
                    saturation+=float((c.abs()>.95).float().mean(1).sum());damaged+=int(((bi>=.75)&(zi<.75)).sum());repaired+=int(((bi<.75)&(zi>=.75)).sum())
                    loss_better_iou_worse+=int(((zl<bl)&(zi<bi)).sum())
                result[key].append(dict(epoch=epoch,n=n,mean_iou=total/n,mean_loss=loss/n,mean_abs_delta=delta/n,saturation_fraction=saturation/n,
                    damaged75=damaged,repaired75=repaired,loss_better_iou_worse=loss_better_iou_worse))
            save(a.out/'CURVES.json',result);print(key,json.dumps(result[key]),flush=True)
    save(a.out/'COMPLETE.json',dict(selection_rows=len(keep),epochs=8,checkpoint_policy='Descriptive only; primary evaluation remains epoch8.'))


if __name__=='__main__':main()
