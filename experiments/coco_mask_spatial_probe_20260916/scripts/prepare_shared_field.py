"""Average learned fields on the original 200-image training selection split only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
import learn_refinement as original
from component_seed_probe import setup


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    for k in ['bank','split','weights','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);setup()
    bank=torch.load(a.bank,map_location='cpu',mmap=True,weights_only=False)
    ids=set(json.loads(a.split.read_text())['selection_image_ids'])
    owners=defaultdict(set)
    for r in bank['records']:owners[(r['image_id'],r['raw_id'])].add(r['annotation_id'])
    seen=set();ix=[]
    for i,r in enumerate(bank['records']):
        k=(r['image_id'],r['raw_id'])
        if r['image_id'] not in ids or len(owners[k])!=1 or k in seen:continue
        seen.add(k);ix.append(i)
    assert len(ix)==3723
    net=original.Refiner('local4').cuda().eval();net.load_state_dict(torch.load(a.weights,weights_only=False)['state_dict'])
    byimage=defaultdict(list)
    for j in range(0,len(ix),128):
        pick=ix[j:j+128];out=(net(bank['x'][pick].cuda().float())*.5).cpu()
        for i,v in zip(pick,out):byimage[bank['records'][i]['image_id']].append(v)
    field=torch.stack([torch.stack(v).mean(0) for v in byimage.values()]).mean(0).reshape(4,4)
    result={'field':field.tolist(),'images_with_valid_candidates':len(byimage),'selection_images':200,'candidates':len(ix),
            'averaging':'per-image mean, then mean across original train2017 selection images; no val or GT outcomes',
            'alpha':.5,'application':'constant shape per ROI; match each original local field crop mean/std'}
    original.save(a.out/'SHARED_FIELD.json',result);original.save(a.out/'COMPLETE.json',{'complete':True,'candidates':len(ix)});print(json.dumps(result))


if __name__=='__main__':main()
