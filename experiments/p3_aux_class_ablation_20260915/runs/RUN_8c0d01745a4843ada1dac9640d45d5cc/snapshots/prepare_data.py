"""Select images only; reuse existing YOLO segmentation labels unchanged."""
import argparse, json, random
from pathlib import Path
import yaml
from recording import atomic_json, now

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads((a.study/'protocol.json').read_text())
    root=Path(cfg['coco_root']);data=a.out/'data';data.mkdir()
    original=yaml.safe_load((root/'coco_full.yaml').read_text())
    images=sorted((root/'images/train2017').glob('*.jpg'))
    assert len(images)==118287
    chosen=sorted(random.Random(cfg['subset_seed']).sample(images,cfg['train_images']))
    vals=sorted((root/'images/val2017').glob('*.jpg'))
    assert len(vals)==5000
    assert {p.stem for p in chosen}.isdisjoint({p.stem for p in vals})
    train=data/'train.txt';train.write_text(''.join(str(p)+'\n' for p in chosen))
    smoke_train=data/'smoke_train.txt';smoke_train.write_text(''.join(str(p)+'\n' for p in chosen[:32]))
    smoke_val=data/'smoke_val.txt';smoke_val.write_text(''.join(str(p)+'\n' for p in vals[:16]))
    for name,tr,va in [('data.yaml',train,root/'images/val2017'),('smoke.yaml',smoke_train,smoke_val)]:
        (data/name).write_text(yaml.safe_dump(dict(path=str(root),train=str(tr),val=str(va),names=original['names']),sort_keys=False))
    atomic_json(a.out/'COMPLETE.json',dict(status='complete',completed_at=now(),training_images=len(chosen),validation_images=len(vals),subset_seed=cfg['subset_seed'],selection='uniform without replacement from sorted train2017 paths',label_conversion=False,smoke_train=32,smoke_val=16))
    print('DATA_READY',data,flush=True)

if __name__=='__main__':main()
