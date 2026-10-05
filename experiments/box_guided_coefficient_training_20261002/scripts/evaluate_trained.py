"""Fixed final checkpoints, full prototypes, same official candidates."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO

from box_guided_head import CoefficientReadout
from train_native import setup,load,feed,gpu_image,read_image,dump,loss_and_grad
from evaluate import evaluate_image,append_rows,summarize


def donor_boxes(x):
    boxes=x['boxes'];result=boxes.clone(); classes=x['predicted_classes'].tolist()
    geometry=[]
    for b in boxes.tolist():
        w,h=b[2]-b[0],b[3]-b[1]
        geometry.append((w*h,w/h) if w>0 and h>0 else None)
    for i,row in enumerate(x['rows']):
        valid=[j for j in range(len(boxes)) if j!=i and geometry[j] is not None]
        same=[j for j in valid if classes[j]==classes[i]]
        pool=same or valid
        row['wrong_box_available']=bool(pool and geometry[i] is not None)
        if not row['wrong_box_available']:continue
        area,ratio=geometry[i]
        j=min(pool,key=lambda j:(abs(math.log(geometry[j][0]/area))+abs(math.log(geometry[j][1]/ratio)),int(x['raw_ids'][j])))
        result[i]=boxes[j]
        row.update(wrong_box_donor_annotation_id=int(x['rows'][j]['annotation_id']),
            wrong_box_donor_raw_id=int(x['raw_ids'][j]),wrong_box_donor_image_id=int(row['image_id']))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--checkpoints',required=True,help='JSON arm->final checkpoint paths')
    p.add_argument('--index');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    setup(cfg['seed']);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    index=json.loads(Path(a.index or str(Path(cfg['cache'])/'INDEX.json')).read_text())
    source=YOLO(cfg['weights']).model.eval().requires_grad_(False)
    paths=json.loads(Path(a.checkpoints).read_text());models={}
    for mode in ('N','P','R'):
        model=CoefficientReadout(source.model[-1].one2one_cv4,mode).cuda().eval()
        if not a.smoke:
            ck=load(paths[mode]);assert ck['mode']==mode and ck['epoch']==cfg['epochs']
            model.load_state_dict(ck['state_dict'])
        model.requires_grad_(False);models[mode]=model
    del source
    coco={'fit':COCO(cfg['annotations_train']),'val':COCO(cfg['annotations_val'])};coco['dev']=coco['fit']
    rows_path=out/'PER_CANDIDATE.jsonl';allrows=[];done=set();loss_sums={}
    if rows_path.exists():
        raise RuntimeError('Use a fresh independent evaluation Run; do not append over partial candidate output')
    expected={};elapsed=time.monotonic()
    for split,items in index.items():
        expected[split]=dict(planned_images=len(items),effective_images=sum(r['n']>0 for r in items),candidates=sum(r['n'] for r in items))
        loss_sums[split]={arm:0. for arm in ('A','N','P','R')};count=0
        for k,item in enumerate(items):
            if not item['n']:continue
            x=read_image(cfg,item['image_id']);fs,selected=feed([x]);wrong=donor_boxes(x)
            with torch.no_grad():
                coefficients={arm:model(fs,selected)[0] for arm,model in models.items()}
                coefficients['A']=x['c0'].cuda()
                wrongsel=[dict(selected[0],sampling_boxes=wrong.cuda())]
                coefficients['RW']=models['R'](fs,wrongsel)[0]
            gx=gpu_image(x)
            if a.smoke:
                for arm,c in coefficients.items():
                    torch.testing.assert_close(c.cpu(),x['c0'],atol=2e-5,rtol=2e-5)
            for arm in ('A','N','P','R'):
                value,_=loss_and_grad(gx,coefficients[arm],len(x['rows']))
                loss_sums[split][arm]+=value
            rs=evaluate_image(gx,coefficients,coco[split]);append_rows(rows_path,rs);allrows.extend(rs);count+=len(rs)
            if k%25==0 or k+1==len(items):
                state=dict(stage='evaluation',split=split,images=k+1,total=len(items),candidates=count,elapsed_s=time.monotonic()-elapsed)
                dump(out/'PROGRESS.json',state);print(json.dumps(state),flush=True)
            del x,gx,fs,coefficients
        n=expected[split]['candidates']
        loss_sums[split]={arm:v/max(1,n) for arm,v in loss_sums[split].items()}
    dump(out/'FINAL_BCE.json',loss_sums)
    result=summarize(allrows,out,expected_counts=expected,seed=20261002,
        bootstrap=100 if a.smoke else cfg['bootstrap'],run_info=dict(checkpoints=paths,
        checkpoint_rule=cfg['checkpoint_rule'],kind='decoder code smoke, untrained' if a.smoke else 'fixed final training evaluation',
        evaluation_scope='held out from this training, historically studied cohort; not a new blind test or AP'))
    dump(out/'COMPLETE.json',dict(completed=True,smoke=a.smoke,candidates=len(allrows),elapsed_s=time.monotonic()-elapsed))


if __name__=='__main__':main()
