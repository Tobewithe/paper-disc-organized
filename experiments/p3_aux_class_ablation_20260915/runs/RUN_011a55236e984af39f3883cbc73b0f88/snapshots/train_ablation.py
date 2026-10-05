"""Paired screen; original vs reg_only differs only in the extra class BCE."""
from __future__ import annotations
import argparse, csv, json, math, random, shutil
from pathlib import Path
import method_core as core
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO, settings
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from recording import atomic_json, now

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--arm',choices=['baseline','original','reg_only'],required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=json.loads((a.study/'protocol.json').read_text())
    assert ultralytics.__version__==cfg['ultralytics']
    assert not (a.out/'args.yaml').exists(), 'Fresh runs only; resume requires explicitly restoring fixed teacher'
    # Reuse the official AMP test model; this does not bypass its numerical check.
    if not a.smoke:
        smoke=next(r for r in cfg['runs'] if r['kind']=='smoke')
        amp_cache=a.study/'runs'/smoke['run_id']/'snapshots/yolo26n.pt'
        if amp_cache.is_file():
            assert Path.cwd().resolve()==(a.out/'snapshots').resolve()
            destination=Path.cwd()/'yolo26n.pt'
            if not destination.exists():shutil.copyfile(amp_cache,destination)
    core.SOURCE_WEIGHT=Path(cfg['weight'])
    core.METHOD_ENABLED=a.arm!='baseline'
    core.AUX_WEIGHT=cfg['aux_weight']
    core.EXTRA_CLASS_WEIGHT=cfg['arms'][a.arm]['extra_class_weight']
    SegmentationModel.loss=core.patched_loss if core.METHOD_ENABLED else core.ORIGINAL_LOSS
    settings.update({k:False for k in ['wandb','comet','mlflow','clearml','neptune'] if k in settings})
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    inherited=core.official_weight_config(core.SOURCE_WEIGHT)
    assert inherited.get('mask_ratio')==1 and inherited.get('overlap_mask') is True
    train_cfg=dict(inherited)
    train_cfg.update(data=str(a.data),epochs=1 if a.smoke else cfg['epochs'],batch=cfg['batch'],
                     workers=0 if a.smoke else cfg['workers'],device=0,project=str(a.out.parent),name=a.out.name,
                     exist_ok=True,seed=cfg['seed'],save=True,save_period=1,val=True,plots=False,cache=False,
                     patience=0,verbose=True,resume=False)
    atomic_json(a.out/'design.json',dict(arm=a.arm,training_config=train_cfg,source_checkpoint_training_config=inherited,
                 auxiliary_weight=core.AUX_WEIGHT if core.METHOD_ENABLED else 0,
                 extra_class_weight=core.EXTRA_CLASS_WEIGHT,teacher='frozen deepcopy of freshly loaded initial student; never resumed',
                 baseline_trainer='official SegmentationTrainer',primary_checkpoint='last.pt'))
    model=YOLO(str(core.SOURCE_WEIGHT))
    previous=dict(core.RUN_STATS);epochs=[]
    def epoch_end(trainer):
        nonlocal previous
        total=dict(core.RUN_STATS);delta={k:total[k]-previous[k] for k in total};previous=total
        row=dict(epoch=trainer.epoch+1,at=now(),auxiliary=delta)
        for branch in ['one2many','one2one']:
            # Observed loss weights are retained if exposed by the installed criterion.
            value=getattr(getattr(trainer.model,'criterion',None),branch,None)
            if value is not None:row[branch+'_criterion']=type(value).__name__
        epochs.append(row);atomic_json(a.out/'auxiliary_epochs.json',epochs)
        print('AUX_EPOCH',json.dumps(row),flush=True)
    model.add_callback('on_train_epoch_end',epoch_end)
    model.train(trainer=core.CounterfactualTrainer if core.METHOD_ENABLED else SegmentationTrainer,**train_cfg)
    assert Path(model.trainer.save_dir).resolve()==a.out.resolve()
    results=list(csv.DictReader((a.out/'results.csv').open()))
    for row in results:
        for key,value in row.items():
            if key and value:assert math.isfinite(float(value)),(key,value)
    assert len(results)==train_cfg['epochs']
    assert (a.out/'weights/last.pt').exists() and (a.out/'weights/best.pt').exists()
    receipt=dict(status='complete',arm=a.arm,epochs=len(results),finite=True,seed=cfg['seed'],
                 finished_at=now(),source_weight=cfg['weight'],method_stats=core.RUN_STATS,
                 extra_class_weight=core.EXTRA_CLASS_WEIGHT,aux_weight=core.AUX_WEIGHT if core.METHOD_ENABLED else 0,
                 runtime_ultralytics=ultralytics.__version__,data=str(a.data),checkpoint_selection='last primary; native best secondary')
    atomic_json(a.out/'TRAINING_COMPLETE.json',receipt)
    print('TRAINING_COMPLETE',json.dumps(receipt),flush=True)

if __name__=='__main__':main()
