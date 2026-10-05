"""Only new algorithmic variable: auxiliary coefficient follows official O2M."""
import argparse,csv,json,math,random,shutil
from pathlib import Path
import method_core as core
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO,settings
from ultralytics.nn.tasks import SegmentationModel
from recording import atomic_json,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=json.loads((a.study/'protocol.json').read_text())
    assert ultralytics.__version__==cfg['ultralytics']
    assert not (a.out/'args.yaml').exists(),'Fresh training only; no implicit resume'
    amp=Path(cfg['amp_cache']);destination=Path.cwd()/'yolo26n.pt'
    if not destination.exists():shutil.copyfile(amp,destination)
    core.SOURCE_WEIGHT=Path(cfg['weight']);core.METHOD_ENABLED=True
    core.AUX_WEIGHT=cfg['aux_weight'];core.EXTRA_CLASS_WEIGHT=0.;core.FOLLOW_O2M=True
    SegmentationModel.loss=core.patched_loss
    settings.update({k:False for k in ['wandb','comet','mlflow','clearml','neptune'] if k in settings})
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    inherited=core.official_weight_config(core.SOURCE_WEIGHT)
    assert inherited.get('mask_ratio')==1 and inherited.get('overlap_mask') is True
    data=cfg['smoke_data'] if a.smoke else cfg['data']
    train_cfg=dict(inherited)
    train_cfg.update(data=data,epochs=cfg['epochs'],batch=cfg['batch'],workers=0 if a.smoke else cfg['workers'],
                     device=0,project=str(a.out.parent),name=a.out.name,exist_ok=True,seed=cfg['seed'],save=True,save_period=1,
                     val=True,plots=False,cache=False,patience=0,verbose=True,resume=False)
    reference=Path(cfg['reference_remote_root'])/'runs'/cfg['reference_training_run']
    reference_args=yaml.safe_load((reference/'args.yaml').read_text())
    if not a.smoke:
        for key,value in train_cfg.items():
            if key not in ['project','name']:
                assert str(reference_args.get(key))==str(value),(key,reference_args.get(key),value)
    atomic_json(a.out/'design.json',dict(arm='reg_scheduled',training_config=train_cfg,source_checkpoint_training_config=inherited,
                 schedule=cfg['schedule'],initial_aux_weight=.5,extra_class_weight=0,reference_run=cfg['reference_training_run'],
                 teacher='frozen initial pretrained student; no resume',only_method_change='fixed auxiliary coefficient replaced by official-O2M-proportional coefficient'))
    model=YOLO(str(core.SOURCE_WEIGHT));previous=dict(core.RUN_STATS);epochs=[]
    def epoch_end(trainer):
        nonlocal previous
        total=dict(core.RUN_STATS);delta={k:total[k]-previous[k] for k in total};previous=total
        observed=[dict(official_o2m=k[0],aux_weight=k[1],batches=v) for k,v in core.SCHEDULE_OBSERVED.items()]
        core.SCHEDULE_OBSERVED.clear()
        expected=cfg['expected_lambda'][trainer.epoch]
        assert len(observed)==1 and math.isclose(observed[0]['aux_weight'],expected,abs_tol=1e-10),observed
        row=dict(epoch=trainer.epoch+1,at=now(),auxiliary=delta,observed_training_weights=observed)
        epochs.append(row);atomic_json(a.out/'auxiliary_epochs.json',epochs)
        print('AUX_EPOCH',json.dumps(row),flush=True)
    model.add_callback('on_train_epoch_end',epoch_end)
    model.train(trainer=core.CounterfactualTrainer,**train_cfg)
    assert Path(model.trainer.save_dir).resolve()==a.out.resolve()
    with (a.out/'results.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==cfg['epochs']
    for row in rows:
        for key,value in row.items():
            if key and value:assert math.isfinite(float(value)),(key,value)
    assert (a.out/'weights/last.pt').exists() and (a.out/'weights/best.pt').exists()
    if not a.smoke:
        actual=yaml.safe_load((a.out/'args.yaml').read_text())
        differences={k:[reference_args.get(k),v] for k,v in actual.items() if reference_args.get(k)!=v}
        assert set(differences)<={'name','project','save_dir'},differences
    else:differences={'scope':'smoke uses 32 train / 16 val images and workers=0'}
    first_epoch_reference=None
    if not a.smoke:
        ref=json.loads((reference/'auxiliary_epochs.json').read_text())[0]['auxiliary']
        first_epoch_reference={k:dict(reference=ref[k],new=epochs[0]['auxiliary'][k]) for k in ref}
    receipt=dict(status='complete',arm='reg_scheduled',epochs=len(rows),finite=True,seed=cfg['seed'],finished_at=now(),
                 source_weight=cfg['weight'],data=data,runtime_ultralytics=ultralytics.__version__,method_stats=core.RUN_STATS,
                 observed_schedule=[r['observed_training_weights'] for r in epochs],first_epoch_reference=first_epoch_reference,
                 config_differences=differences,checkpoint_selection='last primary; native best supplementary')
    atomic_json(a.out/'TRAINING_COMPLETE.json',receipt);print('TRAINING_COMPLETE',json.dumps(receipt),flush=True)

if __name__=='__main__':main()
