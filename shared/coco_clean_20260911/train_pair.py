"""Stock supervised YOLO trainer, CCL injection and experiment-integrity guards."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parent
# These must be set before numerical-library imports and worker construction.
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('WANDB_MODE','disabled')

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO, settings
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ccl_loss import CCLSegmentationLoss,install_ccl

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x

def finite_state(model):
    return all(torch.isfinite(t).all() for t in model.state_dict().values() if isinstance(t,torch.Tensor) and t.is_floating_point())

class GuardedSegmentationTrainer(SegmentationTrainer):
    """No data/loss/optimizer recipe changes: fail on budget drift/nonfinite state."""
    def _build_train_pipeline(self):
        if hasattr(self,'_expected_batch') and self.batch_size!=self._expected_batch:
            raise RuntimeError('Stock OOM retry changed batch; stop for paired protocol review')
        self._expected_batch=self.batch_size
        return super()._build_train_pipeline()

    def optimizer_step(self):
        # Same stock order and clip norm; fail before an invalid optimizer step.
        self.scaler.unscale_(self.optimizer)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.0,error_if_nonfinite=True)
        self._last_grad_norm=float(norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:self.ema.update(self.model)
        self._optimizer_steps=getattr(self,'_optimizer_steps',0)+1

    def preprocess_batch(self,batch):
        # Hash the first two actual batches before transfer, for paired-seed replay.
        counter=getattr(self,'_trace_batches',0)
        if counter<2:
            h=hashlib.sha256()
            for k in ['img','cls','bboxes','masks','batch_idx']:
                v=batch[k].detach().cpu().contiguous()
                h.update(k.encode()+str(tuple(v.shape)).encode()+v.numpy().tobytes())
            self.save_dir.mkdir(parents=True,exist_ok=True)
            with (self.save_dir/'initial_batches.jsonl').open('a') as f:
                f.write(json.dumps(dict(batch=counter,sha256=h.hexdigest(),image_files=batch['im_file'],instances=len(batch['cls'])))+'\n')
            self._trace_batches=counter+1
        return super().preprocess_batch(batch)

    def save_model(self):
        if not finite_state(self.model) or not finite_state(self.ema.ema):
            raise FloatingPointError('Nonfinite live/EMA weights: do not serialize repaired weights')
        for tensor in self.ema.ema.state_dict().values():
            if isinstance(tensor,torch.Tensor) and tensor.is_floating_point() and tensor.abs().max()>65504:
                raise FloatingPointError('FP16 checkpoint overflow: preserve run failure for diagnosis')
        result=super().save_model()
        # Preserve full-precision live model, EMA, optimizer and RNG at each epoch
        # as the latest recovery state. Per-epoch official files are all retained.
        recovery=dict(epoch=self.epoch,model=cpu_tree(self.model.state_dict()),ema=cpu_tree(self.ema.ema.state_dict()),
            optimizer=cpu_tree(self.optimizer.state_dict()),scaler=self.scaler.state_dict(),scheduler=self.scheduler.state_dict(),
            ema_updates=self.ema.updates,args=vars(self.args),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),optimizer_steps=getattr(self,'_optimizer_steps',0),
            note='Full precision recovery material; restarting worker RNG mid-epoch is not claimed bitwise reproducible.')
        temp=self.wdir/'recovery_latest.tmp';torch.save(recovery,temp);temp.replace(self.wdir/'recovery_latest.pt')
        index=[]
        for path in sorted(self.wdir.glob('epoch*.pt')):
            index.append(dict(file=path.name,completed_epoch=int(path.stem[5:])+1,bytes=path.stat().st_size))
        (self.save_dir/'checkpoint_index.json').write_text(json.dumps(index,indent=2))
        return result

def train_start(trainer):
    criterion=trainer.model.init_criterion()
    trainer.model.criterion=criterion
    expected=float(os.environ['CCL_WEIGHT'])
    assert isinstance(criterion.one2many,CCLSegmentationLoss) and isinstance(criterion.one2one,CCLSegmentationLoss)
    assert criterion.one2many.ccl_weight==criterion.one2one.ccl_weight==expected
    actual_train=len(trainer.train_loader.dataset);actual_val=len(trainer.test_loader.dataset)
    expected_images=64 if os.environ.get('COCO_SMOKE')=='1' else 37433
    assert actual_train==expected_images,(actual_train,expected_images)
    assert actual_val==(32 if os.environ.get('COCO_SMOKE')=='1' else 5000)
    receipt=dict(seed=trainer.args.seed,ccl_weight=expected,epochs=trainer.epochs,batch=trainer.batch_size,
        train_images=actual_train,val_images=actual_val,train_instances=sum(len(x['cls']) for x in trainer.train_loader.dataset.labels),
        val_instances=sum(len(x['cls']) for x in trainer.test_loader.dataset.labels),
        optimizer=type(trainer.optimizer).__name__,learning_rates=[g['lr'] for g in trainer.optimizer.param_groups],
        warmup_bias_lr=trainer.args.warmup_bias_lr,save_period=trainer.args.save_period,
        parameter_sha256=hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in trainer.model.state_dict().values())).hexdigest(),
        criterion_branches=[type(criterion.one2many).__name__,type(criterion.one2one).__name__],
        resolved_args=vars(trainer.args))
    (trainer.save_dir/'launch_receipt.json').write_text(json.dumps(receipt,indent=2,default=str))
    print('TRAIN_LAUNCH_VERIFIED',json.dumps({k:v for k,v in receipt.items() if k!='resolved_args'}),flush=True)

def epoch_start(trainer):
    trainer._epoch_start=time.monotonic()
    for branch in [trainer.model.criterion.one2many,trainer.model.criterion.one2one]:
        branch.ccl_stats=dict(calls=0,pairs=0,active_pairs=0,raw_sum=0.0)

def batch_end(trainer):
    if not torch.isfinite(trainer.loss):raise FloatingPointError('Nonfinite batch loss')
    if trainer.batch_size!=trainer._expected_batch:raise RuntimeError('Unexpected batch-budget change')

def epoch_end(trainer):
    c=trainer.model.criterion
    record=dict(completed_epoch=trainer.epoch+1,utc=datetime.now(timezone.utc).isoformat(),elapsed_training_seconds=time.monotonic()-trainer._epoch_start,
        loss={k:float(v) for k,v in trainer.tloss.items()},optimizer_steps=getattr(trainer,'_optimizer_steps',0),
        last_grad_norm=getattr(trainer,'_last_grad_norm',None),o2m=dict(c.one2many.ccl_stats),o2o=dict(c.one2one.ccl_stats))
    if float(os.environ['CCL_WEIGHT'])>0 and (record['o2m']['pairs']==0 or record['o2o']['pairs']==0):
        raise RuntimeError('CCL had no eligible pairs in a complete epoch')
    with (trainer.save_dir/'epoch_integrity.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print('EPOCH_INTEGRITY',json.dumps(record),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    parser.add_argument('--arm',choices=['baseline','ccl01'],required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    os.environ['CCL_WEIGHT']='0.1' if args.arm=='ccl01' else '0'
    os.environ['COCO_SMOKE']='1' if args.smoke else '0'
    torch.set_num_threads(4)
    assert torch.__version__=='2.8.0+cu128' and ultralytics.__version__=='8.4.143' and np.__version__=='2.2.6'
    settings.update({k:False for k in ['wandb','comet','mlflow','clearml','neptune'] if k in settings})
    cfg=json.loads((ROOT/'train_config.json').read_text());cfg.update(seed=args.seed,name=f'{args.arm}_s{args.seed}')
    if args.smoke:
        import yaml
        data=yaml.safe_load((ROOT/'coco_clean.yaml').read_text());data.update(train=str(ROOT/'smoke_train.txt'),val=str(ROOT/'smoke_val.txt'))
        (ROOT/'smoke_data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
        cfg.update(data=str(ROOT/'smoke_data.yaml'),epochs=1,close_mosaic=0,project=str(ROOT/'smoke_runs'))
    else:
        for gate in ['DATA_PIPELINE_PASS.json','CCL_GRADIENT_PASS.json','LAUNCH_GATE.json']:
            receipt=json.loads((ROOT/'audits'/gate).read_text());assert receipt['status']=='PASS',gate
    assert sha(cfg['model'])=='16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5'
    install_ccl()
    model=YOLO(cfg.pop('model'))
    for event,callback in [('on_train_start',train_start),('on_train_epoch_start',epoch_start),('on_train_batch_end',batch_end),('on_train_epoch_end',epoch_end)]:
        model.add_callback(event,callback)
    model.train(trainer=GuardedSegmentationTrainer,**cfg)
    save_dir=Path(model.trainer.save_dir)
    files=list((save_dir/'weights').glob('epoch*.pt'))
    assert len(files)==cfg['epochs'],(len(files),cfg['epochs'])
    final=save_dir/'weights'/f"epoch{cfg['epochs']-1}.pt"
    complete=dict(status='PASS',seed=args.seed,arm=args.arm,epochs=cfg['epochs'],checkpoints=len(files),primary_checkpoint=str(final),
        primary_sha256=sha(final),utc=datetime.now(timezone.utc).isoformat())
    (save_dir/'TRAINING_COMPLETE.json').write_text(json.dumps(complete,indent=2))
    print('TRAINING_COMPLETE',json.dumps(complete),flush=True)

if __name__=='__main__':main()
