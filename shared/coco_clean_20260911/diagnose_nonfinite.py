"""Isolated replay from finite epoch-8 state; never changes formal run artifacts."""
import os
os.environ['CCL_WEIGHT']='0'
os.environ['COCO_SMOKE']='0'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import json
import random
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
from ultralytics import YOLO
from train_pair import GuardedSegmentationTrainer, ROOT, cpu_tree, install_ccl

parser=argparse.ArgumentParser()
parser.add_argument('--batches',type=int,default=2340)
parser.add_argument('--name',default='replay_epoch9')
args=parser.parse_args()
OUT=ROOT/'diagnostics/nonfinite_20260911'/args.name
if OUT.exists():raise RuntimeError(f'Replay output exists: {OUT}')
OUT.mkdir(parents=True)
RECOVERY=ROOT/'runs/baseline_s0/weights/recovery_latest.pt'

class DiagnosticTrainer(GuardedSegmentationTrainer):
    def preprocess_batch(self,batch):
        out=super().preprocess_batch(batch)
        self._diagnostic_batch=out
        self._diagnostic_batch_count=getattr(self,'_diagnostic_batch_count',0)+1
        return out

    def optimizer_step(self):
        try:
            return super().optimizer_step()
        except RuntimeError as error:
            if 'non-finite' not in str(error):raise
            bad=[]; norm64=[]
            for name,p in self.model.named_parameters():
                if p.grad is None:continue
                finite=torch.isfinite(p.grad)
                if not finite.all():
                    bad.append(dict(name=name,shape=list(p.shape),nan=int(torch.isnan(p.grad).sum()),inf=int(torch.isinf(p.grad).sum())))
                norm64.append(p.grad.detach().double().norm().cpu())
            record=dict(status='NONFINITE_REPRODUCED',epoch=self.epoch+1,batch=self._diagnostic_batch_count,
                        loss=float(self.loss),bad_parameters=bad,norm64=float(torch.stack(norm64).norm()),
                        image_files=self._diagnostic_batch.get('im_file'),error=str(error))
            (OUT/'failure.json').write_text(json.dumps(record,indent=2))
            torch.save(dict(model=cpu_tree(self.model.state_dict()),batch=cpu_tree(self._diagnostic_batch),
                            criterion_updates=self.model.criterion.updates,
                            loss_items=cpu_tree(self.loss_items)), OUT/'failure_batch.pt')
            print('NONFINITE_DIAGNOSTIC',json.dumps(record),flush=True)
            raise

    def save_model(self):
        return None

def restore(trainer):
    ck=torch.load(RECOVERY,map_location='cpu',weights_only=False)
    trainer.model.load_state_dict(ck['model'],strict=True)
    trainer.ema.ema.load_state_dict(ck['ema'],strict=True)
    trainer.ema.updates=ck['ema_updates']
    trainer.optimizer.load_state_dict(ck['optimizer'])
    trainer.scaler.load_state_dict(ck['scaler'])
    trainer.scheduler.load_state_dict(ck['scheduler'])
    trainer.start_epoch=ck['epoch']+1
    trainer.model.criterion=trainer.model.init_criterion()
    for _ in range(trainer.start_epoch):trainer.model.criterion.update()
    trainer._optimizer_steps=ck['optimizer_steps']
    random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
    torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    print('DIAGNOSTIC_RESTORE',json.dumps(dict(start_epoch=trainer.start_epoch+1,optimizer_steps=trainer._optimizer_steps,
        o2m=trainer.model.criterion.o2m,o2o=trainer.model.criterion.o2o,
        note='Isolated diagnostic only. Worker augmentation streams are newly created, not original epoch9 replay.')),flush=True)

def batch_end(trainer):
    if not torch.isfinite(trainer.loss):raise FloatingPointError('Nonfinite diagnostic loss')
    if trainer._diagnostic_batch_count>=args.batches:
        (OUT/'bounded_replay.json').write_text(json.dumps(dict(status='NO_FAILURE_IN_BOUNDED_REPLAY',batches=args.batches)))
        print('DIAGNOSTIC_BOUND_REACHED',args.batches,flush=True)
        raise SystemExit(0)

torch.set_num_threads(4)
install_ccl()
cfg=json.loads((ROOT/'train_config.json').read_text())
cfg.update(seed=0,project=str(OUT),name='trainer',val=False,save=False,plots=False)
model=YOLO(cfg.pop('model'))
model.add_callback('on_train_start',restore)
model.add_callback('on_train_batch_end',batch_end)
model.train(trainer=DiagnosticTrainer,**cfg)
