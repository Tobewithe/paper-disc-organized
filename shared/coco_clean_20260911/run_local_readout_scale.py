"""Predeclared larger-data check after the32-image overfit pilot, local GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from readout_input_probe import stable_seed,write_json,sha

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'local_readout_runtime_20260912'
RUN=ROOT/'diagnostics/readout_input_scale1200_20260912'


def main():
    RUN.mkdir(exist_ok=False)
    data=RUNTIME/'data'
    annotation=data/'annotations/instances_train2017.json'
    obj=json.loads(annotation.read_text(encoding='utf-8'))
    old=json.loads((ROOT/'local_readout_selection_20260912.json').read_text())
    excluded=set(old['fit']+old['transfer'])
    ranked=sorted([im['id'] for im in obj['images'] if im['id'] not in excluded],
                  key=lambda iid:stable_seed('readout-scale:20260912',iid))
    selection=dict(fit=ranked[:1200],transfer=ranked[1200:1500],
                   excluded_explored_pilot96=sorted(excluded),annotation_sha256=sha(annotation),
                   selection='all train2017 raw image IDs excluding prior96; no ICI or model filtering')
    write_json(RUN/'selection.json',selection)
    del obj
    config=json.loads((ROOT/'diagnostics/readout_input_local_v2_20260912/local_protocol.json').read_text())
    config.update(experiment='S024_READOUT_SCALE',version='2.0-local-predeclared',train_images=1200,
                  transfer_images=300,epochs=15,run_convex=False,eval_splits=['transfer'])
    config['scope']='1200 fit +300 disjoint heldout train2017 images excluding all96 pilot images; all raw COCO targets for added official task evaluation. This is a frozen-readout study; train2017 seen by pretrained asset, no independent COCO val claim.'
    config['checkpoints']='Every epoch saved, final15 fixed, no heldout model selection. Same initialization/order/2048native640 sampled BCE across arms/seeds.'
    config['linear_extra']='Off in scaled run; already measured in S023 small-fit probe.'
    config['evaluation']='Heldout fixed-attribution raw-COCO IoU plus full-prediction officialCOCOeval AP/R75 and low0/middle(0,.5]/high>.5 strata. No outcome filtering; scores and boxes fixed, matching over all ordinaryGT.'
    config['scaling_rationale']='Preplanned larger cohort separates32-image overfitting from input contribution. PureBCE held constant from pilot; no claim to exceed prior BCE+Dice baseline or spatial novelty solely from this experiment.'
    write_json(RUN/'protocol.json',config)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(RUNTIME/'vendor'),PYTHONUNBUFFERED='1',PYTHONUTF8='1',OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',WANDB_MODE='disabled',
               YOLO_CONFIG_DIR=str(RUNTIME/'settings'))
    commands=[
       [sys.executable,str(ROOT/'cache_readout_input_pilot.py'),'--out',str(RUN/'cache'),'--data-root',str(data),
        '--protocol',str(RUN/'protocol.json'),'--selection',str(RUN/'selection.json'),
        '--train-zip','C:/Dpan/document/model_datasets/datasets/coco/downloads/train2017.zip'],
       [sys.executable,str(ROOT/'train_readout_input_pilot.py'),'--cache',str(RUN/'cache'),'--out',str(RUN/'trained')],
       [sys.executable,str(ROOT/'eval_readout_input_pilot.py'),'--cache',str(RUN/'cache'),'--trained',str(RUN/'trained'),'--out',str(RUN/'evaluation')],
       [sys.executable,str(ROOT/'summarize_readout_input_pilot.py'),'--evaluation',str(RUN/'evaluation')],
       [sys.executable,str(ROOT/'eval_readout_input_task.py'),'--cache',str(RUN/'cache'),'--evaluation',str(RUN/'evaluation'),
        '--out',str(RUN/'task')],
    ]
    start=time.monotonic()
    write_json(RUN/'LAUNCH.json',dict(status='RUNNING',pid=os.getpid(),commands=commands))
    try:
        for i,cmd in enumerate(commands):
            write_json(RUN/'progress.json',dict(status='RUNNING',stage=i,command=cmd,seconds=time.monotonic()-start))
            print(json.dumps(dict(stage=i,command=cmd)),flush=True)
            subprocess.run(cmd,env=env,cwd=ROOT,check=True)
        write_json(RUN/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
    except Exception as exc:
        write_json(RUN/'FAILED.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-start))
        write_json(RUN/'progress.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    main()
