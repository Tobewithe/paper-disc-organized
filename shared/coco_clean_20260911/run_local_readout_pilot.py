"""Local GPU runner using verified remote Ultralytics sources, no pip changes."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'local_readout_runtime_20260912'
RUN=ROOT/'diagnostics/readout_input_local_v2_20260912'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def main():
    RUN.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((RUNTIME/'RECEIPT.json').read_text())
    for name,digest in receipt['archive_sha256'].items():
        archive=RUNTIME/name
        if sha(archive)!=digest:
            raise RuntimeError(f'Archive mismatch: {name}')
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                dest=(RUNTIME/member.name).resolve()
                if not dest.is_relative_to(RUNTIME.resolve()) or not member.isfile():
                    raise RuntimeError(f'Unexpected archive member: {member.name}')
            tar.extractall(RUNTIME)
    for name,digest in {**receipt['runtime_files'],**receipt['data_files']}.items():
        if sha(RUNTIME/name)!=digest:
            raise RuntimeError(f'Extracted file mismatch: {name}')
    data=RUNTIME/'data'
    (data/'annotations').mkdir(parents=True,exist_ok=True)
    (data/'images/train2017').mkdir(parents=True,exist_ok=True)
    annotation=ROOT.parents[1]/'datasets/coco/annotations/instances_train2017.json'
    if sha(annotation)!=receipt['annotation_sha256']:
        raise RuntimeError('Local/remote annotation mismatch')
    dst=data/'annotations/instances_train2017.json'
    if not dst.exists():
        os.link(annotation,dst)
    for image in (RUNTIME/'images/train2017').glob('*.jpg'):
        dst=data/'images/train2017'/image.name
        if not dst.exists():
            os.link(image,dst)
    config=json.loads((ROOT/'readout_input_pilot_protocol.json').read_text())
    config['version']='1.1-local'
    config['runtime']='Local conda pytorch, verified remote ultralytics8.4.143 via isolated PYTHONPATH; all baseline/arms rebuilt on local GPU. Installed ultralytics8.4.100 unchanged. Torch/CV differences logged, no pooling with old remote metrics.'
    config['execution_amendment']='User requests fast local experiment. Main cache has inline source/GT/decoder checks; separate repeated smoke omitted. No hyperparameter or outcome selection changed.'
    configpath=RUN/'local_protocol.json'
    write(configpath,config)
    env=os.environ.copy()
    settings=RUNTIME/'settings'
    settings.mkdir(exist_ok=True)
    env.update(PYTHONPATH=str(RUNTIME/'vendor'),PYTHONUNBUFFERED='1',PYTHONUTF8='1',
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',WANDB_MODE='disabled',
               YOLO_CONFIG_DIR=str(settings))
    commands=[
        [sys.executable,str(ROOT/'local_readout_runtime_witness.py'),'--out',str(RUN/'RUNTIME.json')],
        [sys.executable,str(ROOT/'cache_readout_input_pilot.py'),'--out',str(RUN/'cache'),
         '--data-root',str(data),'--protocol',str(configpath)],
        [sys.executable,str(ROOT/'train_readout_input_pilot.py'),'--cache',str(RUN/'cache'),'--out',str(RUN/'trained')],
        [sys.executable,str(ROOT/'eval_readout_input_pilot.py'),'--cache',str(RUN/'cache'),
         '--trained',str(RUN/'trained'),'--out',str(RUN/'evaluation')],
        [sys.executable,str(ROOT/'summarize_readout_input_pilot.py'),'--evaluation',str(RUN/'evaluation')],
    ]
    write(RUN/'LAUNCH.json',dict(status='RUNNING',pid=os.getpid(),commands=commands,
                               runtime_receipt_sha256=sha(RUNTIME/'RECEIPT.json')))
    start=time.monotonic()
    try:
        for i,command in enumerate(commands):
            write(RUN/'progress.json',dict(status='RUNNING',stage=i,command=command,seconds=time.monotonic()-start))
            print(json.dumps(dict(stage=i,command=command)),flush=True)
            subprocess.run(command,env=env,cwd=ROOT,check=True)
        write(RUN/'COMPLETE.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
    except Exception as exc:
        write(RUN/'FAILED.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-start))
        write(RUN/'progress.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    main()
