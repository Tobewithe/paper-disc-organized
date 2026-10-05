"""Persistent sequential queue: paired seeds 0/1/2, immutable launch inputs."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
JOBS=[dict(seed=seed,arm=arm) for seed in [0,1,2] for arm in ['baseline','ccl01']]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_status(status):
    status['updated_utc']=datetime.now(timezone.utc).isoformat()
    tmp=ROOT/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(ROOT/'queue_status.json')

def check_label_contents():
    for split in ['train2017','val2017']:
        report=json.loads((ROOT/'audits'/f'actual_loader_{split}.json').read_text())
        paths=sorted((ROOT/'official_yolo'/f'{split}.txt').read_text().splitlines())
        h=hashlib.sha256()
        for path in paths:
            iid=int(Path(path).stem)
            label=ROOT/'official_yolo/labels'/split/Path(path).with_suffix('.txt').name
            h.update(str(iid).encode()+b'\0'+(label.read_bytes() if label.exists() else b''))
        assert h.hexdigest()==report['source_label_content_sha256'],f'Label contents changed: {split}'

def check_pair(seed):
    dirs=[ROOT/'runs'/f'{arm}_s{seed}' for arm in ['baseline','ccl01']]
    a,b=[json.loads((d/'launch_receipt.json').read_text()) for d in dirs]
    assert a['parameter_sha256']==b['parameter_sha256'],'Initial parameters differ'
    assert (dirs[0]/'initial_batches.jsonl').read_text()==(dirs[1]/'initial_batches.jsonl').read_text(),'Paired seed batches differ'
    different={k for k in a['resolved_args'] if a['resolved_args'][k]!=b['resolved_args'][k]}
    assert different<={'name','save_dir'},different
    assert a['seed']==b['seed']==seed and a['ccl_weight']==0 and b['ccl_weight']==.1
    aepochs=[json.loads(x) for x in (dirs[0]/'epoch_integrity.jsonl').read_text().splitlines()]
    bepochs=[json.loads(x) for x in (dirs[1]/'epoch_integrity.jsonl').read_text().splitlines()]
    assert len(aepochs)==len(bepochs)==15
    assert [r['optimizer_steps'] for r in aepochs]==[r['optimizer_steps'] for r in bepochs]
    assert all(r['o2m']['pairs']>0 and r['o2o']['pairs']>0 for r in bepochs)
    result=dict(status='PASS',seed=seed,initial_weights_equal=True,initial_augmented_batches_equal=True,
        matched_epochs=15,optimizer_steps=aepochs[-1]['optimizer_steps'],only_config_differences=sorted(different))
    (ROOT/'audits'/f'PAIRED_SEED_{seed}_PASS.json').write_text(json.dumps(result,indent=2))

def main():
    lock=(ROOT/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=json.loads((ROOT/'audits/LAUNCH_GATE.json').read_text())
    assert gate['status']=='PASS'
    frozen=gate['frozen_inputs_sha256']
    status=dict(status='RUNNING',jobs=[dict(**job,status='PENDING') for job in JOBS],pid=os.getpid())
    write_status(status)
    try:
        for entry in status['jobs']:
            for relative,expected in frozen.items():
                assert sha(ROOT/relative)==expected,f'Frozen input changed: {relative}'
            check_label_contents()
            name=f"{entry['arm']}_s{entry['seed']}"
            directory=ROOT/'runs'/name
            if directory.exists():
                done=directory/'TRAINING_COMPLETE.json'
                if not done.exists():raise RuntimeError(f'Existing incomplete run requires inspection: {directory}')
                receipt=json.loads(done.read_text());assert receipt['status']=='PASS' and receipt['epochs']==15
                assert sha(receipt['primary_checkpoint'])==receipt['primary_sha256']
                entry['status']='COMPLETE_EXISTING';write_status(status);continue
            entry['status']='RUNNING';entry['started_utc']=datetime.now(timezone.utc).isoformat();status['active']=name
            log=ROOT/'logs'/f'{name}.log';entry['log']=str(log);write_status(status)
            env=os.environ.copy();env.update(PYTHONHASHSEED=str(entry['seed']),CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
            command=[sys.executable,str(ROOT/'train_pair.py'),'--seed',str(entry['seed']),'--arm',entry['arm']]
            print('START_JOB',name,flush=True)
            with log.open('w') as output:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
                entry['pid']=proc.pid;write_status(status);code=proc.wait()
            entry['exit_code']=code
            if code:raise RuntimeError(f'{name} exited {code}; inspect {log}')
            done=json.loads((directory/'TRAINING_COMPLETE.json').read_text())
            assert done['status']=='PASS' and done['epochs']==15 and done['checkpoints']==15
            entry['status']='COMPLETE';entry['finished_utc']=datetime.now(timezone.utc).isoformat()
            if entry['arm']=='ccl01':check_pair(entry['seed'])
            write_status(status);print('COMPLETE_JOB',name,flush=True)
        status.update(status='COMPLETE',active=None);write_status(status)
        print('ALL_SIX_TRAININGS_COMPLETE',flush=True)
    except Exception as error:
        entry['status']='FAILED';status.update(status='FAILED',error=str(error));write_status(status)
        traceback.print_exc();raise

if __name__=='__main__':main()
