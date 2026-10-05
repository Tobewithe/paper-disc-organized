"""Bounded native-prototype screen; runs only on the authorized Linux root."""
import argparse
from datetime import datetime, timezone
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time,traceback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root).resolve()
    if os.name=='nt' or str(root)!='/root/native_proto_coadaptation_screen_20261003':
        raise RuntimeError('Authorized Linux GPU server root only')
    ready=json.loads((root/'CODE_READY.json').read_text())
    for name,digest in ready['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Prepared source changed: '+name)
    descriptor=os.open(root/'PIPELINE.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    cfg=json.loads((root/'RUN_CONFIG.json').read_text());ids=json.loads((root/'RUN_IDS.json').read_text())
    state=dict(status='running',stage='starting',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        runs=ids['runs'],automatic_retry=False,scope='native prototype route screen; no AP or independent confirmation')
    def save():
        temp=root/'PIPELINE_STATUS.json.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(root/'PIPELINE_STATUS.json')
    try:
        save();training_deadline=None
        for stage in ('smoke_P','smoke_PC','P','PC','evaluation'):
            out=root/'runs'/ids['runs'][stage]
            if (out/'run.json').exists():raise RuntimeError('Existing Run; refuse duplicate')
            cmd=[sys.executable,str(Path(cfg['source_root'])/'runner.py'),'--study',ids['study_id'],
                '--run-id',ids['runs'][stage],'--cwd',str(root),'--output',str(out)]
            for path in (root/'RUN_CONFIG.json',root/'SPLIT.json',Path(cfg['weights']),Path(cfg['cache'])/'CACHE_IDENTITY.json',
                         Path(cfg['original_screen_results']),Path(cfg['original_N_checkpoint'])):
                cmd+=['--input',str(path)]
            for path in (root/'PROTOCOL.md',root/'RUN_CONFIG.json',root/'RUN_IDS.json',root/'REUSED_CODE.json',*sorted((root/'scripts').glob('*.py'))):
                cmd+=['--snapshot',str(path)]
            cmd+=['--expect',str(out/'COMPLETE.json'),'--',sys.executable,'-u']
            if stage=='evaluation':
                cmd += [str(root/'scripts/evaluate_proto.py'),'--config',str(root/'RUN_CONFIG.json'),'--out',str(out)]
            else:
                is_smoke=stage.startswith('smoke_');mode=stage.split('_')[-1]
                if not is_smoke and training_deadline is None:
                    training_deadline=time.time()+2400;state['training_deadline_unix']=training_deadline
                deadline=time.time()+600 if is_smoke else training_deadline
                cmd += [str(root/'scripts/train_proto.py'),'--config',str(root/'RUN_CONFIG.json'),
                        '--mode',mode,'--out',str(out),'--deadline',str(deadline)]
                if is_smoke:cmd+=['--smoke']
            state.update(stage=stage,current_run=ids['runs'][stage]);save()
            env=dict(os.environ,PYTHONPATH=cfg['source_python'],PYTHONUTF8='1',OMP_NUM_THREADS=str(cfg['cpu_threads']))
            child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL)
            state['child_pid']=child.pid;save();code=child.wait();state.pop('child_pid',None)
            if code:raise RuntimeError(f'{stage} exited {code}; retain Run, no automatic retry')
            complete=json.loads((out/'COMPLETE.json').read_text())
            if not (complete.get('completed') or complete.get('passed')):raise RuntimeError(stage+' checks incomplete')
        state.update(status='completed',stage='finished',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc(),finished_at=datetime.now(timezone.utc).isoformat());save();raise


if __name__=='__main__':main()
