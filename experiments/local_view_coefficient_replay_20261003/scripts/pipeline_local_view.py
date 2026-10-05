"""One bounded frozen local-view replay; original weights never updated."""
import argparse
from datetime import datetime,timezone
import json,os
from pathlib import Path
import subprocess,sys,traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve()
    if os.name=='nt' or str(root)!='/root/local_view_coefficient_replay_20261003':
        raise RuntimeError('Authorized Linux server root only')
    fd=os.open(root/'PIPELINE.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    cfg=json.loads((root/'RUN_CONFIG.json').read_text());ident=json.loads((root/'RUN_IDS.json').read_text())
    if not (root/'CODE_READY.json').is_file():raise RuntimeError('Code readiness not published')
    state=dict(status='running',stage='starting',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),runs=ident['runs'],scope='zero-training exploratory replay',automatic_retry=False)
    def save():
        t=root/'PIPELINE_STATUS.json.tmp';t.write_text(json.dumps(state,indent=2));t.replace(root/'PIPELINE_STATUS.json')
    try:
        save()
        for stage in ('smoke','evaluation'):
            out=root/'runs'/ident['runs'][stage]
            if (out/'run.json').exists():raise RuntimeError('Refusing duplicate Run')
            cmd=[sys.executable,str(Path(cfg['source_root'])/'runner.py'),'--study',ident['study_id'],'--run-id',ident['runs'][stage],'--cwd',str(root),'--output',str(out)]
            for f in (root/'RUN_CONFIG.json',root/'SPLIT.json',Path(cfg['weights']),Path(cfg['cache'])/'CACHE_IDENTITY.json',Path(cfg['original_screen_results'])):
                cmd+=['--input',str(f)]
            for f in (root/'PROTOCOL.md',root/'RUN_CONFIG.json',root/'RUN_IDS.json',root/'REUSED_CODE.json',*sorted((root/'scripts').glob('*.py'))):
                cmd+=['--snapshot',str(f)]
            cmd+=['--expect',str(out/'COMPLETE.json'),'--',sys.executable,'-u',str(root/'scripts/evaluate_local_view.py'),'--config',str(root/'RUN_CONFIG.json'),'--out',str(out)]
            if stage=='smoke':cmd+=['--smoke']
            state.update(stage=stage,current_run=ident['runs'][stage]);save()
            env=dict(os.environ,PYTHONPATH=cfg['source_python'],PYTHONUTF8='1',OMP_NUM_THREADS=str(cfg['cpu_threads']))
            child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL)
            state['child_pid']=child.pid;save();code=child.wait();state.pop('child_pid',None)
            if code:raise RuntimeError(f'{stage} exited {code}; no automatic retry')
            complete=json.loads((out/'COMPLETE.json').read_text())
            if not (complete.get('passed') or complete.get('completed')):raise RuntimeError('Required checks not passed')
        state.update(status='completed',stage='finished',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc(),finished_at=datetime.now(timezone.utc).isoformat());save();raise
if __name__=='__main__':main()
