"""One bounded smoke and frozen audit, no optimizer or repeated training."""
import argparse,hashlib,json,os,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    r=Path(args.root).resolve()
    if os.name=='nt' or str(r)!='/root/external_spatial_projection_20261004':raise RuntimeError('Authorized server only')
    for name,h in json.loads((r/'CODE_READY.json').read_text())['files'].items():
        if hashlib.sha256((r/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Code/config changed '+name)
    fd=os.open(r/'PIPELINE.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
    cfg=json.loads((r/'RUN_CONFIG.json').read_text());ids=json.loads((r/'RUN_IDS.json').read_text())
    if not (r/'ASSET_MANIFEST.json').exists():raise RuntimeError('Official SAM assets not ready')
    prep=r/'runs'/ids['runs']['prepare']
    if json.loads((prep/'run.json').read_text()).get('status')!='completed' or not json.loads((prep/'COMPLETE.json').read_text()).get('passed'):raise RuntimeError('Preparation Run incomplete')
    state=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),runs=ids['runs'],no_training=True)
    def save():
        t=r/'PIPELINE_STATUS.json.tmp';t.write_text(json.dumps(state,indent=2));t.replace(r/'PIPELINE_STATUS.json')
    try:
        save()
        for stage in ('smoke','evaluation'):
            out=r/'runs'/ids['runs'][stage]
            if (out/'run.json').exists():raise RuntimeError('Existing Run, no duplicate launch')
            cmd=[sys.executable,str(Path(cfg['source_root'])/'runner.py'),'--study',ids['study_id'],'--run-id',ids['runs'][stage],'--cwd',str(r),'--output',str(out)]
            for path in [r/'RUN_CONFIG.json',r/'SPLIT.json',Path(cfg['cache'])/'CACHE_IDENTITY.json',Path(cfg['prior_results']),Path(cfg['weights']),Path(cfg['sam_checkpoint']),r/'ASSET_MANIFEST.json']:cmd+=['--input',str(path)]
            for path in [r/'PROTOCOL.md',r/'RUN_CONFIG.json',r/'RUN_IDS.json',r/'REUSED_CODE.json',*sorted((r/'scripts').glob('*.py'))]:cmd+=['--snapshot',str(path)]
            cmd+=['--expect',str(out/'COMPLETE.json'),'--',sys.executable,'-u',str(r/'scripts/evaluate_external.py'),'--config',str(r/'RUN_CONFIG.json'),'--out',str(out)]
            if stage=='smoke':cmd+=['--smoke']
            state.update(stage=stage,current_run=ids['runs'][stage]);save()
            child=subprocess.Popen(cmd,cwd=r,env=dict(os.environ,PYTHONPATH=cfg['source_python'],OMP_NUM_THREADS='6',PYTHONUTF8='1'),stdin=subprocess.DEVNULL)
            state['child_pid']=child.pid;save();code=child.wait();state.pop('child_pid',None)
            if code:raise RuntimeError(f'{stage} failed {code}; retain Run')
            if not json.loads((out/'COMPLETE.json').read_text()).get('passed'):raise RuntimeError('Checks incomplete')
        state.update(status='completed',stage='finished',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as e:
        state.update(status='failed',error=repr(e),traceback=traceback.format_exc(),finished_at=datetime.now(timezone.utc).isoformat());save();raise

if __name__=='__main__':main()
