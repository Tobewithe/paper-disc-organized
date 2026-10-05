"""Independent bounded laptop pipeline, no automatic retries or other studies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    if (root/'PIPELINE_STATUS.json').exists(): raise RuntimeError('Pipeline already registered; no implicit retry')
    with (root/'PIPELINE_LOCK.json').open('x',encoding='utf-8') as f: json.dump(dict(pid=os.getpid(),started_at=time.time()),f)
    ids=json.loads((root/'RUN_IDS.json').read_text(encoding='utf-8-sig'));cfg=json.loads((root/'RUN_CONFIG.json').read_text(encoding='utf-8-sig'))
    state=dict(status='running',stage='preparing',pid=os.getpid(),runs=ids['runs'],started_at=time.time(),automatic_followup=False)
    os.environ['PYTHONPATH']=r'D:\coco_wire\py';os.environ['PYTHONUTF8']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    python=str(Path(sys.executable).with_name('python.exe'))
    def status(): save(root/'PIPELINE_STATUS.json',state)
    status()
    def run(stage,script,extra,timeout):
        rid=ids['runs'][stage];out=root/'runs'/rid
        if (out/'run.json').exists(): raise RuntimeError('Run already exists; preserve prior execution')
        state.update(stage=stage,current_run=rid);status()
        command=[python,r'D:\coco_wire\runner.py','--study',ids['study_id'],'--run-id',rid,'--cwd',str(root),'--output',str(out)]
        inputs=[root/'RUN_CONFIG.json',Path(cfg['cache'])/'INDEX.json',Path(cfg['cache'])/'COMPLETE.json',Path(cfg['operator_cache'])/'COMPLETE.json',Path(cfg['weights'])]
        if (root/'INDEX.json').exists():inputs += [root/'INDEX.json',root/'SPLIT.json']
        for path in inputs: command += ['--input',str(path)]
        for path in [root/'RUN_CONFIG.json',root/'PROTOCOL.md',root/'RUN_IDS.json',*sorted((root/'scripts').glob('*.py'))]:command += ['--snapshot',str(path)]
        command += ['--expect',str(out/'COMPLETE.json'),'--',python,'-B','-u',str(root/'scripts'/script),'--config',str(root/'RUN_CONFIG.json'),'--out',str(out),*extra]
        proc=subprocess.Popen(command,cwd=root,env=os.environ.copy(),stdout=sys.stdout,stderr=sys.stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        state['runner_pid']=proc.pid;status()
        try: code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            raise TimeoutError(f'{stage} exceeded its frozen resource budget; no automatic retry')
        if code: raise RuntimeError(f'{stage} exited {code}; see preserved Run logs')
        complete=json.loads((out/'COMPLETE.json').read_text(encoding='utf-8-sig'))
        if not (complete.get('completed') or complete.get('passed')): raise RuntimeError(f'{stage} lacks completion evidence')
    try:
        run('prepare','run_feedback.py',['--stage','prepare'],1800)
        run('smoke','run_feedback.py',['--stage','smoke'],1800)
        deadline=time.time()+cfg['max_training_hours']*3600;state['training_deadline']=deadline;status();paths={}
        for mode in cfg['training_order']:
            remaining=deadline-time.time()
            if remaining<=0: raise TimeoutError('Training resource budget exhausted')
            run(mode,'run_feedback.py',['--stage','train','--mode',mode,'--deadline',str(deadline)],remaining+120)
            paths[mode]=str(root/'runs'/ids['runs'][mode]/'final.pt')
        save(root/'FINAL_CHECKPOINTS.json',paths)
        run('evaluation','evaluate_feedback.py',['--checkpoints',str(root/'FINAL_CHECKPOINTS.json')],5400)
        import shutil
        for name in ('REPORT.md','SUMMARY.json'):
            shutil.copy2(root/'runs'/ids['runs']['evaluation']/name,root/name)
        state.update(status='completed',stage='finished',finished_at=time.time());status()
    except Exception as exc:
        state.update(status='failed',error=repr(exc),finished_at=time.time(),automatic_retry=False);status();raise


if __name__=='__main__':
    root=Path(sys.argv[sys.argv.index('--root')+1]);root.mkdir(parents=True,exist_ok=True)
    with (root/'pipeline.log').open('a',encoding='utf-8',buffering=1) as log:
        sys.stdout=log;sys.stderr=log
        main()
