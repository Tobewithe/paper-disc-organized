"""One bounded laptop-only sequence; no automatic method search or retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);a=parser.parse_args()
    root=Path(a.root);cfg=json.loads((root/'RUN_CONFIG.json').read_text(encoding='utf-8-sig'))
    ids=json.loads((root/'RUN_IDS.json').read_text(encoding='utf-8-sig'))
    state=dict(status='running',stage='preflight',runs=ids['runs'],started_at=time.time(),automatic_followup=False)
    def save():
        tmp=root/'PIPELINE_STATUS.json.tmp';tmp.write_text(json.dumps(state,indent=2),encoding='utf-8')
        tmp.replace(root/'PIPELINE_STATUS.json')
    save()
    # pythonw keeps the orchestrator independent of an interactive console;
    # child runner/worker use python.exe for proper captured stdout/stderr.
    python=str(Path(sys.executable).with_name('python.exe'))
    os.environ['PYTHONPATH']=r'D:\coco_wire\py';os.environ['PYTHONUTF8']='1'
    snapshots=[root/'RUN_CONFIG.json',root/'PROTOCOL.md',root/'RUN_IDS.json',*sorted((root/'scripts').glob('*.py'))]
    def run(stage,script,extra):
        rid=ids['runs'][stage];out=root/'runs'/rid
        if (out/'COMPLETE.json').exists():
            record=json.loads((out/'run.json').read_text(encoding='utf-8-sig'))
            if record.get('status')!='completed':raise RuntimeError(f'{stage}: completion artifact without successful runner')
            return
        if (out/'run.json').exists():raise RuntimeError(f'{stage}: unfinished run; inspect before authorized retry')
        state.update(stage=stage,current_run=rid);save()
        cmd=[python,r'D:\coco_wire\runner.py','--study',ids['study_id'],'--run-id',rid,
             '--cwd',str(root),'--output',str(out),'--input',str(root/'RUN_CONFIG.json'),
             '--input',str(root/'SPLIT.json'),'--input',str(Path(cfg['cache'])/'INDEX.json'),
             '--input',cfg['weights']]
        for source in snapshots:cmd+=['--snapshot',str(source)]
        cmd+=['--expect',str(out/'COMPLETE.json'),'--',python,'-u',str(root/'scripts'/script),
              '--config',str(root/'RUN_CONFIG.json'),'--out',str(out),*extra]
        subprocess.run(cmd,check=True,cwd=root,env=os.environ.copy())
    try:
        assert (Path(cfg['cache'])/'COMPLETE.json').is_file(),'Prior cache must remain complete'
        run('operator_smoke','prepare_operators.py',['--smoke'])
        run('model_smoke','train_evidence.py',['--smoke'])
        smoke=json.loads((root/'runs'/ids['runs']['model_smoke']/'COMPLETE.json').read_text(encoding='utf-8-sig'))
        assert smoke.get('passed') or smoke.get('completed'),'New model smoke did not pass'
        empty=root/'SMOKE_CHECKPOINTS.json';empty.write_text('{}',encoding='utf-8')
        run('decode_smoke','evaluate_evidence.py',['--smoke','--checkpoints',str(empty)])
        run('operators','prepare_operators.py',[])
        deadline=time.time()+3600*cfg['max_training_hours'];state['shared_training_deadline']=deadline;save()
        checkpoints={}
        for mode in cfg['arms']:
            run(mode,'train_evidence.py',['--mode',mode,'--deadline',str(deadline)])
            checkpoints[mode]=str(root/'runs'/ids['runs'][mode]/'final.pt')
        path=root/'FINAL_CHECKPOINTS.json';path.write_text(json.dumps(checkpoints,indent=2),encoding='utf-8')
        run('evaluation','evaluate_evidence.py',['--checkpoints',str(path)])
        state.update(status='completed',stage='finished',finished_at=time.time());save()
    except Exception as exc:
        state.update(status='failed',error=str(exc),finished_at=time.time(),automatic_retry=False);save()
        raise


if __name__=='__main__':main()
