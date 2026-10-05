"""One authorized sequence; record each full-cache/train/evaluation Run separately."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


IDS={
 'cache':'RUN_a64cfc08c15642f98b6cb92e26d47490',
 'N':'RUN_257e10f7c78c4d19a8d01d888652ea6d',
 'P':'RUN_2a2828fc4495480d8a780f337d94a3c6',
 'R':'RUN_62cb5f8c90364db185505ca8e0ef5d12',
 'evaluation':'RUN_976ee94db0ed49d894567e0fc0151224'
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);cfg=json.loads((root/'RUN_CONFIG.json').read_text(encoding='utf-8-sig'))
    state=dict(status='running',stage='cache',runs=IDS,started_at=time.time())
    def status():
        (root/'PIPELINE_STATUS.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    status()
    os.environ['PYTHONPATH']=r'D:\coco_wire\py'
    snapshots=[root/'RUN_CONFIG.json',root/'PROTOCOL.md',*sorted((root/'scripts').glob('*.py'))]
    def run(stage,script,extra):
        rid=IDS[stage];out=root/'runs'/rid
        if (out/'COMPLETE.json').exists():return
        # Failed runs are preserved. A retry must have a new ID selected by the
        # operator; no infinite automatic retry or architecture search.
        if (out/'run.json').exists():raise RuntimeError(f'{stage} existing unfinished Run; inspect before retry')
        state.update(stage=stage,current_run=rid);status()
        cmd=[sys.executable,r'D:\coco_wire\runner.py','--study','STUDY_957126780eb14b82a0a6112902eb18ce',
            '--run-id',rid,'--cwd',str(root),'--output',str(out),
            '--input',str(root/'RUN_CONFIG.json'),'--input',str(root/'SPLIT.json')]
        for source in snapshots:cmd+=['--snapshot',str(source)]
        cmd+=['--expect',str(out/'COMPLETE.json'),'--',sys.executable,'-u',str(root/'scripts'/script),
            '--config',str(root/'RUN_CONFIG.json'),'--out',str(out),*extra]
        subprocess.run(cmd,check=True,cwd=root,env=os.environ.copy())
    try:
        smoke=json.loads((root/'runs/RUN_4d96ed18566543b8957dc70c4c6c16d0/COMPLETE.json').read_text())
        assert smoke['passed'],'New model smoke must pass before formal computation'
        decode=json.loads((root/'runs/RUN_7a9cd28c9a8d46b4a4e9d75822334902/COMPLETE.json').read_text())
        assert decode['completed'] and decode['smoke'],'New evaluation smoke must pass'
        run('cache','prepare_cache.py',[])
        deadline=time.time()+3600*cfg['max_training_hours']
        state['shared_training_deadline']=deadline;status()
        checkpoints={}
        for mode in cfg['arms']:
            run(mode,'train_native.py',['--mode',mode,'--deadline',str(deadline)])
            checkpoints[mode]=str(root/'runs'/IDS[mode]/'final.pt')
        paths=root/'FINAL_CHECKPOINTS.json';paths.write_text(json.dumps(checkpoints,indent=2),encoding='utf-8')
        run('evaluation','evaluate_trained.py',['--checkpoints',str(paths)])
        state.update(status='completed',stage='finished',finished_at=time.time(),automatic_followup=False);status()
    except Exception as exc:
        state.update(status='failed',error=str(exc),finished_at=time.time(),automatic_retry=False);status()
        raise


if __name__=='__main__':main()
