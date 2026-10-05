"""Run the bounded preservation experiment with independent immutable runs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();r=a.root;m=json.loads(a.manifest.read_text())
    ref=r/'runs/RUN_690c1580277a44679752ff727705f2e6';bank=r/'runs/RUN_e83cf453bd6e47feb025ef4259bc8499/training_bank.pt'
    data=Path('/root/autodl-tmp/datasets/coco2017');status=r/'preservation_queue_status.json'
    def save(**kw):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(updated_at=datetime.now(timezone.utc).isoformat(),**kw)));tmp.replace(status)
    def execute(rid,script,args):
        out=r/'runs'/rid
        cmd=[sys.executable,str(r/'runner.py'),'--run-id',rid,'--study',m['study_id'],'--output',str(out),'--cwd',str(r)]
        for name in ('component_seed_probe.py','learn_refinement.py','repair_refinement.py',script):
            cmd+=['--snapshot',str(r/'scripts'/name)]
        cmd+=['--snapshot',str(r/'PRESERVATION_PROTOCOL.json'),'--input',str(ref/'SPLIT.json'),'--input',str(a.manifest),
              '--expect',str(out/'COMPLETE.json'),'--',sys.executable,'-u',str(r/'scripts'/script),*args,'--out',str(out)]
        save(state='running',run_id=rid,script=script);rc=subprocess.call(cmd)
        if rc:save(state='failed',run_id=rid,return_code=rc);raise SystemExit(rc)
    execute(m['loss_check_run_id'],'preservation_loss_check.py',[])
    for entry in m['train']:
        execute(entry['run_id'],'component_seed_probe.py',['train','--reference',str(ref),'--bank',str(bank),
            '--mode',entry['mode'],'--seed',str(entry['seed']),'--objective',entry['objective']])
    execute(m['evaluation_run_id'],'component_seed_probe.py',['evaluate','--reference',str(ref),'--data',str(data),
        '--weights',str(r/'yolo26m-seg.pt'),'--manifest',str(a.manifest)])
    execute(m['timing_run_id'],'time_refinement.py',['--root',str(r),'--data',str(data),'--weights',str(r/'yolo26m-seg.pt')])
    save(state='completed',evaluation_run_id=m['evaluation_run_id'],timing_run_id=m['timing_run_id'])


if __name__=='__main__':main()
