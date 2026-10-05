"""Sequential bounded six-head study. Each training and evaluation gets a Run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());root=a.root
    script=root/'scripts/component_seed_probe.py';ref=root/'runs/RUN_690c1580277a44679752ff727705f2e6'
    bank=root/'runs/RUN_e83cf453bd6e47feb025ef4259bc8499/training_bank.pt'
    data=Path('/root/autodl-tmp/datasets/coco2017');weights=root/'yolo26m-seg.pt'
    runner=root/'runner.py';status_path=root/'component_seed_queue_status.json'
    def status(**extra):
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(updated_at=datetime.now(timezone.utc).isoformat(),**extra)))
        tmp.replace(status_path)
    def run(run_id,action,extras):
        out=root/'runs'/run_id
        cmd=[sys.executable,str(runner),'--run-id',run_id,'--study',manifest['study_id'],'--output',str(out),'--cwd',str(root)]
        for snap in [script,root/'scripts/learn_refinement.py',root/'scripts/repair_refinement.py',root/'ABLATION_PROTOCOL.json']:
            cmd+=['--snapshot',str(snap)]
        cmd+=['--input',str(ref/'SPLIT.json'),'--input',str(a.manifest),'--expect',str(out/'COMPLETE.json'),
              '--',sys.executable,'-u',str(script),action,'--reference',str(ref),'--out',str(out),*extras]
        status(state='running',run_id=run_id,action=action)
        rc=subprocess.call(cmd)
        if rc:
            status(state='failed',run_id=run_id,return_code=rc);raise SystemExit(rc)
    run(manifest['smoke_run_id'],'smoke',['--data',str(data),'--weights',str(weights)])
    for entry in manifest['train']:
        run(entry['run_id'],'train',['--bank',str(bank),'--mode',entry['mode'],'--seed',str(entry['seed'])])
    run(manifest['evaluation_run_id'],'evaluate',['--data',str(data),'--weights',str(weights),'--manifest',str(a.manifest)])
    status(state='completed',evaluation_run_id=manifest['evaluation_run_id'])


if __name__=='__main__':main()
