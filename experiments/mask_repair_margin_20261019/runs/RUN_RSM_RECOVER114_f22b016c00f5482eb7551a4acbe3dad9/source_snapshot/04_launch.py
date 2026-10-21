"""One explicit detached Windows execution; no scheduler, timer or automatic retry."""
from pathlib import Path
import argparse,json,os,subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['smoke','full'],required=True);p.add_argument('--parent-run');a=p.parse_args()
    study=Path(__file__).resolve().parents[1]
    receipt=study/f'LAUNCH_{a.scope}.json'
    if receipt.exists():raise FileExistsError('existing launch receipt; inspect, do not duplicate')
    command=[sys.executable,'-B','-u',str(study/'scripts/run_audit.py'),'--scope',a.scope]
    if a.parent_run:command+=['--parent-run',a.parent_run]
    with (study/f'launcher_{a.scope}.log').open('x',encoding='utf8') as output:
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
           cwd=str(study.parents[1]),creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    record={'pid':proc.pid,'command':command,'scheduler':None,'automatic_retry':False}
    receipt.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8');print(json.dumps(record))

if __name__=='__main__':main()
