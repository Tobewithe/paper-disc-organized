import argparse, json, os, subprocess, sys, time, uuid
from pathlib import Path


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=['smoke','main'],required=True);a=ap.parse_args()
    root=a.root;ids=json.loads((root/'deployment_ids.json').read_text());cfg=json.loads((root/'PROTOCOL.json').read_text())
    stages=[('smoke_'+m,'train_native.py',['--mode',m,'--epochs','1','--smoke']) for m in ('A','B','C')] if a.phase=='smoke' else (
        [('train_'+m,'train_native.py',['--mode',m,'--epochs',str(cfg['epochs'])]) for m in ('A','B','C')]+
        [('eval_'+m,'evaluate.py',['--mode',m]) for m in ('R','A','B','C')]+[('analyze','analyze.py',[])])
    for key,_,_ in stages:
        if key not in ids['run_ids']:ids['run_ids'][key]='RUN_'+uuid.uuid4().hex
    write(root/'deployment_ids.json',ids)
    for key,script,extra in stages:
        rid=ids['run_ids'][key];out=root/'runs'/rid
        if out.exists():
            rec=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {}
            if rec.get('status')=='completed' and (out/'COMPLETE.json').exists():continue
            raise RuntimeError(f'Incomplete existing run {key}; preserve it and allocate an explicit retry ID')
        write(root/f'{a.phase}_status.json',{'status':'running','stage':key,'run_id':rid,'queue_pid':os.getpid(),'started_at':time.time()})
        cmd=[sys.executable,str(root/'scripts/runner.py'),'--study',ids['study_id'],'--run-id',rid,'--output',str(out),'--cwd',str(root),
             '--input',cfg['weights'],'--input',str(root/'data/SPLIT.json'),'--snapshot',str(root/'PROTOCOL.json'),
             '--snapshot',str(root/'scripts'/script),'--expect',str(out/'COMPLETE.json')]
        if script=='train_native.py':cmd+=['--snapshot',str(root/'scripts/coefficient_objective.py')]
        cmd+=['--',sys.executable,'-u',str(root/'scripts'/script),'--root',str(root)]
        if script!='analyze.py':cmd+=['--weights',cfg['weights']]
        cmd+=extra
        code=subprocess.call(cmd)
        if code:
            write(root/f'{a.phase}_status.json',{'status':'failed','stage':key,'run_id':rid,'exit_code':code});raise SystemExit(code)
    write(root/f'{a.phase}_status.json',{'status':'completed','stages':[s[0] for s in stages],'finished_at':time.time()})


if __name__=='__main__':main()
