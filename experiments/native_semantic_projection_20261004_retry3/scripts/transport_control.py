"""Desktop file transport only; all computation stays on authorized server."""
import sys,json,hashlib,base64,ast
from pathlib import Path
P=Path(r'C:\Dpan\codexproject\paper-disc-organized')
sys.path.insert(0,str(P/'experiments/prototype_guided_evidence_selection_20261003/scripts'))
import server_transport as t
root=P/'experiments/native_semantic_projection_20261004_retry3'
remote='/root/native_semantic_projection_20261004_retry3'
py='/root/miniconda3/bin/python'
def run(code,timeout=60):
    encoded=base64.b64encode(code.encode()).decode()
    return t.ssh(py+' -c '+repr("import base64;exec(base64.b64decode('"+encoded+"'))"),timeout)
mode=sys.argv[1]
if mode=='preflight':
    print(run("import pathlib,json,subprocess\nr=pathlib.Path('"+remote+"')\np=r/'PIPELINE_STATUS.json'\nprint(json.dumps({'study_exists':r.exists(),'pipeline_exists':p.exists(),'claim_exists':(r/'PIPELINE.claim').exists(),'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()}))\nfor p in pathlib.Path('/proc').iterdir():\n if p.name.isdigit():\n  try:\n   s=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')\n   if 'pipeline_' in s or 'train_' in s or 'evaluate_' in s: print(p.name,s[:250])\n  except OSError:pass"))
elif mode=='status':
    print(run("import json,pathlib,subprocess\nr=pathlib.Path('"+remote+"')\np=r/'PIPELINE_STATUS.json'\nprint(p.read_text() if p.exists() else 'NO_PIPELINE')\nprint(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))\nprint(subprocess.check_output(['ps','-eo','pid,etimes,args'],text=True))"))
elif mode=='upload':
    required = ['PROTOCOL.md','RUN_CONFIG.json','RUN_IDS.json','SPLIT.json','REUSED_CODE.json','study.json',
                'scripts/pipeline_semantic.py','scripts/evaluate_semantic.py','scripts/semantic_source.py','scripts/full_projection.py',
                'scripts/online_runtime.py','scripts/evaluation_metrics.py','scripts/collect_semantic.py','scripts/transport_control.py']
    absent = [name for name in required if not (root / name).is_file()]
    if absent: raise RuntimeError('Launch package incomplete: '+repr(absent))
    files = [root / name for name in required]
    for path in files:
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    reuse = json.loads((root/'REUSED_CODE.json').read_text(encoding='utf-8-sig'))
    for name, row in reuse.items():
        if hashlib.sha256((root/'scripts'/name).read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Reused runtime differs: '+name)
    ready={'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'frozen_before_launch':True}
    (root/'CODE_READY.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print(run("from pathlib import Path\nr=Path('"+remote+"')\nassert not (r/'PIPELINE_STATUS.json').exists()\n(r/'scripts').mkdir(parents=True,exist_ok=True)"))
    t.put([p for p in files if p.parent==root]+[root/'CODE_READY.json'],remote+'/',300)
    t.put([p for p in files if p.parent==root/'scripts'],remote+'/scripts/',300)
    print(run("import json,pathlib,hashlib\nr=pathlib.Path('"+remote+"')\nf=json.loads((r/'CODE_READY.json').read_text())['files']\nassert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in f.items())\nprint(json.dumps({'verified_files':len(f)}))"))
elif mode=='launch':
    print(run("import pathlib,subprocess,json,os\nr=pathlib.Path('"+remote+"')\nassert not (r/'PIPELINE_STATUS.json').exists() and not (r/'PIPELINE.claim').exists()\ngpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()\nassert not gpu, 'GPU busy: '+gpu\nwith (r/'pipeline.stdout.log').open('xb') as o, (r/'pipeline.stderr.log').open('xb') as e:\n p=subprocess.Popen(['"+py+"','-u',str(r/'scripts/pipeline_semantic.py'),'--root',str(r)],cwd=r,stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True)\nprint(json.dumps({'pid':p.pid,'root':str(r)}))"))
elif mode=='finalize':
    files=[root/'REPORT.md',root/'study.json']
    t.put(files,remote+'/',300)
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    code="import pathlib,json,hashlib,subprocess\nr=pathlib.Path('"+remote+"')\ne="+repr(expected)+"\nassert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in e.items())\ns=json.loads((r/'PIPELINE_STATUS.json').read_text())\nprint(json.dumps({'status':s['status'],'finished_at':s['finished_at'],'root_reports_verified':e,'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()}))"
    print(run(code))
elif mode=='brief':
    print(run("import pathlib,json\nr=pathlib.Path('"+remote+"')\ns=json.loads((r/'PIPELINE_STATUS.json').read_text())\nprint(json.dumps({k:s.get(k) for k in ('status','stage','pid','child_pid','error','finished_at')},ensure_ascii=False))\np=r/'runs'/s['current_run']\nfor f in ('PROGRESS.json','FAILURE.json','COMPLETE.json'):\n q=p/f\n if q.exists():print(f+'\\n'+q.read_text(errors='replace')[-5000:])"))
elif mode=='detail':
    print(run("import pathlib,json\nr=pathlib.Path('"+remote+"')\ns=json.loads((r/'PIPELINE_STATUS.json').read_text())\nprint(json.dumps({k:s.get(k) for k in ('status','stage','pid','child_pid','error','stage_results')},ensure_ascii=False))\np=r/'runs'/s['current_run']\nfor f in ('PROGRESS.json','FAILURE.json','COMPLETE.json','AUDIT.json','prepare.stderr.log','stderr.log','stdout.log'):\n q=p/f\n if q.exists(): print(f+'\\n'+q.read_text(errors='replace')[-6500:])\nfor f in ('pipeline.stderr.log','pipeline.stdout.log'):\n q=r/f\n if q.exists():print(f+'\\n'+q.read_text(errors='replace')[-4000:])"))
else:raise ValueError(mode)
