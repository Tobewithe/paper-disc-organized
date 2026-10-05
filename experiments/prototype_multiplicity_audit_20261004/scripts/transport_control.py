"""Hidden server-only transmission; no local model execution."""
import ast,base64,hashlib,json,sys
from pathlib import Path
P=Path(r'C:\Dpan\codexproject\paper-disc-organized')
sys.path.insert(0,str(P/'experiments/prototype_guided_evidence_selection_20261003/scripts'))
import server_transport as t
root=P/'experiments/prototype_multiplicity_audit_20261004'
remote='/root/prototype_multiplicity_audit_20261004'
py='/root/miniconda3/bin/python'
def run(code,timeout=60):
    encoded=base64.b64encode(code.encode()).decode()
    return t.ssh(py+' -c '+repr("import base64;exec(base64.b64decode('"+encoded+"'))"),timeout)
mode=sys.argv[1]
if mode=='preflight':
    print(run("import pathlib,json,subprocess\nr=pathlib.Path('"+remote+"')\nprint(json.dumps({'study_exists':r.exists(),'pipeline_exists':(r/'PIPELINE_STATUS.json').exists(),'claim_exists':(r/'PIPELINE.claim').exists(),'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()}))\nfor p in pathlib.Path('/proc').iterdir():\n if p.name.isdigit():\n  try:\n   s=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')\n   if any(v in s for v in ('pipeline_','train_','evaluate_','audit_multiplicity.py')): print(p.name,s[:250])\n  except OSError:pass"))
elif mode=='upload':
    names=['PROTOCOL.md','RUN_CONFIG.json','RUN_IDS.json','study.json','SPLIT.json','REUSED_CODE.json']
    required=['audit_multiplicity.py','assignment_replay.py','pipeline_audit.py','collect_audit.py','transport_control.py','native_proto_model.py','online_runtime.py','runtime_utils.py','box_guided_head.py','evaluation_metrics.py','frozen_evaluation.py']
    files=[root/n for n in names]+[root/'scripts'/n for n in required]
    assert all(p.is_file() for p in files),'Missing launch asset'
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),str(p))
    for n,row in json.loads((root/'REUSED_CODE.json').read_text(encoding='utf-8-sig')).items():
        assert hashlib.sha256((root/'scripts'/n).read_bytes()).hexdigest()==row['sha256'],'Modified frozen dependency '+n
    ready={'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'frozen_before_launch':True}
    (root/'CODE_READY.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print(run("from pathlib import Path\nr=Path('"+remote+"')\nassert not (r/'PIPELINE_STATUS.json').exists() and not (r/'PIPELINE.claim').exists()\n(r/'scripts').mkdir(parents=True,exist_ok=True)"))
    t.put([p for p in files if p.parent==root]+[root/'CODE_READY.json'],remote+'/',300)
    t.put([p for p in files if p.parent==root/'scripts'],remote+'/scripts/',300)
    print(run("import pathlib,json,hashlib\nr=pathlib.Path('"+remote+"')\nf=json.loads((r/'CODE_READY.json').read_text())['files']\nassert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in f.items())\nprint(json.dumps({'verified_files':len(f)}))"))
elif mode=='launch':
    print(run("import pathlib,json,subprocess\nr=pathlib.Path('"+remote+"')\nassert not (r/'PIPELINE_STATUS.json').exists() and not (r/'PIPELINE.claim').exists()\ngpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()\nassert not gpu,'GPU busy: '+gpu\nwith (r/'pipeline.stdout.log').open('xb') as o,(r/'pipeline.stderr.log').open('xb') as e:\n p=subprocess.Popen(['"+py+"','-u',str(r/'scripts/pipeline_audit.py'),'--root',str(r)],cwd=r,stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True)\nprint(json.dumps({'pid':p.pid,'root':str(r)}))"))
elif mode in ('brief','detail'):
    tail=3000 if mode=='brief' else 7000
    print(run("import pathlib,json,subprocess\nr=pathlib.Path('"+remote+"')\ns=json.loads((r/'PIPELINE_STATUS.json').read_text())\nprint(json.dumps({k:s.get(k) for k in ('status','stage','pid','child_pid','error','finished_at')}))\np=r/'runs'/s['current_run']\nfor f in ('PROGRESS.json','FAILURE.json','COMPLETE.json','stderr.log'):\n q=p/f\n if q.exists():print(f+'\\n'+q.read_text(errors='replace')[-"+str(tail)+":])\nfor key in ('pid','child_pid'):\n pid=s.get(key)\n if pid:\n  proc=pathlib.Path('/proc')/str(pid)\n  print(key,str(pid),(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace') if proc.exists() else 'missing')\nprint('GPU',subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip())"))
elif mode=='finalize':
    files=[root/'REPORT.md',root/'study.json']
    t.put(files,remote+'/',300)
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    print(run("import pathlib,json,hashlib,subprocess\nr=pathlib.Path('"+remote+"')\ne="+repr(expected)+"\nassert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in e.items())\ns=json.loads((r/'PIPELINE_STATUS.json').read_text())\nprint(json.dumps({'status':s['status'],'finished_at':s.get('finished_at'),'verified':e,'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()}))"))
else:raise ValueError(mode)
