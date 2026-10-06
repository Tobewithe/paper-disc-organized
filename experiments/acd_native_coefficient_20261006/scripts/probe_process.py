import json,psutil,time
from pathlib import Path
root=Path('D:/coco_wire/experiments/acd_native_coefficient_20261006/runs')
for p in root.glob('*/*.json'):
 if p.name not in ['PROCESS.json','LAUNCH.json']: continue
 r=json.loads(p.read_text()); pid=r.get('pid');print(p.parent.name,p.name,pid,'alive',psutil.pid_exists(pid))
 if psutil.pid_exists(pid):
  try: print(psutil.Process(pid).status(),psutil.Process(pid).cmdline())
  except Exception as e:print(type(e).__name__)
for rid in ['RUN_BASELINE_SMOKE_S0','RUN_ACD_PIPELINE_S0']:
 p=root/rid/'stdout.log'
 print(rid,'bytes',p.stat().st_size,'mtime',time.time()-p.stat().st_mtime)
 print(p.read_text(encoding='utf-8',errors='replace')[-2000:])
