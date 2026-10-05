import datetime,json,os
from pathlib import Path
def now(): return datetime.datetime.now().astimezone().isoformat()
def atomic_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f".{os.getpid()}.tmp");t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");t.replace(p)
