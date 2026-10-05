ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
PATH='D:/coco_wire/prototype_guided_evidence_selection_20261003/.relay_bundles/RUN_973a07a938934b81bdd46796fcd203da/aab9d41b44356de6229800e9835a1a7a72e2aa150746b03df5b4860898ac9df0.zip'
WANT='ad98554d342e47fb4c66095686f4109c81d521b1c6f1503d78cdf38e8a4d4ff5'
from pathlib import Path
import hashlib,json
r=Path(ROOT)/'.relay_bundles'/RUN;p=Path(PATH)
if p.resolve().parent!=r.resolve() or p.suffix!='.zip':raise RuntimeError('Unsafe laptop bundle removal')
if p.exists():
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 if h.hexdigest()!=WANT:raise RuntimeError('Laptop bundle changed before removal')
 p.unlink()
print(json.dumps({'removed':str(p)}))
