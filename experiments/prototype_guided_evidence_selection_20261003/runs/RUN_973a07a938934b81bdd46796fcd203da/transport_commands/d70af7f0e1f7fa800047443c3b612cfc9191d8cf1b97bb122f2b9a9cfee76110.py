ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
PATH='D:/coco_wire/prototype_guided_evidence_selection_20261003/.relay_bundles/RUN_973a07a938934b81bdd46796fcd203da/36c3242b0ee357fb03bcd1b207b4ea62db6eade7c0125a622de3a7a1965fa289.zip'
WANT='91991bc54dfea1c561cf36ffa59a4457be0706d3137e6f415302f4dab39587d0'
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
