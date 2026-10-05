ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_973a07a938934b81bdd46796fcd203da'
PATH='D:/coco_wire/prototype_guided_evidence_selection_20261003/.relay_bundles/RUN_973a07a938934b81bdd46796fcd203da/36b6a11362389862c4e8890549c8b0d3de2cd17577bff30348a4b677f6d232bb.zip'
WANT='f1bc1630c3f9563773b907e8a6e555205c261aa8d234d2d9ef6cf0c9d758b0d1'
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
