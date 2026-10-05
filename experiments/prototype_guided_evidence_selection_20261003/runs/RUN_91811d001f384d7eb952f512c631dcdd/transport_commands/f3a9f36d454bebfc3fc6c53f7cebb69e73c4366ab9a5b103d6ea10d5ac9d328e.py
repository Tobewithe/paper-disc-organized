ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_91811d001f384d7eb952f512c631dcdd'
PATH='D:/coco_wire/prototype_guided_evidence_selection_20261003/.relay_bundles/RUN_91811d001f384d7eb952f512c631dcdd/9de5667cf34f63a8b97e5dfc9755e3eeb0656ec978a8962754f392a931e1ff66.zip'
WANT='3e4c52ae5127cffcf165054a6746536be1e015e70b82ec4da3f6f6429abe5e82'
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
