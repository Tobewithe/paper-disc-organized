ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_91811d001f384d7eb952f512c631dcdd'
PATH='D:/coco_wire/prototype_guided_evidence_selection_20261003/.relay_bundles/RUN_91811d001f384d7eb952f512c631dcdd/3f438894de1e796a529f6685102bca7978f6ab22337ea04b7f96799cf87739e2.zip'
WANT='8e35e9ee3d97f28b1402efcb889b9f7561b63819fe04ad2ebe2fa3ee734ba39d'
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
