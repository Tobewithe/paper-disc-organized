ROOT='D:/coco_wire/prototype_guided_evidence_selection_20261003'
RUN='RUN_3617758ad03b4eb1934a8b1a9a654cb6'
PATH='D:/coco_wire/prototype_guided_evidence_selection_20261003/.relay_bundles/RUN_3617758ad03b4eb1934a8b1a9a654cb6/7afa001c7864cca8f1aa77dd7c8767b841563ee2835ee091cde3759adb72abfc.zip'
WANT='454e1f71aad892dfb9f150bda05ca6b888dc9a53c3161093322b716388655485'
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
