"""Read local raw COCO only to preselect IDs; no image/model outcomes."""
import json
from pathlib import Path
from readout_input_probe import stable_seed,sha,write_json
root=Path(__file__).resolve().parent
path=root.parents[1]/'datasets/coco/annotations/instances_train2017.json'
obj=json.loads(path.read_text(encoding='utf-8'))
images={r['id']:r for r in obj['images']}
chosen=sorted(images,key=lambda iid:stable_seed('readout-input:20260912',iid))[:96]
out=root/'local_readout_selection_20260912.json'
write_json(out,dict(fit=chosen[:32],transfer=chosen[32:],images=[images[i] for i in chosen],
                   all_train_images=len(images),annotation_sha256=sha(path),
                   source='All raw COCO train2017 image IDs before model/ICI filtering'))
print(json.dumps(dict(path=str(out),images=len(chosen),source_images=len(images),annotation_sha256=sha(path))))
