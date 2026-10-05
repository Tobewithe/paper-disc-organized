import json
from pathlib import Path
from relative_ownership_experiment import read
from frozen_mechanism_probe import ROOT

available = {int(p.stem) for p in (ROOT/'data/images/train2017').glob('*.jpg')}
old = set(json.loads((ROOT/'diagnostics/coefficient_pilot_cache_v4_20260911/selection.json').read_text())['train'])
witness = set(json.loads((ROOT/'diagnostics/structure_train_witness32_v2_20260911/protocol.json').read_text())['images'])
eligible = available-old-witness
rows = [r for r in read(ROOT/'census/train2017_images.csv') if int(r['image_id']) in eligible]
print(json.dumps(dict(available=len(available), eligible=len(rows),
                     high_images=sum(int(r['n_high_same'])>0 for r in rows),
                     zero_high_images=sum(int(r['n_high_same'])==0 for r in rows),
                     high_gt=sum(int(r['n_high_same']) for r in rows),
                     all_gt=sum(int(r['n_instances']) for r in rows))))
