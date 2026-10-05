"""Export boundary-stable full-val GT metadata for the existing official evaluator."""
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parent
old=root.parents[1]/'refine-logs/coco-evaluation/COCO_INSTANCE_MANIFEST_20260911_032703.csv'
with old.open(encoding='utf-8-sig') as f:legacy={int(r['annotation_id']):r for r in csv.DictReader(f)}
with (root/'census/val2017_instances.csv').open(encoding='utf-8-sig') as f:current=list(csv.DictReader(f))
assert len(current)==len(legacy)==36335
rows=[]
for r in current:
    row=legacy[int(r['annotation_id'])].copy()
    row.update(ici_same=r['ici_same'],ici_all_categories=r['ici_all'],ici_bin=r['ici_bin'])
    rows.append(row)
dest=root/'census/COCO_EVAL_INSTANCE_MANIFEST.csv'
with dest.open('w',encoding='utf-8-sig',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
assert sum(float(r['ici_same'])>.5 for r in rows)==5245
assert sum(float(r['ici_same'])>1 for r in rows)==1523
print('EVAL_MANIFEST_PASS',len(rows),flush=True)
