"""Sensitivity only: do not silently relabel frozen subsets or bins."""
import csv
import json
from pathlib import Path

p=Path(__file__).resolve().parent/'census'
out={};eps=1e-10
for split in ['train2017','val2017']:
    raw=[0,0];snap=[0,0];count=0;changes=[]
    with (p/(split+'_instances.csv')).open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            x=float(r['ici_same']);raw[0]+=x>.5;raw[1]+=x>1
            y=.5 if abs(x-.5)<=eps else 1. if abs(x-1)<=eps else x
            snap[0]+=y>.5;snap[1]+=y>1;count+=1
            if (x>.5)!=(y>.5):changes.append(dict(image_id=int(r['image_id']),annotation_id=int(r['annotation_id']),raw=x))
    out[split]=dict(instances=count,raw_high_gt05=raw[0],tolerance_high_gt05=snap[0],raw_extreme_gt1=raw[1],tolerance_extreme_gt1=snap[1],high_group_changes=changes)
(p/'ICI_BOUNDARY_SENSITIVITY.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out),flush=True)
