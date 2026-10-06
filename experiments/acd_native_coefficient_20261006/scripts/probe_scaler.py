import sys,json
sys.path.insert(0,'D:/coco_wire/vendor_8.4.100')
import torch
from pathlib import Path
p=Path('D:/coco_wire/experiments/acd_native_coefficient_20261006/runs/RUN_BASELINE_FEASIBILITY_S0/trainer/weights/last.pt')
x=torch.load(p,map_location='cpu',weights_only=False)
print(json.dumps({'epoch_zero_based':x['epoch'],'scaler':x.get('scaler'),'optimizer_groups':len(x.get('optimizer',{}).get('param_groups',[]))}))
