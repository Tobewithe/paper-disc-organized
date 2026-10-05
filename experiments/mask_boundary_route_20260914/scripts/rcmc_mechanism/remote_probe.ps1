$ErrorActionPreference = 'Stop'
$PSVersionTable.PSVersion.ToString()
Get-Command pwsh -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
$probeCode = @'
import sys,json,importlib.metadata as md
sys.path.insert(0,'D:/coco_wire/py')
import torch,numpy,cv2,ultralytics
from pathlib import Path
print(json.dumps({'python':sys.version,'executable':sys.executable,'ultralytics':ultralytics.__version__,'ultralytics_file':ultralytics.__file__,'torch':torch.__version__,'numpy':numpy.__version__,'cv2':cv2.__version__,'packages':{p:md.version(p) for p in ['pycocotools','scipy','scikit-learn','Pillow','matplotlib']},'cuda':torch.cuda.is_available()},indent=2))
for p in ['D:/coco_wire/data/annotations/instances_val2017.json','D:/coco_wire/data/images/val2017','D:/mdoeldata/pigcv-task05/models/yolo26m-seg.pt','D:/coco_wire/runs/mask_boundary_route_20260914/RUN_2ae67d6556a14848bceb5783a72e5790/candidate_records.csv','D:/coco_wire/runs/mask_boundary_route_20260914/RUN_e0e0d46defb74e3d9a90094f49ce26fe/decisions.csv','D:/coco_wire/scripts/runner.py']:
    x=Path(p);print(str(x),x.exists(),x.stat().st_size if x.is_file() else '')
'@
$probeCode | & 'C:/Users/28358/anaconda3/envs/pytorch/python.exe' -
if ($LASTEXITCODE -ne 0) { throw 'Environment probe failed' }
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3
Get-PSDrive D | Select-Object Name,Used,Free | ConvertTo-Json
