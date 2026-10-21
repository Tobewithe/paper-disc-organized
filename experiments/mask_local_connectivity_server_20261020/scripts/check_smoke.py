"""CPU-only cross-device gate; no optimizer or inference."""
import json,hashlib
from pathlib import Path
S=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows={}
for arm in 'ABCD':
    folder=S/'arms'/arm;p=read(folder/'SMOKE_POINTER.json');run=folder/'runs'/p['run_id']
    assert read(run/'run.json')['execution_status']=='completed'
    assert sha(run/'SUMMARY.json')==p['summary_sha256']
    rows[arm]=read(run/'SUMMARY.json')
    assert rows[arm]['status']=='passed' and rows[arm]['optimizer_steps']==4
for key in ['initial_tensor_sha256','train_identity','val_identity','train_source_batch','native_G_rows_exact']:
    assert all(rows[a][key]==rows['A'][key] for a in 'ABCD'),key
out={'status':'passed','freeze_sha256':sha(S/'SERVER_FREEZE_MANIFEST.json'),'cross_gpu_native_identity_exact':True,'initial_tensor_sha256':rows['A']['initial_tensor_sha256'],'smoke_steps_discarded':16,'arms':{a:{'forecast_hours':r['forecast_hours'],'native_G_rows_exact':r['native_G_rows_exact'],'cuda_peak_reserved':r['cuda_peak_reserved']} for a,r in rows.items()}}
(S/'SERVER_SMOKE_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
