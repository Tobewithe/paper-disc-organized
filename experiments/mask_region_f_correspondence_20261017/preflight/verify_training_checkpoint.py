import json,hashlib,torch,sys
from pathlib import Path
s=Path('experiments/mask_region_f_correspondence_20261017');pair=json.loads((s/'TRAINING_PAIR_FREEZE.json').read_text(encoding='utf8'));out={}
for arm in ('A','Q'):
 p=s/'runs'/pair[arm]['run_id']; saved=torch.load(p/'last.pt',map_location='cpu',weights_only=True)
 h=hashlib.sha256()
 for name,value in sorted(saved['state_dict'].items()):
  t=value.detach().cpu().contiguous();h.update(name.encode());h.update(str((list(t.shape),t.dtype)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
 summary=json.loads((p/'SUMMARY.json').read_text(encoding='utf8'))
 assert h.hexdigest()==summary['final_state_sha256']
 steps={str(k):int(v['step'].item()) for k,v in saved['optimizer_state_dict']['state'].items()}
 assert len(steps)==17 and set(steps.values())=={1500}
 assert all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values())
 out[arm]={'tensor_state_sha256':h.hexdigest(),'actual_optimizer_state_steps':steps,'all_parameters_finite':True,'checkpoint_sha256':hashlib.sha256((p/'last.pt').read_bytes()).hexdigest()}
receipt={'status':'passed','purpose':'CPU inspection of saved checkpoint/optimizer tensors; no forward or training','interpreter':sys.executable,'python':sys.version,'torch':torch.__version__,'A_Q':out}
target=s/'preflight/TRAINING_CHECKPOINT_INDEPENDENT_LOCAL.json'
with target.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
