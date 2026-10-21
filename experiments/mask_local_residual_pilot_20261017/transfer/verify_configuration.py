"""Post-run independent configuration inspection, not another model experiment."""
from pathlib import Path
import hashlib,json,sys
import torch

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def tensor_hash(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        t=t.cpu().contiguous();h.update(n.encode());h.update(str((list(t.shape),t.dtype)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    assert read(STUDY/'PROCESS_STATUS.json')['status']=='completed_and_verified'
    freeze=read(STUDY/'EXECUTION_FREEZE.json')
    for rel,digest in freeze['files'].items():assert sha(ROOT/rel)==digest,rel
    prep=STUDY/'runs'/read(STUDY/'PREPARE_POINTER.json')['run_id'];m=read(prep/'DATA_MANIFEST.json')
    split=read(ROOT/'experiments/mask_direction_learnability_20261016/SPLIT.json')
    orders=read(ROOT/'experiments/mask_direction_learnability_20261016/IMAGE_ORDERS.json')['epochs']
    assert len(m['records'])==1600 and {r['image_id'] for r in m['records']}==set(split['fit_image_ids'])
    assert not set(split['fit_image_ids'])&set(split['heldout_image_ids'])
    initial=torch.load(prep/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
    assert tensor_hash(initial)==m['initial_state_sha256'] and sha(prep/'initial.pt')==m['initial_sha256']
    for row in m['records']:assert sha(prep/row['path'])==row['sha256'],row['path']
    arms={}
    for arm in ('LOCAL','UNIFORM'):
        run=STUDY/'runs'/read(STUDY/f'FULL_{arm}_POINTER.json')['run_id']
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        start=torch.load(run/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
        assert all(torch.equal(start[n],initial[n]) for n in initial)
        assert all(t.dtype==torch.float32 for t in saved['state_dict'].values())
        assert sum(t.numel() for t in saved['state_dict'].values())==3297
        assert any(not torch.equal(initial[n],saved['state_dict'][n]) for n in initial)
        assert read(run/'IMAGE_ORDERS.json')==orders
        assert saved['steps']==1200 and saved['epochs']==3
        groups=saved['optimizer_state_dict']['param_groups']
        assert len(groups)==1 and len(groups[0]['params'])==6
        effective={k:v for k,v in groups[0].items() if k!='params'}
        assert json.dumps(effective,sort_keys=True)==json.dumps(read(STUDY/'CONFIG.json')['optimizer'],sort_keys=True)
        states=saved['optimizer_state_dict']['state']
        assert len(states)==6 and all(float(v['step'])==1200 for v in states.values())
        assert all(torch.isfinite(t).all() for v in states.values() for t in v.values() if torch.is_tensor(t))
        arms[arm]=dict(run_id=run.name,checkpoint_sha256=sha(run/'last.pt'),initial_state_sha256=tensor_hash(start),final_state_sha256=tensor_hash(saved['state_dict']),optimizer_parameter_states=len(states),optimizer_steps=1200,visits=4800)
    images=read(STUDY/'SOURCE_IMAGE_CACHE_AUDIT.json')
    assert images['status']=='passed' and images['source_images']==2000 and images['cache_files']==2000
    ev=STUDY/'runs'/read(STUDY/'EVALUATION_POINTER.json')['run_id']
    tasks=[json.loads(line) for line in (ev/'TASKS.jsonl').read_text(encoding='utf8').splitlines()]
    overlap=dict(local_only_recovered=[r['annotation_id'] for r in tasks if not r['N'] and r['L'] and not r['G']],
                 both_recovered=[r['annotation_id'] for r in tasks if not r['N'] and r['L'] and r['G']],
                 G_only_recovered=[r['annotation_id'] for r in tasks if not r['N'] and r['G'] and not r['L']],
                 local_only_damage=[r['annotation_id'] for r in tasks if r['N'] and not r['L'] and r['G']],
                 both_damaged=[r['annotation_id'] for r in tasks if r['N'] and not r['L'] and not r['G']])
    result=dict(status='passed',source_files=len(freeze['files']),prepared_fit_files=1600,heldout_training_visits=0,source_images=2000,source_caches=2000,parameter_count=3297,arms=arms,new_training_steps=0,existing_task_overlap=overlap,script_sha256=sha(Path(__file__)))
    with (STUDY/'INDEPENDENT_CONFIGURATION.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
