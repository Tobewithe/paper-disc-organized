from lrp_common import *
import argparse,torch
from lrp_math import PixelResidual,objective

def main():
    p=argparse.ArgumentParser(); p.add_argument('--arm',choices=['LOCAL','UNIFORM'],required=True); p.add_argument('--scope',choices=['smoke','full'],required=True)
    args=p.parse_args(); full=args.scope=='full'
    prep=STUDY/'runs'/read_json(STUDY/'PREPARE_POINTER.json')['run_id']; completed(prep)
    data=inspect_inputs(); manifest=read_json(prep/'DATA_MANIFEST.json')
    if full:
        for arm in ('LOCAL','UNIFORM'):
            smoke=STUDY/'runs'/read_json(STUDY/f'SMOKE_{arm}_POINTER.json')['run_id']; completed(smoke)
            require(read_json(smoke/'SUMMARY.json')['nonzero_gradient_parameters'],'invalid smoke')
        previous=[r for r in (STUDY/'runs').glob(f'RUN_LRP_FULL_{args.arm}_*/run.json') if read_json(r).get('optimization_started')]
        require(not previous,'formal arm already consumed budget')
    with Run(('FULL_' if full else 'SMOKE_')+args.arm,prep,7200 if full else 900) as run:
        orders=read_json(CLD/'IMAGE_ORDERS.json')['epochs'] if full else [data['split']['fit_image_ids'][:8]]
        expected_steps=1200 if full else 2
        for rec in manifest['records']: require(sha256(prep/rec['path'])==rec['sha256'],'training tensor changed')
        require(sha256(prep/'initial.pt')==manifest['initial_sha256'],'initial changed')
        cfg=read_json(STUDY/'CONFIG.json')
        contract=dict(arm=args.arm,scope=args.scope,expected_steps=expected_steps,expected_visits=expected_steps*4,
                      environment=data['environment'],data_manifest_sha256=sha256(prep/'DATA_MANIFEST.json'),
                      initial_sha256=manifest['initial_sha256'],initial_state_sha256=manifest['initial_state_sha256'],optimizer=cfg['optimizer'])
        run.freeze(contract); write_new(run.path/'IMAGE_ORDERS.json',orders)
        torch.manual_seed(0); model=PixelResidual().cuda(); model.load_state_dict(torch.load(prep/'initial.pt',map_location='cpu',weights_only=True)['state_dict'])
        require(tensor_state_hash(model)==manifest['initial_state_sha256'],'initial tensor mismatch')
        spec=dict(cfg['optimizer']); spec['betas']=tuple(spec['betas'])
        optimizer=torch.optim.AdamW(model.parameters(),**spec)
        actual={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
        require(json.dumps(actual,sort_keys=True)==json.dumps(cfg['optimizer'],sort_keys=True),'optimizer defaults drift')
        write_new(run.path/'ACTUAL_OPTIMIZER.json',actual)
        torch.save(dict(state_dict=model.state_dict()),run.path/'initial.pt')
        index={r['image_id']:r for r in manifest['records']}
        gradients=set(); steps=0; visits=0; logs=[]
        run.update(optimization_started=True)
        with (run.path/'training.jsonl').open('x',encoding='utf8',buffering=1) as log:
            for epoch,order in enumerate(orders):
                for offset in range(0,len(order),4):
                    group=order[offset:offset+4]; require(len(group)==4,'partial image batch')
                    optimizer.zero_grad(set_to_none=True)
                    for iid in group:
                        run.check_budget(); require(iid in index and iid not in data['split']['heldout_image_ids'],'training leakage')
                        payload=torch.load(prep/index[iid]['path'],map_location='cpu',weights_only=True)
                        value,info=objective(model,payload['instances'],args.arm)
                        require(torch.isfinite(value).item(),'nonfinite objective')
                        (value/4).backward(); visits+=1
                        log.write(json.dumps(dict(epoch=epoch,step=steps+1,image_id=iid,loss=float(value.detach()),**info))+'\n')
                    for name,param in model.named_parameters():
                        require(param.grad is not None and torch.isfinite(param.grad).all(),'invalid gradient')
                        if torch.count_nonzero(param.grad): gradients.add(name)
                    optimizer.step(); steps+=1
                    require(all(torch.isfinite(p).all() for p in model.parameters()),'nonfinite parameter')
                    logs.append(dict(step=steps,epoch=epoch,image_ids=group))
                    if steps%50==0 or not full:
                        run.update(optimizer_steps=steps,image_visits=visits); print('TRAIN',args.arm,steps,'/',expected_steps,flush=True)
        require(steps==expected_steps and visits==expected_steps*4 and gradients,'training budget/gradient')
        require(tensor_state_hash(model)!=manifest['initial_state_sha256'],'no parameter change')
        torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),arm=args.arm,steps=steps,epochs=len(orders)),run.path/'last.pt')
        jsonl(run.path/'OPTIMIZER_STEPS.jsonl',logs)
        for rec in manifest['records']: require(sha256(prep/rec['path'])==rec['sha256'],'training input drift after execution')
        config_audit(run,optimizer_steps=steps,image_visits=visits,parameters=3297,trainable_whitelist=[n for n,_ in model.named_parameters()],heldout_visits=0,
                     initial_state_sha256=manifest['initial_state_sha256'],final_state_sha256=tensor_state_hash(model),checkpoint_sha256=sha256(run.path/'last.pt'))
        run.update(optimizer_steps=steps,image_visits=visits)
        run.finish(dict(arm=args.arm,steps=steps,image_visits=visits,nonzero_gradient_parameters=sorted(gradients),final_state_sha256=tensor_state_hash(model),checkpoint_sha256=sha256(run.path/'last.pt')),scientific='trained_not_evaluated' if full else 'engineering_only')
    save(STUDY/f'{"FULL" if full else "SMOKE"}_{args.arm}_POINTER.json',dict(run_id=run.run_id))

if __name__=='__main__': main()
