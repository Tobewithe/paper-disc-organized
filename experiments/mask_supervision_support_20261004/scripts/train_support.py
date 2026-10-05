"""Fixed 3-epoch mask-only support redistribution, native one2one cv4 only."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
if '--config' in sys.argv:
    _cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text(encoding='utf-8-sig'))
    sys.path.insert(0,_cfg['source_python'])
import torch
from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256
import dense_runtime as dr

def load_support(cfg, x, entry):
    path=Path(cfg['support_dir'])/entry['filename']
    assert sha256(path)==entry['sha256'], 'Support hash changed'
    s=torch.load(path,map_location='cpu',weights_only=False)
    n=len(x['raw_ids'])
    assert s['original_count']==entry['n_original']==n
    assert len(s['raw_ids'])==entry['n_total'] and int(s['image_id'])==int(x['image_id'])
    assert s['source_image_cache_sha256']==x['_asset_integrity']['compressed_sha256']
    for key in ('raw_ids','owners','target_boxes','levels'):
        torch.testing.assert_close(s[key][:n],x[key],atol=0,rtol=0)
    assert s['is_original'][:n].all() and not s['is_original'][n:].any()
    keys=('split','image_id','annotation_id','branch','raw_id','pyramid_level','target_gt_idx')
    for a,b in zip(s['rows'][:n],x['rows']):
        assert all(a[k]==b[k] for k in keys)
    assert len(set(s['raw_ids'].tolist()))==len(s['raw_ids'])
    assert set(s['owners'].tolist())==set(x['owners'].tolist())
    for owner in x['owners']:
        assert abs(float(s['weights'][s['owners']==owner].sum())-1.)<1e-12
    return s

def save(path, model, optimizer, epoch, cfg, history, channels, identity):
    tmp=Path(str(path)+'.tmp')
    torch.save(dict(mode='D',epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
        config=cfg,history=history,feature_channels=channels,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),support_identity=identity,
        checkpoint_rule='fixed final epoch3; no dev selection'),tmp)
    tmp.replace(path)

def run(cfg,out):
    started=time.monotonic()
    deadline=started+float(cfg['training_max_seconds'])
    def budget():
        if time.monotonic()>deadline:
            raise TimeoutError('Fixed 1800-second training budget exhausted; no auto-resume')
    assert cfg['epochs']==3 and cfg['training_max_seconds']==1800
    assert cfg['original_loss_weight']==.5 and cfg['dense_topk']==cfg['dense_topk2']==7
    if (out/'last.pt').exists() or (out/'HISTORY.json').exists():
        raise RuntimeError('Do not overwrite/restart an existing training Run')
    root=Path(cfg['server_root']); ids=load_json(root/'RUN_IDS.json')['runs']
    receipts={}
    for stage in ('smoke','prepare'):
        p=root/'runs'/ids[stage]/'COMPLETE.json'; r=load_json(p)
        assert r['passed'], stage+' not passed'
        receipts[stage]=dict(path=str(p),sha256=sha256(p))
    idx=load_index(cfg); planned=idx['fit']; items=[r for r in planned if r['n']>0]
    assert len(planned)==1024 and len(idx['dev'])==256 and not idx.get('val')
    support_index=Path(cfg['support_dir'])/'INDEX.json'
    si=load_json(support_index); assert si['complete'] and not si['smoke']
    assert si['planned_image_ids']==[int(r['image_id']) for r in planned]
    entries={int(e['image_id']):e for e in si['images']}
    assert len(entries)==len(planned)
    for r in planned: assert entries[int(r['image_id'])]['n_original']==r['n']
    dr.setup(cfg['seed']); replay=FrozenReplay(cfg,device='cuda')
    source_initial=dr.state_digest(replay.source)
    model=dr.new_model(replay,cfg).train(); control=dr.check_control(cfg,model)
    optimizer=dr.optimizer_for(model,cfg); original_buffers=dr.buffers(model)
    initial_parameters={k:p.detach().clone() for k,p in model.named_parameters()}
    initial_sha=hashlib.sha256(json.dumps(dr.state_digest(model),sort_keys=True).encode()).hexdigest()
    identity=dict(index_sha256=sha256(support_index),receipts=receipts,control=control)
    dump(out/'ENVIRONMENT.json',replay.import_info); dump(out/'RESOLVED_CONFIG.json',cfg)
    dump(out/'MODEL.json',dict(mode='D',architecture='unchanged native-only N',counts=model.parameter_counts(),
        trainable=model.trainable_parameter_names(),initialization_state_sha256=initial_sha,
        feature_channels=replay.feature_channels,source_initial=source_initial,
        bn_running_buffers='fixed original; affine trainable',objective='per GT .5 original + .5 mean extra; no-extra original 1',
        support_identity=identity,budget_seconds=1800))
    dump(out/'COHORT.json',dict(planned_images=len(planned),effective_images=len(items),
        original_candidates=sum(r['n'] for r in items),support_positions=sum(e['n_total'] for e in entries.values()),
        no_positive_images=[r['image_id'] for r in planned if not r['n']],
        domain='original one2one evaluation population; mask-only topk2=7 fit support'))
    control_history=load_json(Path(cfg['control_run'])/'HISTORY.json')
    history=[]; updates=0
    trace=(out/'UPDATES.jsonl').open('x',encoding='utf-8')
    try:
        for epoch in range(cfg['epochs']):
            budget(); order=list(items); random.Random(cfg['seed']+epoch).shuffle(order)
            order_sha=hashlib.sha256(json.dumps([int(r['image_id']) for r in order],separators=(',',':')).encode()).hexdigest()
            assert order_sha==control_history[epoch]['image_order_sha256'], 'Unpaired image order'
            epoch_start=time.monotonic(); total=0.; seen=positions=0; groups=math.ceil(len(order)/cfg['effective_batch_images'])
            for step,offset in enumerate(range(0,len(order),cfg['effective_batch_images'])):
                budget(); group=order[offset:offset+cfg['effective_batch_images']]
                denominator=sum(r['n'] for r in group)
                factor=dr.lr_factor(epoch+(step+1)/groups,cfg)
                for og in optimizer.param_groups: og['lr']=og['initial_lr']*factor
                optimizer.zero_grad(set_to_none=True)
                for inner in range(0,len(group),cfg['microbatch_images']):
                    budget(); subset=group[inner:inner+cfg['microbatch_images']]
                    images=[load_asset(cfg,r['image_id'],verify=True) for r in subset]
                    supports=[load_support(cfg,x,entries[int(x['image_id'])]) for x in images]
                    selections=[dict(raw_ids=s['raw_ids'].to(replay.device)) for s in supports]
                    features=replay.replay(images); coefficients=model(features,selections); gradients=[]
                    for x,s,c in zip(images,supports,coefficients):
                        payload=dr.gpu_payload(x,replay.device)
                        payload.update(owners=s['owners'].to(replay.device),target_boxes=s['target_boxes'].to(replay.device))
                        value,gradient=dr.weighted_loss_and_grad(payload,c,denominator,s['weights'])
                        assert bool(torch.isfinite(gradient).all())
                        gradients.append(gradient); total+=value; seen+=s['original_count']; positions+=len(c)
                    torch.autograd.backward(coefficients,gradients)
                    del images,supports,features,coefficients,gradients,selections
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True))
                before=[p.detach().clone() for p in model.parameters()]
                optimizer.step(); updates+=1
                update_norm=math.sqrt(sum(float((p.detach()-old).double().square().sum()) for p,old in zip(model.parameters(),before)))
                trace.write(json.dumps(dict(epoch=epoch+1,step=step+1,update=updates,gradient_norm_before_clip=norm,
                    update_l2=update_norm,lr=cfg['branch_lr']*factor,original_denominator=denominator))+'\n')
                if step%25==0 or step+1==groups:
                    trace.flush(); torch.cuda.synchronize()
                    progress=dict(stage='training',mode='D',epoch=epoch+1,epochs=3,step=step+1,steps=groups,
                        trajectory_weighted_bce=total/max(seen,1),original_candidates=seen,support_positions=positions,
                        gradient_norm_before_clip=norm,update_l2=update_norm,elapsed_s=time.monotonic()-started,
                        remaining_budget_s=deadline-time.monotonic(),peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
            assert seen==sum(r['n'] for r in items)
            assert positions==sum(e['n_total'] for e in entries.values())
            dr.verify_buffers(model,original_buffers);replay.assert_unchanged()
            assert dr.state_digest(replay.source)==source_initial
            assert all(p.grad is None for p in replay.source.parameters())
            history.append(dict(epoch=epoch+1,trajectory_weighted_bce=total/seen,trajectory_not_fixed_checkpoint_objective=True,
                candidates=seen,support_positions=positions,images=len(items),seconds=time.monotonic()-epoch_start,
                image_order_sha256=order_sha,bn_unchanged=True,all_buffers_unchanged=True,replay_source_unchanged=True))
            save(out/'last.pt',model,optimizer,epoch+1,cfg,history,replay.feature_channels,identity)
            dump(out/'HISTORY.json',history)
        budget()
        changed=[k for k,p in model.named_parameters() if not torch.equal(p.detach(),initial_parameters[k])]
        assert changed, 'No native parameters updated'
        displacement=math.sqrt(sum(float((p.detach()-initial_parameters[k]).double().square().sum()) for k,p in model.named_parameters()))
        control_pt=torch.load(cfg['control_checkpoint'],map_location='cpu',weights_only=False)
        n_displacement=math.sqrt(sum(float((control_pt['state_dict'][k].to(p.device)-initial_parameters[k]).double().square().sum()) for k,p in model.named_parameters()))
        assert sha256(support_index)==identity['index_sha256']
        assert sha256(cfg['control_checkpoint'])==control['checkpoint_sha256']
        save(out/'final.pt',model,optimizer,3,cfg,history,replay.feature_channels,identity)
        dump(out/'UPDATE_AUDIT.json',dict(passed=True,changed_parameters=changed,updates=updates,
            displacement_D=displacement,displacement_N=n_displacement,
            note='Same image/optimizer updates and total GT weight do not guarantee same gradient magnitude or pixel FLOPs'))
        dump(out/'COMPLETE.json',dict(completed=True,passed=True,kind='training',mode='D',epochs=3,
            checkpoint='final.pt',checkpoint_sha256=sha256(out/'final.pt'),checkpoint_rule='fixed epoch3',
            elapsed_training_s=time.monotonic()-started,all_frozen_states_unchanged=True,automatic_retry=False))
    finally:
        trace.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    try:
        if os.name=='nt' or not torch.cuda.is_available(): raise RuntimeError('Authorized Linux CUDA server only')
        run(resolve_runtime_config(load_json(args.config)),out)
    except BaseException as exc:
        dump(out/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False))
        dump(out/'COMPLETE.json',dict(completed=False,passed=False,error=repr(exc)))
        raise
if __name__=='__main__': main()
