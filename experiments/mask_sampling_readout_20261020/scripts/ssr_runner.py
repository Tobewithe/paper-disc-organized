"""Finite laptop smoke, six-arm training and full-output evaluation stages."""
from __future__ import annotations
import argparse,contextlib,copy,json,math,os,shutil,sys,time
from ssr_runtime import *

def prepare_contract():
    # Preparation is non-model execution; official SHA identity already validated.
    parent=read(STUDY/'inputs/PARENT_DATA_CONTRACT.json')
    paths=read(STUDY/'inputs/IMAGE_PATHS.json');config=read(STUDY/'CONFIG.json')
    required=('weights','profile','adapter_initial','existing_G','train_annotations','val_annotations','historical_native_trace')
    contract=copy.deepcopy(parent)
    from native_stream import historical_val_plan
    contract['val_batches']=historical_val_plan(read(config['historical_native_trace']))
    contract['inputs']={k:{'path':config[k],'sha256':sha(config[k])} for k in required}
    write(STUDY/'inputs/DATA_CONTRACT.json',contract)
    shutil.copyfile(STUDY/'inputs/PARENT_IMAGE_ORDERS.json',STUDY/'inputs/IMAGE_ORDERS.json')
    for split in ('train','val'):
        assert set(paths[split])==set(str(i) for i in parent[split+'_ids'])
        (STUDY/'inputs'/(split+'.txt')).write_text(''.join(paths[split][str(i)]+'\n' for i in parent[split+'_ids']),encoding='utf-8')
    print('DATA_CONTRACT_READY',flush=True)

def new_models(config,training=True):
    import torch
    from ssr_model import ARMS,make_model
    initial=torch.load(config['adapter_initial'],map_location='cpu',weights_only=True)
    models={arm:make_model(arm,initial).cuda() for arm in ARMS}
    for model in models.values():model.train(training)
    return models

def native(config,split,run):
    from native_stream import NativeBatches
    return NativeBatches(config,STUDY/'inputs',split,run.path/('native_'+split))

def loss_for(model,cache,rows,features,targets,phase):
    from ssr_model import predict
    from adapter_model import loss_grids
    from soft_iou_loss import soft_iou_grouped_loss
    delta=predict(model,cache,features,phase)
    logits,baseline,roi=loss_grids(cache,cache['c0']+delta)
    return soft_iou_grouped_loss(logits,baseline,roi,targets)

def smoke():
    import torch
    from pycocotools.coco import COCO
    from adapter_data import training_join,projected_targets
    from native_output_bridge import decode_native,encode_rle
    from ssr_model import predict
    with Run('smoke') as run:
        profile,env=environment();write(run.path/'ENVIRONMENT.json',env)
        stream=native(run.config,'train',run)
        base_before=state_hash(stream.model)
        index=max(range(len(stream.plan)),key=lambda i:(math.prod(stream.plan[i]['rect_shape']),-i))
        started=time.perf_counter();caches,rows,features=stream.batch(index,masks=True)
        native_seconds=time.perf_counter()-started
        coco=COCO(run.config['train_annotations'])
        targets=[projected_targets(coco,c,training_join(r,coco.imgToAnns.get(c['image_id'],[]))) for c,r in zip(caches,rows)]
        models=new_models(run.config);receipts={};steps=0
        for arm,model in models.items():
            opt=optimizer(model,run.config);initial=state_hash(model);grads=set();times=[]
            with torch.no_grad():
                for c,r in zip(caches,rows):
                    d=predict(model,c,features,'eval:0')
                    require(torch.count_nonzero(d).item()==0,'initial residual is not zero')
                    if r:
                        decoded=decode_native(c,c['c0']+d)
                        require(all(encode_rle(decoded[i])==p['segmentation'] for i,p in enumerate(r)),'initial zero-delta RLE mismatch')
            for group in range(4):
                run.check();torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
                for i in range(group*4,group*4+4):
                    loss,account=loss_for(model,caches[i],rows[i],features,targets[i],'train:0')
                    require(torch.isfinite(loss).item(),'nonfinite smoke loss');(loss/4).backward()
                for name,p in model.named_parameters():
                    if p.grad is not None:
                        require(torch.isfinite(p.grad).all().item(),'nonfinite gradient')
                        if torch.count_nonzero(p.grad).item():grads.add(name)
                opt.step();steps+=1;torch.cuda.synchronize();times.append(time.perf_counter()-t)
                run.update(phase='smoke_updates',arm=arm,optimizer_steps=steps)
            require(state_hash(model)!=initial,'smoke parameters unchanged')
            represented={l['level'] for c in caches for l in c['raw_locations']}
            inactive={n for n,p in model.named_parameters()}-grads
            allowed={n for n in inactive if any(n.startswith(f'feature.{l}.') for l in set(range(3))-represented)}
            require(inactive<=allowed,'unexplained inactive parameters: '+str(inactive-allowed))
            receipts[arm]={'parameters':sum(p.numel() for p in model.parameters()),'gradient_names':sorted(grads),
                           'inactive_source_levels_only':sorted(allowed),'four_update_seconds':times,'initial_RLE_exact':True}
            print('SMOKE_ARM',arm,receipts[arm]['parameters'],times,flush=True)
        estimated=(sum(max(v['four_update_seconds']) for v in receipts.values())+native_seconds/4)*15000/.75
        require(estimated<=96*3600,'smoke forecast exceeds fixed total 96h training budget')
        require(state_hash(stream.model)==base_before,'base model changed')
        del models,opt,features,caches,rows,targets,stream,coco
        torch.cuda.empty_cache()
        val=native(run.config,'val',run);caches,rows,features=val.batch(0,masks=True)
        # The historical validation trace is the laptop source of tensor and RLE identity.
        hist={r['image_id']:r for r in read(run.config['historical_native_trace'])['images']}
        for c in caches:
            h=hist[c['image_id']]
            for key in ('image_sha256','input_tensor_sha256','input_hw','original_hw','batch_index','batch_position'):
                require(json.loads(json.dumps(c[key]))==json.loads(json.dumps(h[key])),'historical val geometry drift: '+key)
            original=torch.load(Path(run.config['historical_native_trace']).parent/h['cache'],map_location='cpu',weights_only=True)
            for key in ('proto','c0','boxes_input','scores','classes'):
                require(torch.equal(c[key].detach().cpu(),original[key]),'historical val tensor drift: '+key)
        write(run.path/'ENGINEERING_RECEIPTS.json',receipts)
        run.finish({'status':'passed','train_source_batch':index,'native_images':32,'steps_per_arm':4,'optimizer_steps':steps,
                    'all_initial_RLE_exact':True,'all_trainable_gradients_verified':True,'historical_val_exact':True,
                    'forecast_training_seconds_with_25percent_reserve':estimated,'forecast_training_hours':estimated/3600,
                    'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()})

def train():
    import torch
    from ssr_model import ARMS
    from pycocotools.coco import COCO
    from adapter_data import training_join,projected_targets
    from native_stream import batch_identity
    require(read(pointer('smoke')/'SUMMARY.json')['status']=='passed','smoke prerequisite')
    require(not any(read(p).get('optimization_started') for p in (STUDY/'runs').glob('RUN_SSR_TRAIN_*/run.json')),'formal budget already started; no automatic restart')
    with Run('train') as run:
        profile,env=environment();write(run.path/'ENVIRONMENT.json',env)
        stream=native(run.config,'train',run);base_before=state_hash(stream.model)
        models=new_models(run.config);opts={a:optimizer(m,run.config) for a,m in models.items()}
        counts={a:0 for a in ARMS};images={a:0 for a in ARMS};update_counts={a:{n:0 for n,p in m.named_parameters()} for a,m in models.items()}
        for arm in ARMS:
            folder=run.path/arm;folder.mkdir();save_model(folder/'initial.pt',models[arm],opts[arm],0,0)
            write(folder/'PARAMETERS.json',{n:list(p.shape) for n,p in models[arm].named_parameters()})
            write(folder/'ACTUAL_OPTIMIZER.json',{k:v for k,v in opts[arm].param_groups[0].items() if k!='params'})
        orders=read(STUDY/'inputs/IMAGE_ORDERS.json');coco=COCO(run.config['train_annotations'])
        identities={};joins={}
        with (run.path/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for epoch,sequence in enumerate(orders):
                for item in sequence:
                    run.check();t=time.perf_counter();bi=item['source_batch_index']
                    caches,rows,features=stream.batch(bi,masks=False)
                    identity=json.loads(json.dumps(batch_identity(caches,rows,features)))
                    if str(bi) in identities:require(identities[str(bi)]==identity,'frozen native tensors changed across epochs')
                    else:identities[str(bi)]=identity
                    targets=[]
                    for c,r in zip(caches,rows):
                        join=training_join(r,coco.imgToAnns.get(c['image_id'],[]));key=str(c['image_id'])
                        normalized={str(k):v for k,v in join.items()}
                        if key in joins:require(joins[key]==normalized,'training join changed')
                        else:joins[key]=normalized
                        targets.append(projected_targets(coco,c,join))
                    run.update(phase='formal_training',optimization_started=True,epoch=epoch+1,source_batch=bi,
                               optimizer_steps=sum(counts.values()),steps_by_arm=counts,images_by_arm=images)
                    for selection in item['microgroups']:
                        for arm in ARMS:
                            run.check();model=models[arm];opt=opts[arm];opt.zero_grad(set_to_none=True)
                            losses=[]
                            for i in selection:
                                loss,account=loss_for(model,caches[i],rows[i],features,targets[i],'train:'+str(epoch))
                                require(torch.isfinite(loss).item(),'nonfinite formal loss')
                                (loss/len(selection)).backward();losses.append(float(loss.detach()))
                            for name,p in model.named_parameters():
                                if p.grad is not None:
                                    require(torch.isfinite(p.grad).all().item(),'nonfinite formal gradient')
                                    update_counts[arm][name]+=1
                            opt.step();counts[arm]+=1;images[arm]+=len(selection)
                            log.write(json.dumps({'arm':arm,'epoch':epoch+1,'step':counts[arm],'image_ids':[caches[i]['image_id'] for i in selection],
                                                  'mean_loss':sum(losses)/len(losses)},allow_nan=False)+'\n')
                    if counts['D']%20==0:
                        run.update(phase='formal_training',epoch=epoch+1,source_batch=bi,optimizer_steps=sum(counts.values()),steps_by_arm=counts,images_by_arm=images)
                        print('TRAIN',epoch+1,counts,'batch_seconds',round(time.perf_counter()-t,3),flush=True)
                    del caches,rows,features,targets,loss
                require(state_hash(stream.model)==base_before,'base changed at epoch end')
                for arm in ARMS:
                    save_model(run.path/arm/('last.pt' if epoch==2 else f'epoch_{epoch+1}_unselected.pt'),models[arm],opts[arm],counts[arm],epoch+1)
                write(run.path/'TRAINING_NATIVE_IDENTITY.json',identities);write(run.path/'TRAINING_JOIN.json',joins)
            require(all(v==15000 for v in counts.values()) and all(v==60000 for v in images.values()) and len(joins)==20000,'formal budget incomplete')
            for arm in ARMS:
                for name,p in models[arm].named_parameters():
                    actual=int(opts[arm].state.get(p,{}).get('step',0))
                    require(actual==update_counts[arm][name],'optimizer update accounting differs')
            write(run.path/'PARAMETER_UPDATE_COUNTS.json',update_counts)
            require(state_hash(stream.model)==base_before,'base changed')
        run.update(optimizer_steps=sum(counts.values()),steps_by_arm=counts,images_by_arm=images)
        run.finish({'status':'completed','steps_by_arm':counts,'images_by_arm':images,'epochs':3,'training_images':len(joins),
                    'all_native_inputs_and_targets_shared':True,'frozen_base_exact':True,
                    'checkpoints':{a:{'path':a+'/last.pt','sha256':sha(run.path/a/'last.pt')} for a in ARMS}},'trained_not_yet_evaluated')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare_contract','smoke','train','evaluate','statistics','pipeline'))
    args=ap.parse_args()
    if args.stage=='prepare_contract':prepare_contract();return
    if args.stage=='smoke':smoke();return
    if args.stage=='train':train();return
    if args.stage=='evaluate':
        from ssr_evaluation import evaluate_stage
        evaluate_stage();return
    if args.stage=='statistics':
        from ssr_statistics import statistics_stage
        statistics_stage();return
    if args.stage=='pipeline':
        log=(STUDY/'pipeline.log').open('x',encoding='utf-8',buffering=1)
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                smoke();train()
                from ssr_evaluation import evaluate_stage
                evaluate_stage()
                from ssr_statistics import statistics_stage
                statistics_stage()
            except BaseException:
                import traceback
                traceback.print_exc();raise

if __name__=='__main__':main()
