"""Registered engineering smoke or one fixed three-epoch B/C adapter arm."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from posttrain_run import (ROOT, STUDY, RegisteredRun, configure_native, tensor_state_hash,
                           read_json, require, sha256, write_new)

BASE = ROOT / 'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
WEIGHT = ROOT / 'assets/models/coco_clean_20260911/yolo26m-seg.pt'
CODE_FILES = ['experiments/mask_readout_mechanism_v2_20261012/scripts/' + n for n in (
    'run_s2_training.py','adapter_model.py','adapter_data.py','adapter_primitives.py','s1_geometry.py',
    'posttrain_run.py','test_adapter_model.py')] + [
    'experiments/mask_readout_mechanism_v2_20261012/S2_EXECUTION_20261013.md',
    'experiments/mask_readout_mechanism_v2_20261012/MECHANISM_AND_DEFAULTS.md',
    'shared/evaluation/native_output_bridge.py','shared/evaluation/native_source_pins.json',
    'shared/evaluation/posttrain_audit.py','paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json']
OPTIMIZER = dict(lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=.0001,
                 amsgrad=False, foreach=False, maximize=False, capturable=False, differentiable=False, fused=False)


def orders(fit_ids, epochs):
    rng = random.Random(0)
    result = []
    for _ in range(epochs):
        ids = list(fit_ids)
        rng.shuffle(ids)
        result.append(ids)
    return result


def zero_replay(run, model, source, ids):
    import torch
    from native_output_bridge import decode_native, encode_rle
    checks = []
    model.eval()
    with torch.inference_mode():
        for image_id in ids:
            cache, rows = source.load(image_id)
            delta = model(cache, 'eval:0')
            require(torch.count_nonzero(delta).item() == 0, 'nonzero initialized residual')
            changed = cache['c0'].to(delta) + delta
            decoded = decode_native(cache, changed)
            require(all(encode_rle(decoded[i]) == row['segmentation'] for i,row in enumerate(rows)), 'adapter zero-delta native RLE mismatch')
            checks.append({'image_id':image_id,'rows':len(rows),'exact_zero_delta_RLE':True})
            run.check_budget()
    return checks


def train(run, model, source, coco, image_orders, optimizer, whitelist):
    import torch
    from adapter_model import loss_grids, vector_grouped_loss
    from adapter_data import training_join, projected_targets
    model.train()
    started = time.monotonic()
    steps, image_count, no_matches, empty_rois = 0, 0, 0, 0
    joins = {}
    gradient_names = set()
    with (run.path/'training.jsonl').open('x',encoding='utf8') as log:
        for epoch, sequence in enumerate(image_orders):
            epoch_losses = []
            for offset in range(0,len(sequence),4):
                group = sequence[offset:offset+4]
                optimizer.zero_grad(set_to_none=True)
                for image_id in group:
                    run.check_budget()
                    cache, rows = source.load(image_id)
                    if image_id not in joins:
                        joins[image_id] = training_join(rows,coco.imgToAnns.get(image_id,[]))
                    targets = projected_targets(coco,cache,joins[image_id])
                    delta = model(cache,'train:'+str(epoch))
                    logits,baseline,roi = loss_grids(cache,cache['c0'].to(delta)+delta)
                    loss, accounting = vector_grouped_loss(logits,baseline,roi,targets)
                    require(torch.isfinite(loss).item(), 'non-finite training loss')
                    (loss / len(group)).backward()
                    image_count += 1
                    no_matches += int(accounting['matched_supervised_instances']==0)
                    empty_rois += len(accounting['empty_roi_rows'])
                    row={'epoch':epoch,'image_id':image_id,'loss':float(loss.detach()),**accounting}
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    epoch_losses.append(row['loss'])
                    del cache,rows,targets,delta,logits,baseline,roi,loss
                for name,parameter in model.named_parameters():
                    if parameter.grad is not None:
                        require(torch.isfinite(parameter.grad).all().item(),'non-finite adapter gradient')
                        if torch.count_nonzero(parameter.grad).item():
                            gradient_names.add(name)
                optimizer.step()
                steps += 1
                if steps % 25 == 0 or offset+len(group)==len(sequence):
                    log.flush()
                    run.update(phase='adapter_training',arm=model.arm,epoch=epoch+1,optimizer_steps=steps,images_seen=image_count)
                    print('TRAIN',model.arm,'epoch',epoch+1,'steps',steps,'images',image_count,
                          'loss',round(sum(epoch_losses)/len(epoch_losses),6),'seconds',round(time.monotonic()-started,1),flush=True)
            require(len(epoch_losses)==len(sequence),'training image omitted')
    require(set(n for n,p in model.named_parameters() if p.requires_grad)==set(whitelist),'trainable whitelist drift')
    require(gradient_names and any(n.startswith('readout.2') for n in gradient_names),'new module received no gradient')
    require(steps==sum((len(seq)+3)//4 for seq in image_orders),'optimizer step budget mismatch')
    require(image_count==sum(map(len,image_orders)),'image budget mismatch')
    write_new(run.path/'TRAINING_JOIN.json',{'policy':'class_box50_score_order_one_to_one_ordinary_GT',
             'images':{str(i):{'output_rank_to_annotation_id':{str(k):v for k,v in j.items()}} for i,j in joins.items()}})
    return {'optimizer_steps':steps,'images_seen':image_count,'epochs':len(image_orders),
            'images_without_matched_supervision':no_matches,'empty_roi_instances_across_epochs':empty_rois,
            'parameters_with_nonzero_gradient':sorted(gradient_names),'elapsed_training_seconds':time.monotonic()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('smoke','full'),required=True)
    parser.add_argument('--arm',choices=('B','C'),default='B')
    parser.add_argument('--cache-run',type=Path,required=True)
    parser.add_argument('--smoke-run',type=Path)
    parser.add_argument('--paired-run',type=Path)
    args=parser.parse_args()
    require(read_json(args.cache_run/'CONFIG_AUDIT.json')['status']=='passed','cache configuration audit failed')
    require(read_json(args.cache_run/'SUMMARY.json')['exported_images']==2000,'full native fit cache required')
    kind='S2_ADAPTER_SMOKE' if args.scope=='smoke' else 'S2_TRAIN_'+args.arm
    with RegisteredRun(kind,'S2',args.cache_run,CODE_FILES,900 if args.scope=='smoke' else 7200) as run:
        profile, environment=configure_native()
        import torch
        from pycocotools.coco import COCO
        from adapter_model import CoefficientAdapter
        from adapter_data import FrozenCache
        require(sha256(WEIGHT)==profile['model_sha256'],'frozen weight drift')
        split=read_json(args.cache_run/'SPLIT_CONTRACT.json')
        cache_native=read_json(args.cache_run/'trace/NATIVE_ARGS.json')
        base_native=read_json(BASE/'trace/NATIVE_ARGS.json')
        differing={k:{'train_cache':cache_native.get(k),'S0':base_native.get(k)}
                   for k in set(cache_native)|set(base_native) if cache_native.get(k)!=base_native.get(k)}
        require(set(differing)<= {'data','project'},'native train cache configuration differs beyond declared data/output paths')
        write_new(run.path/'NATIVE_BASELINE_CONFIG_COMPARE.json',{'status':'passed','differences':differing,
                  'train_cache_args_sha256':sha256(args.cache_run/'trace/NATIVE_ARGS.json'),
                  'baseline_args_sha256':sha256(BASE/'trace/NATIVE_ARGS.json')})
        fit_ids=split['fit_ids'][:8] if args.scope=='smoke' else split['fit_ids']
        dev_ids=split['dev_ids']
        require(len(split['fit_ids'])==2000 and not set(fit_ids)&set(dev_ids),'data split drift')
        image_orders=orders(fit_ids,1 if args.scope=='smoke' else 3)
        write_new(run.path/'IMAGE_ORDERS.json',image_orders)
        torch.manual_seed(0)
        model=CoefficientAdapter(args.arm).cuda()
        initial_hash=tensor_state_hash(model)
        whitelist=[n for n,p in model.named_parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER)
        actual_optimizer={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
        optimizer_names=[n for n,p in model.named_parameters() if any(p is q for group in optimizer.param_groups for q in group['params'])]
        require(optimizer_names==whitelist,'optimizer includes non-whitelisted parameters')
        contract={'stage':'S2_adapter_training','scope':args.scope,'arm':args.arm,'seed':0,
                  'epochs':len(image_orders),'fit_images':len(fit_ids),'image_microbatch':1,'gradient_accumulation':4,
                  'effective_image_batch':4,'expected_optimizer_steps':sum((len(s)+3)//4 for s in image_orders),
                  'optimizer':actual_optimizer,'amp':False,'dtype':'float32','scheduler':None,'early_stopping':False,
                  'checkpoint_selection':'last_epoch_only','augmentation':None,'model_state_initial_sha256':initial_hash,
                  'parameters':sum(p.numel() for p in model.parameters()),'trainable_whitelist':whitelist,
                  'feature_channels':[256,512,512],'prototype_channels':32,'spatial_size':8,'preservation_weight':.1,
                  'fit_dev_split_sha256':sha256(args.cache_run/'SPLIT_CONTRACT.json'),
                  'image_orders_sha256':sha256(run.path/'IMAGE_ORDERS.json'),
                  'order_algorithm':'Python random.Random(0), persistent generator, shuffle fresh fit list per epoch',
                  'source_cache_run':args.cache_run.name,'source_trace_sha256':sha256(args.cache_run/'trace/TRACE_MANIFEST.json'),
                  'base_model_frozen_mode':'No base model instantiated; read-only native P/c0/F with before/after SHA',
                  'profile_sha256':sha256(ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json'),
                  'damage_sample_budget':1024,'environment':environment}
        run.freeze(contract)
        write_new(run.path/'ENVIRONMENT.json',environment)
        write_new(run.path/'ACTUAL_OPTIMIZER.json',actual_optimizer)
        torch.save({'state_dict':model.state_dict(),'arm':model.arm,'seed':0},run.path/'initial.pt')
        if args.scope=='full':
            require(args.smoke_run is not None and read_json(args.smoke_run/'DECISION.json')['status']=='passed','adapter smoke required')
            require(read_json(args.smoke_run/'CONFIG_AUDIT.json')['status']=='passed','smoke config unverified')
            require(read_json(args.smoke_run/'SOURCE_MANIFEST.json')['files']==run.sources,'training source changed after smoke')
            require(read_json(args.smoke_run/'EVAL_CONTRACT.json')['model_state_initial_sha256']==initial_hash,'initialization changed')
            for prior in (STUDY/'runs').glob('RUN_S2_TRAIN_'+args.arm+'_*/run.json'):
                if prior.parent!=run.path:
                    require(not read_json(prior).get('optimization_started'),'formal arm budget already consumed')
            if args.arm=='C':
                require(args.paired_run is not None,'C must name the completed B run')
                require(read_json(args.paired_run/'DECISION.json')['status']=='passed','B training did not finish')
                require(read_json(args.paired_run/'CONFIG_AUDIT.json')['status']=='passed','B configuration failed audit')
                b_contract=read_json(args.paired_run/'EVAL_CONTRACT.json')
                for key in ('epochs','seed','optimizer','model_state_initial_sha256','fit_dev_split_sha256','image_orders_sha256','source_trace_sha256','parameters','trainable_whitelist'):
                    require(json.dumps(contract[key],sort_keys=True)==json.dumps(b_contract[key],sort_keys=True),'B/C contract drift: '+key)
                require(read_json(args.paired_run/'SOURCE_MANIFEST.json')['files']==run.sources,'B/C source changed')
                write_new(run.path/'PAIRED_B.json',{'run_id':args.paired_run.name,'contract_sha256':sha256(args.paired_run/'EVAL_CONTRACT.json')})
        source=FrozenCache(args.cache_run/'trace',fit_ids)
        coco=COCO(str(args.cache_run/'dataset/training_annotations.json'))
        checks={}
        if args.scope=='smoke':
            with (run.path/'tests.log').open('x',encoding='utf8') as stream:
                result=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('test_adapter_model.py'))],stdout=stream,stderr=subprocess.STDOUT)
            require(result.returncode==0,'adapter behavioral tests failed')
            dev=FrozenCache(BASE/'trace',dev_ids[:8])
            checks['B_zero_delta']=zero_replay(run,model,dev,dev_ids[:8])
            torch.manual_seed(0)
            control=CoefficientAdapter('C').cuda()
            require(tensor_state_hash(control)==initial_hash,'B/C initialization differs')
            checks['C_zero_delta']=zero_replay(run,control,dev,dev_ids[:8])
            checks['train_zero_delta']=zero_replay(run,model,source,fit_ids)
            dev.verify()
            del dev,control
            write_new(run.path/'ENGINEERING_CONTROLS.json',checks)
        run.update(optimization_started=True,phase='adapter_training',arm=args.arm)
        trained=train(run,model,source,coco,image_orders,optimizer,whitelist)
        require(tensor_state_hash(model)!=initial_hash,'adapter weights did not update')
        require(all(torch.isfinite(p).all().item() for p in model.parameters()),'non-finite trained parameters')
        torch.save({'state_dict':model.state_dict(),'arm':args.arm,'seed':0,'contract_sha256':sha256(run.path/'EVAL_CONTRACT.json'),
                    'optimizer_state_dict':optimizer.state_dict(),'completed_epochs':len(image_orders)},run.path/'last.pt')
        source.verify()
        require(sha256(WEIGHT)==profile['model_sha256'],'base weight bytes changed')
        require({k:v for k,v in optimizer.param_groups[0].items() if k!='params'}==actual_optimizer,'optimizer hyperparameters changed')
        from native_output_bridge import native_environment
        after=native_environment()
        require(all(after[k]==v for k,v in profile['runtime_policy'].items()),'numerical runtime changed')
        write_new(run.path/'CONFIG_AUDIT.json',{'status':'passed','scope':'completed adapter training',
                  'checks':{'frozen_source_and_package_versions':True,'effective_optimizer_unchanged':True,
                            'effective_batch4_and_exact_steps':trained['optimizer_steps']==contract['expected_optimizer_steps'],
                            'seed_and_initial_state':True,'frozen_native_cache_bytes_unchanged':True,'base_weight_bytes_unchanged':True,
                            'only_new_parameters_in_optimizer':True,'finite_loss_gradients_parameters':True,'runtime_policy_unchanged':True,
                            'same_B_C_training_contract':True if args.arm=='C' else 'B reference; C checked on its run'},
                  'actual_optimizer':actual_optimizer,'actual_steps':trained['optimizer_steps'],'actual_images_seen':trained['images_seen'],
                  'training_contract_sha256':sha256(run.path/'EVAL_CONTRACT.json'),'checkpoint_sha256':sha256(run.path/'last.pt'),
                  'intended_changes_from_native_baseline':['new residual module and declared training-only optimizer; frozen native inputs unchanged']})
        run.check_budget()
        run.finish({**trained,'arm':args.arm,'checkpoint_sha256':sha256(run.path/'last.pt'),
                    'initial_state_sha256':initial_hash,'final_state_sha256':tensor_state_hash(model)},
                   scientific='engineering_only' if args.scope=='smoke' else 'trained_not_evaluated',
                   next_action='Formal B then C at frozen budget' if args.scope=='smoke' else 'Finish paired arm, then freeze checkpoint and evaluate S2')
        print('TRAIN_COMPLETE',run.run_id,'steps',trained['optimizer_steps'],flush=True)


if __name__=='__main__':
    main()
