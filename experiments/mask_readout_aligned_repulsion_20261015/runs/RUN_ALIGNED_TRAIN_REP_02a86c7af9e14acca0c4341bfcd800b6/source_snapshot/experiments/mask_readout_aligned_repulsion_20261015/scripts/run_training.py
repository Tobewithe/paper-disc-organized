"""One frozen E4 training budget, original order/GT join/optimizer and BCE loop."""
import argparse
import json
import shutil
import subprocess
from common import *
from adapter_model import CoefficientAdapter
from aligned_loss import train,REPULSION_WEIGHT


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('smoke','full'));p.add_argument('--arm',choices=('R','REP'),default='REP');p.add_argument('--smoke-run',default='');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.preflight:
        expected,_,_,env=inspect();print(json.dumps(dict(status='passed',inputs=len(expected['files']),environment=env)));return
    require(a.scope,'scope required')
    with ExperimentRun(('SMOKE_' if a.scope=='smoke' else 'TRAIN_')+a.arm,a.scope,CACHE_RUN) as run:
        expected,_,_,environment=inspect();import torch
        from adapter_data import FrozenCache
        from pycocotools.coco import COCO
        full=a.scope=='full';split=read_json(CACHE_RUN/'SPLIT_CONTRACT.json');fit_ids=split['fit_ids'] if full else split['fit_ids'][:8]
        require(len(split['fit_ids'])==2000 and len(split['dev_ids'])==512 and not set(split['fit_ids'])&set(split['dev_ids']),'split differs')
        orders=read_json(B_RUN/'IMAGE_ORDERS.json') if full else original_training.orders(fit_ids,1)
        write_new(run.path/'IMAGE_ORDERS.json',orders)
        torch.manual_seed(0);model=CoefficientAdapter('B').cuda();model.load_state_dict(torch.load(B_RUN/'initial.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True);initial_hash=tensor_state_hash(model)
        whitelist=[n for n,p in model.named_parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(model.parameters(),**original_training.OPTIMIZER)
        actual_optimizer={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
        old=read_json(B_RUN/'EVAL_CONTRACT.json')
        require(json.dumps(actual_optimizer,sort_keys=True)==json.dumps(old['optimizer'],sort_keys=True),'optimizer changed')
        require(not optimizer.state,'optimizer must be fresh')
        contract=dict(old,scope=a.scope,epochs=len(orders),fit_images=len(fit_ids),expected_optimizer_steps=1500 if full else 2,
            image_orders_sha256=sha256(run.path/'IMAGE_ORDERS.json'),model_state_initial_sha256=initial_hash,parameters=167872,
            trainable_whitelist=whitelist,inputs_sha256=expected,environment=environment,method_id='E4-ALIGNED-v1',experiment_arm=a.arm,repulsion_weight=REPULSION_WEIGHT[a.arm],
            initialization_base_sha256=sha256(B_RUN/'initial.pt'),changes=['native-input-resolution GT BCE','fixed GT-excluded repulsion term' if a.arm=='REP' else 'zero repulsion control'],
            gt_loss='native-input GT BCE + .1 proto preservation BCE + declared repulsion',wall_seconds=run.wall_seconds)
        if full:
            smoke=STUDY/'runs'/a.smoke_run;require(read_json(smoke/'run.json')['execution_status']=='completed','smoke required')
            require(read_json(smoke/'EVAL_CONTRACT.json')['model_state_initial_sha256']==initial_hash,'initialization changed after smoke')
            require(read_json(smoke/'EVAL_CONTRACT.json')['experiment_arm']==a.arm,'wrong smoke arm')
            for prior in (STUDY/'runs').glob('RUN_ALIGNED_TRAIN_'+a.arm+'_*/run.json'):
                if prior.parent!=run.path:require(not read_json(prior).get('optimization_started'),'formal budget already consumed')
        run.freeze(contract)
        if full:require(read_json(smoke/'SOURCE_MANIFEST.json')['files']==run.sources,'sources changed after smoke')
        allowed={'scope','epochs','fit_images','expected_optimizer_steps','image_orders_sha256','model_state_initial_sha256','parameters','trainable_whitelist'}
        require(all(contract[k]==v for k,v in old.items() if k not in allowed),'undeclared training config changed')
        write_new(run.path/'ORIGINAL_CONFIG_COMPARE.json',dict(status='passed',changes=contract['changes'],changed_original_fields=sorted(k for k in old if contract[k]!=old[k])))
        torch.save(dict(state_dict=model.state_dict(),arm='B',seed=0),run.path/'initial.pt')
        write_new(run.path/'ACTUAL_OPTIMIZER.json',actual_optimizer);write_new(run.path/'ENVIRONMENT.json',environment)
        source=FrozenCache(CACHE_RUN/'trace',fit_ids);coco=COCO(str(TRAIN_GT))
        if not full:
            with (run.path/'tests.log').open('x',encoding='utf8') as f:
                result=subprocess.run([sys.executable,'-B',str(STUDY/'scripts/test_aligned.py')],stdout=f,stderr=subprocess.STDOUT)
            require(result.returncode==0,'synthetic control failed')
            dev=FrozenCache(BASE/'trace',split['dev_ids'][:8])
            checks=dict(fit=original_training.zero_replay(run,model,source,fit_ids),dev=original_training.zero_replay(run,model,dev,split['dev_ids'][:8]))
            write_new(run.path/'ENGINEERING_CONTROLS.json',checks);dev.verify();del dev
        torch.cuda.reset_peak_memory_stats();run.update(optimization_started=True,phase='adapter_training',new_training_steps=0)
        trained=train(run,model,source,coco,orders,optimizer,whitelist,a.arm)
        require(tensor_state_hash(model)!=initial_hash,'model did not learn');require(trained['optimizer_steps']==contract['expected_optimizer_steps'],'step count differs')
        join=read_json(run.path/'TRAINING_JOIN.json')['images'];old_join=read_json(B_RUN/'TRAINING_JOIN.json')['images']
        require(join=={str(i):old_join[str(i)] for i in fit_ids},'training pairing differs')
        if full:require(sha256(run.path/'TRAINING_JOIN.json')==sha256(B_RUN/'TRAINING_JOIN.json'),'training join bytes differ')
        torch.save(dict(state_dict=model.state_dict(),arm='B',seed=0,contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),
            optimizer_state_dict=optimizer.state_dict(),completed_epochs=len(orders)),run.path/'last.pt')
        source.verify();again=inspect();require(again[0]==expected and again[3]==environment,'post-run config changed')
        require({k:v for k,v in optimizer.param_groups[0].items() if k!='params'}==actual_optimizer,'optimizer changed')
        import psutil
        mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss)
        require(peak<=4*1024**3 and torch.cuda.max_memory_reserved()<=6.5*1024**3,'memory budget exceeded');run.check_budget()
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',actual_steps=trained['optimizer_steps'],actual_images_seen=trained['images_seen'],
            checkpoint_sha256=sha256(run.path/'last.pt'),contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),peak_rss_bytes=peak,
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),checks=dict(original_shared_initial_weights=True,only_declared_new_parameters=True,
            unchanged_GT_join_orders_optimizer_and_preservation=True,declared_aligned_loss_and_repulsion=True,frozen_inputs_sources_environment_unchanged=True,finite_gradients=True)))
        run.update(new_training_steps=trained['optimizer_steps'],experiment_arm=a.arm)
        run.finish(dict(**trained,arm='B',checkpoint_sha256=sha256(run.path/'last.pt'),initial_state_sha256=initial_hash,final_state_sha256=tensor_state_hash(model)),
            scientific='trained_not_evaluated' if full else 'engineering_only',next_action='Independent checkpoint audit then same-population S2 evaluation' if full else 'One full E4 arm from original initialization')
        print('TRAIN_COMPLETE',run.run_id,trained['optimizer_steps'],flush=True)


if __name__=='__main__':main()
