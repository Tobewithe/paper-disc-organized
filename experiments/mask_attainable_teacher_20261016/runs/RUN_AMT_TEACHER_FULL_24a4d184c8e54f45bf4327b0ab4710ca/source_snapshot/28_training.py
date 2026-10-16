"""Original loop/init/architecture/budget, only train-only selected targets substituted."""
import argparse
import copy
import json
from common import *
from targets import TeacherTargets
from soft_iou_loss import LOSS_SPEC

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope',choices=['smoke','full'],required=True)
    p.add_argument('--teacher-run',type=Path,required=True)
    p.add_argument('--smoke-run',type=Path)
    args=p.parse_args()
    full=args.scope=='full'
    teacher=args.teacher_run.resolve()
    require(teacher.is_relative_to((STUDY/'runs').resolve()),'teacher from another Study')
    for file in ('DECISION.json','CONFIG_AUDIT.json','INDEPENDENT_TEACHER.json'):
        require(read_json(teacher/file)['status']=='passed','teacher gate/audit unverified')
    require(read_json(teacher/'TEACHER_MANIFEST.json')['scope']==args.scope,'teacher/student scope mismatch')
    with Run('TRAIN' if full else 'TRAIN_SMOKE',teacher,7200 if full else 900) as run:
        data=inspect_inputs()
        import torch
        import adapter_model
        import adapter_data
        from pycocotools.coco import COCO
        original_training=previous.original_training
        ids=data['split']['fit_ids'] if full else data['split']['fit_ids'][:8]
        image_orders=read_json(INITIAL/'IMAGE_ORDERS.json') if full else original_training.orders(ids,1)
        if full:
            shutil.copyfile(INITIAL/'IMAGE_ORDERS.json',run.path/'IMAGE_ORDERS.json')
        else:
            write_new(run.path/'IMAGE_ORDERS.json',image_orders)
        torch.manual_seed(0)
        model=adapter_model.CoefficientAdapter('B').cuda()
        saved=torch.load(INITIAL/'initial.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict'],strict=True)
        require(saved['arm']=='B' and saved['seed']==0,'initial arm/seed differs')
        del saved
        initial=tensor_state_hash(model)
        original_contract=data['original_contract']
        require(initial==original_contract['model_state_initial_sha256'],'original initialization changed')
        whitelist=[n for n,p in model.named_parameters() if p.requires_grad]
        require(whitelist==original_contract['trainable_whitelist'],'trainable whitelist differs')
        optimizer=torch.optim.AdamW(model.parameters(),**original_training.OPTIMIZER)
        require(not optimizer.state,'optimizer resumed')
        actual={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
        require(json.dumps(actual,sort_keys=True)==json.dumps(original_contract['optimizer'],sort_keys=True),'optimizer drift')
        require([id(p) for g in optimizer.param_groups for p in g['params']]==[id(p) for p in model.parameters()],'optimizer parameter identities differ')
        contract=copy.deepcopy(original_contract)
        contract.update(scope=args.scope,epochs=3 if full else 1,fit_images=len(ids),
            expected_optimizer_steps=1500 if full else 2,image_orders_sha256=sha256(run.path/'IMAGE_ORDERS.json'),
            gt_loss=LOSS_SPEC,target_policy=SPEC,teacher_run_id=teacher.name,
            teacher_manifest_sha256=sha256(teacher/'TEACHER_MANIFEST.json'),
            independent_teacher_sha256=sha256(teacher/'INDEPENDENT_TEACHER.json'),
            initialization_file_sha256=sha256(INITIAL/'initial.pt'),inputs_sha256=data['inputs_sha256'],
            external_evaluator_sources=data['external_evaluator_sources'],reference_training_run=CONTROL.name,
            single_scientific_variable='GT_target_to_conditional_attainable_hard_mask_target',
            original_training_loop_sha256=sha256(Path(original_training.__file__)),wall_seconds=run.wall_seconds)
        if full:
            require(args.smoke_run is not None,'student smoke missing')
            for file in ('DECISION.json','CONFIG_AUDIT.json'):
                require(read_json(args.smoke_run/file)['status']=='passed','student smoke failed')
            require(read_json(args.smoke_run/'EVAL_CONTRACT.json')['model_state_initial_sha256']==initial,'smoke initial differs')
            for path in (STUDY/'runs').glob('RUN_AMT_TRAIN_*/run.json'):
                if path.parent!=run.path and read_json(path).get('kind')=='AMT_TRAIN':
                    require(not read_json(path).get('optimization_started'),'formal training budget consumed')
        run.freeze(contract)
        require(read_json(teacher/'SOURCE_MANIFEST.json')['files']==run.sources,'teacher/training frozen code differs')
        if full:
            require(read_json(args.smoke_run/'SOURCE_MANIFEST.json')['files']==run.sources,'smoke/full source differs')
        allowed=set() if full else {'scope','epochs','fit_images','expected_optimizer_steps','image_orders_sha256'}
        equality={k:contract[k]==v for k,v in original_contract.items()}
        require(all(v or k in allowed for k,v in equality.items()),'original configuration changed')
        reference=read_json(CONTROL/'EVAL_CONTRACT.json')
        require(contract['gt_loss']==reference['gt_loss'],'soft-IoU formula changed')
        write_new(run.path/'ORIGINAL_CONFIG_COMPARE.json',dict(status='passed',original_field_equality=equality,
            smoke_budget_exceptions=sorted(allowed),soft_IoU_formula_identical=True,
            changed_training_variable=contract['single_scientific_variable'],native_argument_differences=data['native_argument_differences']))
        write_new(run.path/'ENVIRONMENT.json',data['environment'])
        write_new(run.path/'ACTUAL_OPTIMIZER.json',actual)
        shutil.copyfile(INITIAL/'initial.pt',run.path/'initial.pt')
        source=adapter_data.FrozenCache(CACHE/'trace',ids)
        coco=COCO(str(TRAIN_GT))
        if not full:
            controls=original_training.zero_replay(run,model,source,ids)
            write_new(run.path/'ENGINEERING_CONTROLS.json',dict(zero_delta=controls))
        original_targets=adapter_data.projected_targets
        original_loss=adapter_model.vector_grouped_loss
        provider=TeacherTargets(teacher,original_targets)
        adapter_data.projected_targets=provider
        adapter_model.vector_grouped_loss=provider.loss
        try:
            run.update(optimization_started=True,phase='adapter_training')
            trained=original_training.train(run,model,source,coco,image_orders,optimizer,whitelist)
        finally:
            adapter_data.projected_targets=original_targets
            adapter_model.vector_grouped_loss=original_loss
        require(tensor_state_hash(model)!=initial,'student did not update')
        require(all(torch.isfinite(p).all().item() for p in model.parameters()),'nonfinite student')
        require([r['image_id'] for r in provider.visits]==[i for seq in image_orders for i in seq],'actual target visits differ')
        require(provider.pending is None,'last target not consumed')
        old_join=read_json(INITIAL/'TRAINING_JOIN.json')['images']
        require(read_json(run.path/'TRAINING_JOIN.json')['images']=={str(i):old_join[str(i)] for i in ids},'actual joins changed')
        if full:
            require(sha256(run.path/'TRAINING_JOIN.json')==sha256(INITIAL/'TRAINING_JOIN.json'),'formal join bytes differ')
        require({k:v for k,v in optimizer.param_groups[0].items() if k!='params'}==actual,'optimizer changed during training')
        torch.save(dict(state_dict=model.state_dict(),arm='B',seed=0,contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),
            optimizer_state_dict=optimizer.state_dict(),completed_epochs=len(image_orders)),run.path/'last.pt')
        write_new(run.path/'TARGET_USAGE.json',dict(visits=provider.visits,
            teacher_replacements_across_epochs=sum(x['teacher_replaced'] for x in provider.visits)))
        source.verify()
        verify_after(data)
        require(sha256(teacher/'TEACHER_MANIFEST.json')==contract['teacher_manifest_sha256'],'teacher manifest changed')
        for rec in provider.records.values():
            require(sha256(teacher/rec['path'])==rec['sha256'],'teacher package changed during training')
        require(trained['optimizer_steps']==contract['expected_optimizer_steps'],'step budget changed')
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',original_config_equal_except_smoke=all(v or k in allowed for k,v in equality.items()),
            original_loop_and_soft_IoU_unchanged=True,only_target_generation_changed=True,same_training_join=True,
            original_initialization_fresh_optimizer=True,frozen_cache_and_teacher_and_runtime_unchanged=True,
            only_adapter_parameters_trainable=True,actual_steps=trained['optimizer_steps'],actual_image_visits=trained['images_seen'],
            teacher_replacements_across_epochs=sum(x['teacher_replaced'] for x in provider.visits),checkpoint_sha256=sha256(run.path/'last.pt')))
        run.check_budget()
        run.finish(dict(**trained,checkpoint_sha256=sha256(run.path/'last.pt'),initial_state_sha256=initial,
            final_state_sha256=tensor_state_hash(model),teacher_replacements_across_epochs=sum(x['teacher_replaced'] for x in provider.visits)),
            scientific='trained_not_evaluated' if full else 'engineering_only',
            next_action='Independent serialized training audit, then frozen S2 comparison; no automatic S3' if full else 'Full teacher then fresh original-init training')
        print('AMT_TRAIN_COMPLETE',run.run_id,trained['optimizer_steps'],flush=True)

if __name__=='__main__':
    main()
