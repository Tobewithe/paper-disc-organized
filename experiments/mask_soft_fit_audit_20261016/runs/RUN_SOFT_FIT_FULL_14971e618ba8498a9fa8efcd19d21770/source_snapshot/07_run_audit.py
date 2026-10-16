"""Evaluate the existing final soft-IoU model on its original fit2000, without training."""
from common import *
import argparse
from collections import Counter
import contextlib
import gc
import time
from audit_stats import area_group,iou_group,build_summary
from posttrain_audit import population,compare_outputs
from run_fit_diagnostic import baseline_to_global,fixed_effects
from s1_geometry import iter_prediction_array,full_rle_iou

def model_from_checkpoint():
    import torch
    from adapter_model import CoefficientAdapter
    saved=torch.load(TRAIN/'last.pt',map_location='cpu',weights_only=True)
    require(saved['arm']=='B' and saved['completed_epochs']==3,'wrong checkpoint identity')
    require(saved['contract_sha256']==sha256(TRAIN/'EVAL_CONTRACT.json'),'checkpoint contract mismatch')
    require({int(v['step']) for v in saved['optimizer_state_dict']['state'].values()}=={1500},'wrong optimizer history')
    model=CoefficientAdapter('B').cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'],strict=True)
    require(sum(p.numel() for p in model.parameters())==167872,'architecture changed')
    require(tensor_state_hash(model)==read_json(TRAIN/'SUMMARY.json')['final_state_sha256'],'wrong state')
    return model

def export_fit(run,source,coco,baseline):
    import torch
    from pycocotools import mask as masks
    from adapter_data import training_join,projected_targets
    from adapter_model import loss_grids
    from soft_iou_loss import soft_iou_grouped_loss
    from native_output_bridge import decode_native,encode_rle
    from area_gate import mix_row
    model=model_from_checkpoint()
    before_hash=tensor_state_hash(model)
    saved_join=read_json(TRAIN/'TRAINING_JOIN.json')['images']
    by_gt={(r['image_id'],r['annotation_id']):r for r in baseline}
    coverage,losses,indices,checks=[],[],[],[]
    areas={}
    started=time.monotonic()
    with (run.path/'U_predictions.json').open('x',encoding='utf8') as uout, (run.path/'G_predictions.json').open('x',encoding='utf8') as gout, (run.path/'TRAINING_PAIRS.jsonl').open('x',encoding='utf8') as pairs, torch.inference_mode():
        uout.write('['); gout.write('[')
        comma=False
        for number,image_id in enumerate(source.records,1):
            run.check_budget()
            cache,original=source.load(image_id)
            c0=cache['c0'].cuda()
            decoded=decode_native(cache,c0)
            require(all(encode_rle(decoded[i])==r['segmentation'] for i,r in enumerate(original)),'zero-delta native replay differs')
            del decoded
            join=training_join(original,coco.imgToAnns.get(image_id,[]))
            expected={int(k):v for k,v in saved_join[str(image_id)]['output_rank_to_annotation_id'].items()}
            require(join==expected,'training supervision join drift')
            target=projected_targets(coco,cache,join)
            delta=model(cache,'eval:0')
            require(torch.isfinite(delta).all().item() and not delta.requires_grad,'invalid frozen residual')
            coeff=c0+delta
            logits,native,roi=loss_grids(cache,coeff)
            loss0,stats0=soft_iou_grouped_loss(native,native,roi,target)
            loss1,stats1=soft_iou_grouped_loss(logits,native,roi,target)
            for k in ('matched_supervised_instances','preserved_instances','empty_roi_rows','no_output_image','empty_GT_in_ROI_instances'):
                require(stats0[k]==stats1[k],'loss denominator drift')
            losses.append(dict(image_id=image_id,zero=dict(total_loss=float(loss0),**stats0),U=dict(total_loss=float(loss1),**stats1)))
            valid_roi=(roi.sum((1,2))>0).cpu().tolist()
            valid=[k for k in join if valid_roi[k]]
            del target,logits,native,roi,loss0,loss1
            decoded=decode_native(cache,coeff)
            urows=[{**r,'segmentation':encode_rle(decoded[i])} for i,r in enumerate(original)]
            del decoded
            mixed=[mix_row(a,b) for a,b in zip(original,urows)]
            grows=[v[0] for v in mixed]
            gate=[v[1]['apply_delta'] for v in mixed]
            for rank,index in enumerate(cache['prediction_indices']):
                areas[index]=mixed[rank][1]['native_mask_area']
            compare_outputs(original,urows); compare_outputs(original,grows)
            coverage.append(dict(image_id=image_id,outputs=len(original),gate_outputs=sum(gate),joined=len(join),gate_joined=sum(gate[k] for k in join),valid_supervised=len(valid),gate_valid_supervised=sum(gate[k] for k in valid),preserved=sum(valid_roi),gate_preserved=sum(g and v for g,v in zip(gate,valid_roi)),gate_supervised_categories=sorted({original[k]['category_id'] for k in valid if gate[k]})))
            require(len(valid)==stats0['matched_supervised_instances'],'supervision denominator mismatch')
            for rank,aid in join.items():
                gt_rle=coco.annToRLE(coco.anns[aid])
                oid=source.manifest['predictions_sha256']+':'+str(cache['prediction_indices'][rank])
                row=dict(image_id=image_id,annotation_id=aid,training_output_id=oid,category_id=original[rank]['category_id'],valid_supervision=valid_roi[rank],native_area=mixed[rank][1]['native_mask_area'],gate_enabled=gate[rank],same_as_bbox75_pair=oid==by_gt[(image_id,aid)]['bbox75_match_output_id'],baseline_iou=full_rle_iou(original[rank]['segmentation'],gt_rle),U_iou=full_rle_iou(urows[rank]['segmentation'],gt_rle))
                pairs.write(json.dumps(row,allow_nan=False)+'\n')
            for a,b in zip(urows,grows):
                if comma: uout.write(','); gout.write(',')
                uout.write(json.dumps(a,separators=(',',':'),allow_nan=False))
                gout.write(json.dumps(b,separators=(',',':'),allow_nan=False))
                comma=True
            indices.extend(cache['prediction_indices'])
            checks.append(dict(image_id=image_id,outputs=len(original),zero_delta_RLE_exact=True,training_join_exact=True,nonmask_fields_exact=True))
            if number%50==0 or number==len(source.records):
                uout.flush(); gout.flush(); pairs.flush()
                run.update(phase='fit_checkpoint_export',completed_images=number,total_images=len(source.records))
                print('FIT_EXPORT',number,'/',len(source.records),'seconds',round(time.monotonic()-started,1),flush=True)
            del cache,original,c0,delta,coeff,urows,grows,mixed
        uout.write(']\n'); gout.write(']\n')
    require(indices==sorted(indices) and len(indices)==len(set(indices)),'row ordering changed')
    require(tensor_state_hash(model)==before_hash,'checkpoint state mutated')
    require(not model.training and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'gradient or training mode present')
    jsonl(run.path/'COVERAGE_BY_IMAGE.jsonl',coverage)
    jsonl(run.path/'LOSS_BY_IMAGE.jsonl',losses)
    jsonl(run.path/'ZERO_DELTA_AND_JOIN_CHECKS.jsonl',checks)
    write_new(run.path/'ROW_MAPPING.json',dict(source_prediction_sha256=source.manifest['predictions_sha256'],original_global_indices=indices))
    write_new(run.path/'FROZEN_MODEL_STATE.json',dict(before=before_hash,after=tensor_state_hash(model),requires_grad=False,gradients_present=False,model_eval=True,new_optimizer_steps=0))
    del model
    torch.cuda.empty_cache(); gc.collect()
    return indices,areas,coverage,losses

def paired_table(baseline,u,g,indices,source_sha,areas):
    ue=fixed_effects(baseline,u,indices,source_sha)
    ge=fixed_effects(baseline,g,indices,source_sha)
    original={(r['image_id'],r['annotation_id']):r for r in baseline}
    result=[]
    require(len(ue)==len(ge),'U/G pair domain mismatch')
    for a,b in zip(ue,ge):
        require((a['image_id'],a['annotation_id'],a['baseline_output_detection_id'])==(b['image_id'],b['annotation_id'],b['baseline_output_detection_id']),'fixed pair identity mismatch')
        row=original[(a['image_id'],a['annotation_id'])]
        index=int(a['baseline_output_detection_id'].rsplit(':',1)[1])
        ar=areas[index]
        result.append(dict(image_id=a['image_id'],annotation_id=a['annotation_id'],category_id=row['gt_category_id'],population=a['population'],baseline_output_detection_id=a['baseline_output_detection_id'],baseline_iou=a['baseline_iou'],U_iou=a['method_iou'],G_iou=b['method_iou'],native_area=ar,native_area_group=area_group(ar),native_iou_group=iou_group(a['baseline_iou']),gate_enabled=ar>=9216))
    return result

def task_table(baseline,u,g,areas):
    methods=[{(r['image_id'],r['annotation_id']):r for r in rows} for rows in (u,g)]
    require(all(set(m)=={(r['image_id'],r['annotation_id']) for r in baseline} for m in methods),'task GT coverage mismatch')
    result=[]
    for row in baseline:
        key=(row['image_id'],row['annotation_id'])
        a,b=[m[key] for m in methods]
        require(row['is_ignored']==a['is_ignored']==b['is_ignored'],'ignore changed')
        if row['is_ignored']: continue
        success=row['task_mask75']=='success'
        index=int(row['segm75_match_output_id'].rsplit(':',1)[1]) if success else None
        ar=areas[index] if success else None
        result.append(dict(image_id=row['image_id'],annotation_id=row['annotation_id'],category_id=row['gt_category_id'],baseline_success=success,U_success=a['task_mask75']=='success',G_success=b['task_mask75']=='success',success_native_output_id=row['segm75_match_output_id'],success_native_area=ar,success_native_area_group=area_group(ar) if ar is not None else 'not_baseline_success'))
    return result

def dev_reuse(run,ids):
    from pycocotools import mask as masks
    keep=set(ids)
    baseline=[r for r in read_jsonl(DEV/'BASELINE_SUBSET_GT_LEDGER.jsonl') if r['image_id'] in keep]
    u=[r for r in read_jsonl(DEV/'B_GT_LEDGER.jsonl') if r['image_id'] in keep]
    g=[r for r in read_jsonl(GATE/'MIXED_GT_LEDGER.jsonl') if r['image_id'] in keep]
    mapping=read_json(DEV/'B_ROW_MAPPING.json')
    require(mapping==read_json(GATE/'ROW_MAPPING.json'),'dev prediction index mappings differ')
    native_path=BASE/'trace/native/predictions.json'
    areas={i:int(masks.area(r['segmentation'])) for i,r in enumerate(iter_prediction_array(native_path)) if r['image_id'] in keep}
    rows=paired_table(baseline,u,g,mapping['original_global_indices'],mapping['source_prediction_sha256'],areas)
    tasks=task_table(baseline,u,g,areas)
    for name,data in (('DEV_FIXED_PAIRS.jsonl',rows),('DEV_TASKS.jsonl',tasks),('DEV_BASELINE_GT_LEDGER.jsonl',baseline),('DEV_U_GT_LEDGER.jsonl',u),('DEV_G_GT_LEDGER.jsonl',g)):
        jsonl(run.path/name,data)
    return rows,tasks

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scope',choices=('smoke','full'),required=True)
    ap.add_argument('--smoke-run',type=Path)
    args=ap.parse_args()
    with AuditRun(args.scope) as run:
        data=inspect_inputs()
        ids=data['split']['fit_ids'][:8] if args.scope=='smoke' else data['split']['fit_ids']
        contract=dict(scope=args.scope,checkpoint_sha256=CKPT_SHA,checkpoint_run_id=TRAIN.name,fit_ids=ids,dev_ids=data['split']['dev_ids'],input_files=data['files'],environment=data['environment'],external_evaluator_sources=data['external_sources'],native_argument_differences=data['native_argument_differences'],native_profile_sha256=sha256(PROFILE),new_training_steps=0,base_model_forward_calls=0,oracle_optimization_calls=0,phase='eval:0',gate_area_min=9216,decoder='original decode_native full coefficient matrix',preprocessing='original batch16 rect per-image cached geometry; no new base-model forward',dtype='float32',AMP=False,method_arms=['U','G'],fit_thresholds=dict(target_mean_min=.005,target_ci_lower_strict_min=0,all_bbox75_min=-.001,success_damage_census_max=.01),bootstrap=dict(n=2000,seed=1729,CI=[.025,.975]),wall_seconds=run.wall_seconds,max_artifact_bytes=8*1024**3,scientific_scope='observational exposed fit/dev diagnostic, not a causal sample-size experiment')
        run.freeze(contract)
        if args.scope=='full':
            require(args.smoke_run is not None,'smoke required')
            require(read_json(args.smoke_run/'run.json')['execution_status']=='completed','smoke incomplete')
            require(read_json(args.smoke_run/'CONFIG_AUDIT.json')['status']=='passed','smoke config failed')
            require(read_json(args.smoke_run/'SOURCE_MANIFEST.json')['files']==run.sources,'source changed after smoke')
            old=read_json(args.smoke_run/'EVAL_CONTRACT.json')
            require(old['input_files']==data['files'] and old['environment']==data['environment'],'smoke input drift')
        study=read_json(STUDY/'study.json')
        study.update(execution_status='running',active_run_id=run.run_id)
        save(STUDY/'study.json',study)
        write_new(run.path/'INPUT_AUDIT.json',dict(status='passed',matched_original_sources=data['matched_original_sources'],verified_source_artifacts=data['verified_source_artifacts'],input_files=data['files']))
        from adapter_data import FrozenCache
        from pycocotools.coco import COCO
        from build_native_ledger import evaluate_ledger
        source=FrozenCache(CACHE/'trace',ids)
        indices=sorted(source.predictions)
        oldmap=read_json(FIT/'ROW_MAPPING.json')
        require(oldmap['source_prediction_sha256']==source.manifest['predictions_sha256'],'training baseline source changed')
        if args.scope=='full': require(oldmap['original_global_indices']==indices,'training baseline output set changed')
        baseline=[r for r in read_jsonl(FIT/'BASELINE_GT_LEDGER.jsonl') if r['image_id'] in set(ids)]
        jsonl(run.path/'BASELINE_GT_LEDGER.jsonl',baseline)
        coco=COCO(str(GT))
        expected={(a['image_id'],a['id']) for a in coco.dataset['annotations'] if a['image_id'] in set(ids)}
        require({(r['image_id'],r['annotation_id']) for r in baseline}==expected,'baseline GT domain changed')
        exported,areas,coverage,losses=export_fit(run,source,coco,baseline)
        require(exported==indices,'fit output domain changed')
        methods={}; ap_receipt={}
        for arm in ('U','G'):
            run.update(phase=arm+'_fit_COCO_ledger')
            with (run.path/(arm+'_ledger.log')).open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
                methods[arm],ap_receipt[arm]=evaluate_ledger(GT,read_json(run.path/(arm+'_predictions.json')),ids,sha256(run.path/(arm+'_predictions.json')))
            jsonl(run.path/(arm+'_GT_LEDGER.jsonl'),methods[arm])
            run.check_budget(); gc.collect()
        rows=paired_table(baseline,methods['U'],methods['G'],indices,source.manifest['predictions_sha256'],areas)
        tasks=task_table(baseline,methods['U'],methods['G'],areas)
        jsonl(run.path/'FIT_FIXED_PAIRS.jsonl',rows); jsonl(run.path/'FIT_TASKS.jsonl',tasks)
        run.update(phase='reuse_existing_dev_and_statistics')
        dev,dev_tasks=dev_reuse(run,data['split']['dev_ids'])
        summary=build_summary(rows,dev,tasks,dev_tasks,coverage,losses,args.scope)
        summary['secondary_train_AP']=ap_receipt
        summary['baseline_fit_AP_source']=str(FIT/'SUMMARY.json')
        summary['fit_images']=len(ids)
        summary['dev_images']=512
        summary['zero_delta_exact_outputs']=len(indices)
        for arm,previous in (('U',read_json(DEV/'SUMMARY.json')['B']),('G',read_json(GATE/'SUMMARY.json'))):
            for metric in ('target','all_bbox75'):
                actual=summary['dev'][arm][metric]
                require(abs(actual['mean']-previous[metric]['mean'])<1e-12,'prior dev mean mismatch')
                require(all(abs(a-b)<1e-12 for a,b in zip(actual['ci95'],previous[metric]['ci95'])),'prior dev CI mismatch')
        require(summary['dev']['G']['target']['pairs']==655,'dev target denominator changed')
        if args.scope=='full':
            require(sum(r['population']=='T' for r in rows)>=200,'insufficient fit T')
            require(len(ids)==2000 and len(indices)==218209,'original fit output denominator changed')
            require(summary['training_coverage']['joined']==14276 and summary['training_coverage']['valid_supervised']==14268,'original supervision denominator changed')
        run.update(phase='post_run_configuration_audit')
        source.verify()
        final=inspect_inputs()
        require(final['files']==data['files'] and final['environment']==data['environment'],'input/runtime changed')
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file())<=8*1024**3,'artifact budget exceeded')
        run.check_budget()
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',input_files_unchanged=True,source_freeze_unchanged=True,original_checkpoint_state_unchanged=True,zero_delta_RLE_exact_outputs=len(indices),all_nonmask_output_fields_unchanged=True,training_join_and_loss_denominators_exact=True,full_GT_coverage_and_separate_bbox_segm_matching=True,all_bbox75_identities_unchanged=True,gate_threshold=9216,development_outputs_reused=True,development_original_means_CIs_equal=True,new_training_steps=0,base_model_forward_calls=0,oracle_optimization_calls=0))
        decision=summary['decision']
        run.finish(summary,status='passed',scientific=decision['finding'],next_action=decision['next_action'])
        study.update(execution_status='completed' if args.scope=='full' else 'smoke_passed',active_run_id=None,**{args.scope+'_run_id':run.run_id})
        save(STUDY/'study.json',study)
        print('AUDIT_COMPLETED',run.run_id,json.dumps(decision),flush=True)

if __name__=='__main__':
    main()
