"""Frozen finite amplitude audit. No optimizer, backward pass, or model training."""
from ray_common import *
import argparse,gc,time
from collections import defaultdict


def execute(run):
    import numpy as np
    import torch
    from pycocotools import mask as masks
    from adapter_data import FrozenCache
    from adapter_model import CoefficientAdapter
    from native_output_bridge import decode_native,encode_rle,native_environment
    from native_geometry import native_stages,roi_for,lifted_basis
    from s1_geometry import full_rle_iou
    from ray_geometry import conversion_table,lookup,build_sweep,curve,candidates,predict,binary_iou,choose_actual
    from analysis import summarize
    from verify_ray import verify

    cfg,expected,profile,environment=inspect_inputs();prep=read_json(STUDY/'PREPARATION.json');scope=run.meta['scope']
    require(environment==prep['environment'],'environment drift from preparation')
    panel=prep[scope+'_targets'];ids=sorted({r['image_id'] for r in panel})
    run.freeze(dict(scope=scope,targets=panel,image_ids=ids,environment=environment,native_profile=profile,input_files=expected['files'],config=cfg,
        candidate_policy='same_final_Box75_detection',decoder='pinned_full_N_FP32_native_then_binary_scale_byte',
        checkpoint=dict(run_id=TRAIN.name,sha256=sha256(TRAIN/'last.pt'),arm='B',phase='eval:0'),
        new_training_steps=0,base_model_forward_calls=0,GT_assisted_selection=True,not_deployment_AP=True))
    write_new(run.path/'TARGETS.json',panel)
    control_path=(C_FULL if scope=='full' else C_SMOKE)/'PAIR_RESULTS.jsonl'
    controls={r['output_detection_id']:r for r in read_jsonl(control_path)}
    historical={r['output_detection_id']:r for r in read_jsonl(AUDIT/'PAIRED_MASKS.jsonl')}
    source=FrozenCache(BASE/'trace',ids)
    write_new(run.path/'INPUT_CACHE_RECORDS.json',[source.records[i] for i in ids])
    cp=torch.load(TRAIN/'last.pt',map_location='cpu',weights_only=True)
    require(cp['arm']=='B' and cp['completed_epochs']==3,'checkpoint metadata differs')
    model=CoefficientAdapter('B').cuda().eval();model.load_state_dict(cp['state_dict'],strict=True);model.requires_grad_(False)
    del cp
    state_hash=tensor_state_hash(model)
    previous_delta=torch.load(S3/'DELTA.pt',map_location='cpu',weights_only=True)
    by_image=defaultdict(list)
    for t in panel:by_image[t['image_id']].append(t)
    rows=[];zero_rows=0;delta_exact=0;processing_seconds=0.;native_candidate_calls=0
    (run.path/'sweeps').mkdir()
    with (run.path/'PAIR_RESULTS.jsonl').open('x',encoding='utf8') as stream,torch.inference_mode():
        for image_number,image_id in enumerate(ids):
            resources(run);started=time.monotonic()
            cache,native_rows=source.load(image_id);indices=cache['prediction_indices'];rank_map={v:i for i,v in enumerate(indices)}
            targets=by_image[image_id];ranks=[rank_map[int(t['output_detection_id'].rsplit(':',1)[1])] for t in targets]
            require(len(ranks)==len(set(ranks)),'duplicate target detection')
            first0,final0,z0s=native_stages(cache,cache['c0'],ranks)
            require(torch.equal(final0,decode_native(cache).cpu()),'native stage/bridge mismatch')
            for rank,prediction in enumerate(native_rows):
                require(encode_rle(final0[rank])==prediction['segmentation'],'zero-delta RLE mismatch');zero_rows+=1
            delta=model(cache,'eval:0').cpu()
            require(delta.shape==cache['c0'].shape and torch.isfinite(delta).all(),'invalid existing delta')
            if image_id in previous_delta:
                require(torch.equal(delta,previous_delta[image_id]),'checkpoint delta differs from S3');delta_exact+=1
            transform=conversion_table(cache);basis=lifted_basis(cache)
            cases={}
            for target,rank in zip(targets,ranks,strict=True):
                old=controls[target['output_detection_id']];native=native_rows[rank]
                require(old['native_prediction']==native and old['annotation_id']==target['annotation_id'],'source C identity differs')
                require(old['cache_sha256']==source.records[image_id]['cache_sha256'],'source C cache differs')
                require(old['arms']['C']['solver']['solver_status']=='converged','C source unconverged')
                require(old['c0']==cache['c0'][rank].tolist(),'source c0 differs')
                roi=roi_for(cache,rank).numpy();z0=z0s[rank].numpy().astype(np.float64)
                direction=(basis.reshape(32,-1).T@delta[rank].double()).reshape(cache['native_decode_hw']).numpy()
                gt=masks.decode(old['GT_rle']).astype(bool)
                require(np.array_equal(lookup(first0[rank].numpy(),transform),final0[rank].numpy()),'native conversion lookup mismatch')
                sweep=build_sweep(z0,direction,roi,gt,transform,cfg['ray']['alpha_max'])
                alphas=candidates(sweep,cfg['ray'])
                require(len(alphas)<=cfg['ray']['max_actual_candidates'],'candidate budget exceeded')
                event_path=run.path/'sweeps'/f"{target['annotation_id']}.npz"
                np.savez_compressed(event_path,**sweep)
                sweep_alpha,sweep_iou=curve(sweep)
                record=dict(**target,rank=rank,global_index=indices[rank],cache_sha256=source.records[image_id]['cache_sha256'],
                    raw_id=int(cache['raw_ids'][rank]),native_prediction=native,GT_rle=old['GT_rle'],c0=cache['c0'][rank].tolist(),
                    adapter_delta=delta[rank].tolist(),sweep_file=event_path.relative_to(run.path).as_posix(),sweep_sha256=sha256(event_path),
                    sweep_interval_count=len(sweep_alpha),proxy_interval_best_iou=float(sweep_iou.max()),candidate_alphas=alphas,
                    numerical_valid=True,arms=dict(N=dict(original_rle=native['segmentation'],original_iou=full_rle_iou(native['segmentation'],old['GT_rle']),alpha=0.)))
                cases[rank]=dict(record=record,old=old,z0=z0,direction=direction,roi=roi,gt=gt,alphas=alphas,candidates={})
            del basis

            def check_conversion(first,final):
                non_targets=[i for i in range(len(indices)) if i not in cases]
                require(torch.equal(first[non_targets],first0[non_targets]) and torch.equal(final[non_targets],final0[non_targets]),'non-target masks changed')
                for rank in cases:require(np.array_equal(lookup(first[rank].numpy(),transform),final[rank].numpy()),'actual binary conversion lookup mismatch')

            def keep_candidate(case,rank,alpha,coefficients,first,final,grid):
                record=case['record'];actual=grid.numpy().astype(np.float64);proxy=case['z0']+alpha*case['direction']
                roi=case['roi'];error=np.abs(actual[roi]-proxy[roi]);tol=cfg['numerics']['proxy_atol']+cfg['numerics']['proxy_rtol']*np.abs(actual[roi])
                proxy_mask=predict(case['z0'],case['direction'],roi,alpha,transform);rle=encode_rle(final[rank])
                real_iou=full_rle_iou(rle,record['GT_rle']);proxy_iou=binary_iou(proxy_mask,case['gt'])
                valid=bool(np.all(error<=tol) and abs(real_iou-proxy_iou)<=cfg['numerics']['original_iou_tolerance'])
                record['numerical_valid'] &= valid
                result=dict(alpha=float(alpha),coefficients_fp32=coefficients[rank].tolist(),original_rle=rle,original_iou=real_iou,
                    proxy_original_rle=encode_rle(torch.tensor(proxy_mask,dtype=torch.uint8)),proxy_original_iou=proxy_iou,
                    proxy_logit_exceeded_pixels=int((error>tol).sum()),proxy_max_abs_error=float(error.max()) if len(error) else 0.,
                    proxy_original_iou_error=abs(real_iou-proxy_iou),proxy_original_different_pixels=int(np.count_nonzero(proxy_mask!=final[rank].numpy())),numerical_valid=valid)
                case['candidates'][float(alpha)]=result

            for rank,case in cases.items():keep_candidate(case,rank,0.,cache['c0'],first0,final0,z0s[rank])
            coefficients=cache['c0'].clone()
            for rank,case in cases.items():coefficients[rank]=(cache['c0'][rank].double()+delta[rank].double()).float()
            first,final,grids=native_stages(cache,coefficients,ranks);check_conversion(first,final)
            for rank,case in cases.items():
                keep_candidate(case,rank,1.,coefficients,first,final,grids[rank]);record=case['record']
                record['arms']['U']=dict(case['candidates'][1.])
                if scope=='full':
                    history=historical[record['output_detection_id']]
                    require(history['native_prediction']==record['native_prediction'] and history['GT_rle']==record['GT_rle'],'historical comparator identity')
                    require(history['ungated_rle']==record['arms']['U']['original_rle'],'unscaled U differs from prior Run')
                    record['arms']['G']=dict(original_rle=history['gated_rle'],original_iou=full_rle_iou(history['gated_rle'],record['GT_rle']))
            del first,final,grids,coefficients
            coefficients=cache['c0'].clone()
            for rank,case in cases.items():coefficients[rank]=torch.tensor(case['old']['arms']['C']['coefficients_fp32'])
            first,final,grids=native_stages(cache,coefficients,ranks);check_conversion(first,final)
            for rank,case in cases.items():
                rle=encode_rle(final[rank]);require(rle==case['old']['arms']['C']['original_rle'],'C coefficient replay differs')
                case['record']['arms']['C']=dict(original_rle=rle,original_iou=full_rle_iou(rle,case['record']['GT_rle']),coefficients_fp32=coefficients[rank].tolist())
            del first,final,grids,coefficients
            for case in cases.values():case['remaining']=[a for a in case['alphas'] if a not in (0.,1.)]
            for slot in range(max(len(c['remaining']) for c in cases.values())):
                resources(run);coefficients=cache['c0'].clone()
                for rank,case in cases.items():
                    if slot<len(case['remaining']):coefficients[rank]=(cache['c0'][rank].double()+case['remaining'][slot]*delta[rank].double()).float()
                first,final,grids=native_stages(cache,coefficients,ranks);native_candidate_calls+=1;check_conversion(first,final)
                for rank,case in cases.items():
                    if slot<len(case['remaining']):keep_candidate(case,rank,case['remaining'][slot],coefficients,first,final,grids[rank])
                del first,final,grids,coefficients
                torch.cuda.empty_cache()
            for rank,case in cases.items():
                record=case['record'];record['candidates']=[case['candidates'][a] for a in case['alphas']]
                selected=choose_actual(record['candidates']);record['selected_candidate_index']=selected
                record['arms']['R']=dict(record['candidates'][selected])
                require(record['arms']['R']['original_iou']>=max(record['arms'][a]['original_iou'] for a in ('N','U')),'selected ray worse than included endpoint')
                stream.write(json.dumps(record,allow_nan=False)+'\n');rows.append(record)
            stream.flush();processing_seconds+=time.monotonic()-started
            run.update(phase='finite_ray_audit',completed_images=image_number+1,completed_pairs=len(rows),adapter_forward_calls=image_number+1,
                zero_delta_exact_rows=zero_rows,native_candidate_decode_calls=native_candidate_calls)
            print('PROGRESS',image_number+1,'/',len(ids),'pairs',len(rows),'seconds',round(processing_seconds,1),flush=True)
            del cases,case,record,delta,cache,native_rows,first0,final0,z0s,transform
            gc.collect();torch.cuda.empty_cache()
    require(len(rows)==len(panel),'incomplete panel')
    require(tensor_state_hash(model)==state_hash,'adapter state changed')
    require(native_environment()==environment,'runtime changed')
    source.verify()
    for rel,digest in expected['files'].items():require(sha256(ROOT/rel)==digest,'post-run input drift: '+rel)
    summary=summarize(rows,cfg,scope=='full')
    summary.update(zero_delta_exact_rows=zero_rows,adapter_forward_calls=len(ids),S3_delta_exact_images=delta_exact,
        processing_seconds=processing_seconds,projected_full_seconds_from_smoke=processing_seconds*256/len(rows)*1.2+120 if scope=='smoke' else None)
    write_new(run.path/'STATISTICS.json',summary)
    independent=verify(run.path);write_new(run.path/'INDEPENDENT_LAPTOP.json',independent)
    require(independent['status']=='passed','independent verification failed')
    if scope=='smoke':
        require(summary['numerical_valid'],'smoke numeric mismatch')
        require(summary['projected_full_seconds_from_smoke']<=cfg['budget']['full_seconds'],'smoke exceeds projected full budget')
    write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',input_files_verified_before_and_after=len(expected['files']),cache_images_verified=len(ids),
        unchanged_adapter_state_sha256=state_hash,checkpoint_sha256=sha256(TRAIN/'last.pt'),environment_before_equals_after=True,
        zero_delta_exact_rows=zero_rows,N_U_C_replay_exact=True,all_non_target_masks_unchanged=True,all_detection_fields_unchanged=True,
        actual_binary_geometry_exact=True,numerical_proxy_checks_passed=summary['numerical_valid'],new_training_steps=0,base_model_forward_calls=0,
        adapter_forward_calls=len(ids),protocol_sha256=sha256(STUDY/'PROTOCOL.md')))
    next_action={'AMPLITUDE_PRIORITY':'Freeze a GT-free conditional amplitude prediction design; no automatic training',
                 'DIRECTION_PRIORITY':'Freeze a concrete beneficial-direction learning design; do not continue amplitude-only investment',
                 'MIXED_DIRECTION_AMPLITUDE':'Freeze a minimal joint direction/amplitude readout design, using the measured capture and residual gap',
                 'ENGINEERING_ONLY':'Execute the frozen full panel once after smoke verification'}.get(summary['decision'],'Retain evidence and resolve the concrete validity limitation without changing scientific parameters')
    run.finish(summary,status='passed' if summary['numerical_valid'] else 'inconclusive',scientific=summary['decision'],next_action=next_action)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['smoke','full'],required=True);p.add_argument('--parent-run');args=p.parse_args()
    parent=BASE
    if args.scope=='full':
        require(args.parent_run and Path(args.parent_run).name==args.parent_run,'exact smoke Run ID required')
        parent=STUDY/'runs'/args.parent_run;meta=read_json(parent/'run.json')
        require(meta['scope']=='smoke' and meta['execution_status']=='completed' and meta['exit_code']==0,'smoke incomplete')
        require(read_json(parent/'DECISION.json')['status']=='passed' and read_json(parent/'INDEPENDENT_LAPTOP.json')['status']=='passed','smoke not verified')
    require(not list((STUDY/'runs').glob('RUN_ADAPTER_RAY_'+args.scope.upper()+'_*')),'Run already exists; do not duplicate')
    run=RayRun(args.scope,parent);study=read_json(STUDY/'study.json')
    try:
        with run:
            study.update(execution_status='running',scientific_status='unadjudicated',artifact_status='partial',active_run_id=run.run_id,current_phase=args.scope)
            save(STUDY/'study.json',study);execute(run)
    finally:
        if (run.path/'run.json').exists():
            meta=read_json(run.path/'run.json');study.update(execution_status=meta['execution_status'],scientific_status=meta['scientific_status'],artifact_status=meta['artifact_status'],
                active_run_id=None,latest_run_id=run.run_id,current_phase='full_ready_after_smoke' if args.scope=='smoke' and meta['execution_status']=='completed' else 'awaiting_transfer_and_local_verification' if meta['execution_status']=='completed' else 'engineering_failure_retained')
            save(STUDY/'study.json',study)

if __name__=='__main__':main()
