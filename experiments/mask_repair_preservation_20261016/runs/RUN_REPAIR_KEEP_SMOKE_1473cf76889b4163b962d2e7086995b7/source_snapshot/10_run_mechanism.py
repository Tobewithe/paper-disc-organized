"""One registered smoke or full finite-coefficient comparison; no base/adapter forward."""
from mechanism_common import *
import argparse
from collections import defaultdict
import contextlib
import gc
import time


def execute(run):
    import torch
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    from adapter_data import FrozenCache
    from s1_geometry import full_rle_iou
    from native_output_bridge import decode_native, encode_rle, native_environment
    from native_geometry import native_stages, roi_for, target_grid, lifted_basis, pixel_counts
    from objectives import solve, rms, equal_scale
    from analysis_stats import summarize
    from verify import verify

    cfg, expected, profile, environment = inspect_inputs()
    preparation = read_json(STUDY / 'PREPARATION.json')
    require(environment == preparation['environment'], 'runtime differs from preparation')
    scope = run.meta['scope']
    panel = preparation[scope + '_targets']
    ids = sorted({r['image_id'] for r in panel})
    run.freeze(dict(scope=scope, environment=environment, native_profile=profile,
        candidate_policy='fixed_output_detection', input_files=expected['files'],
        panel_sha256=cfg['panel']['sha256'], targets=panel, image_ids=ids, config=cfg,
        optimizer_surrogate='native_FP32_z0_plus_FP64_bilinear_P_times_delta',
        actual_decoder='pinned_full_shape_FP32_process_mask_native_then_scale_masks_byte',
        new_training_steps=0, base_model_forward_calls=0, comparison_scope='exposed finite-oracle panel, not deployed AP'))
    write_new(run.path / 'TARGETS.json', panel)
    source = FrozenCache(BASE / 'trace', ids)
    with (run.path / 'coco_load.log').open('x',encoding='utf8') as log, contextlib.redirect_stdout(log):
        coco = COCO(str(GT))
    historical = {r['output_detection_id']:r for r in read_jsonl(AUDIT / 'PAIRED_MASKS.jsonl')}
    by_image = defaultdict(list)
    for row in panel:
        by_image[row['image_id']].append(row)
    write_new(run.path / 'INPUT_CACHE_RECORDS.json', [source.records[i] for i in ids])
    rows, solves, zero_rows, process_seconds = [], 0, 0, 0.
    with (run.path / 'PAIR_RESULTS.jsonl').open('x',encoding='utf8') as stream:
        for image_number,image_id in enumerate(ids):
            resources(run)
            started = time.monotonic()
            cache, native_rows = source.load(image_id)
            indices = cache['prediction_indices']
            ranks_by_index = {index:rank for rank,index in enumerate(indices)}
            targets = by_image[image_id]
            target_ranks = [ranks_by_index[int(t['output_detection_id'].rsplit(':',1)[1])] for t in targets]
            require(len(target_ranks)==len(set(target_ranks)), 'duplicated target detection')
            first0, final0, z0_all = native_stages(cache, cache['c0'], target_ranks)
            require(torch.equal(final0, decode_native(cache).cpu()), 'native stage/bridge replay differs')
            for rank,prediction in enumerate(native_rows):
                require(encode_rle(final0[rank])==prediction['segmentation'], 'native zero-delta RLE differs')
                zero_rows += 1
            basis = lifted_basis(cache)
            cases = {}
            for target,rank in zip(targets,target_ranks,strict=True):
                native = native_rows[rank]
                ann = coco.anns[target['annotation_id']]
                require(ann['image_id']==native['image_id']==image_id, 'GT/detection image mismatch')
                require(ann['category_id']==native['category_id']==target['category_id'], 'GT/detection category mismatch')
                require(not ann.get('iscrowd',0) and not ann.get('ignore',0), 'ignored GT')
                require(target['output_detection_id']==source.manifest['predictions_sha256']+':'+str(indices[rank]), 'detection namespace mismatch')
                gt = coco.annToRLE(ann)
                if isinstance(gt['counts'],bytes):gt={**gt,'counts':gt['counts'].decode('ascii')}
                native_iou = full_rle_iou(native['segmentation'],gt)
                require(abs(native_iou-target['baseline_mask_iou'])<=1e-12 and native_iou<.75, 'native fixed-pair IoU mismatch')
                require(float(masks.iou([native['bbox']],[ann['bbox']],[0])[0,0])>=.75, 'not a fixed Box75 detection')
                roi = roi_for(cache,rank)
                truth = target_grid(masks.decode(gt),cache)
                z0 = z0_all[rank][roi].double()
                design = basis[:,roi].T.contiguous()
                y = truth[roi].double()
                initial_counts = pixel_counts(first0[rank],first0[rank],truth,roi)
                record = dict(**target,rank=rank,global_index=indices[rank],raw_id=cache['raw_ids'][rank],
                    cache_sha256=source.records[image_id]['cache_sha256'],c0=cache['c0'][rank].tolist(),
                    boxes_input=cache['boxes_input'][rank].tolist(),input_hw=list(cache['native_decode_hw']),
                    native_prediction=native,GT_rle=gt,target_grid_rle=encode_rle(truth.byte()),
                    roi_pixels=int(roi.sum()),native_tp=initial_counts['native_tp'],
                    numerical_valid=True,arms=dict(N=dict(original_iou=native_iou,original_rle=native['segmentation'],
                        input_rle=encode_rle(first0[rank]),**initial_counts)))
                if scope=='full':
                    old=historical[target['output_detection_id']]
                    require(old['native_prediction']==native and old['GT_rle']==gt, 'historical comparator identity changed')
                    for arm,key in (('R_G','gated_rle'),('R_S1','oracle_rle')):
                        record['arms'][arm]=dict(original_rle=old[key],original_iou=full_rle_iou(old[key],gt))
                case = dict(record=record,design=design,z0=z0,target=y,truth=truth,roi=roi,deltas={})
                for arm in ('C','E','EK'):
                    delta,receipt=solve(design,z0,y,arm,cfg['solver'],lambda:resources(run))
                    solves += 1
                    case['deltas'][arm]=delta
                    record['arms'][arm]=dict(delta_float64=delta.tolist(),solver=receipt)
                cases[rank]=case

            def evaluate_arm(arm):
                coefficients=cache['c0'].clone()
                for rank,case in cases.items():
                    coefficients[rank]=(cache['c0'][rank].double()+case['deltas'][arm]).float()
                first,final,grids=native_stages(cache,coefficients,target_ranks)
                unchanged=[i for i in range(len(indices)) if i not in cases]
                require(torch.equal(first[unchanged],first0[unchanged]) and torch.equal(final[unchanged],final0[unchanged]),
                        'non-target outputs changed')
                for rank,case in cases.items():
                    record=case['record'];actual=grids[rank][case['roi']].double()
                    proxy=case['z0']+case['design']@case['deltas'][arm]
                    tolerance=cfg['numerics']['proxy_atol']+cfg['numerics']['proxy_rtol']*actual.abs()
                    bad=int(((proxy-actual).abs()>tolerance).sum())
                    record['numerical_valid'] &= bad==0
                    result=record['arms'][arm]
                    rle=encode_rle(final[rank])
                    result.update(coefficients_fp32=coefficients[rank].tolist(),original_rle=rle,
                        original_iou=full_rle_iou(rle,record['GT_rle']),input_rle=encode_rle(first[rank]),
                        actual_delta_logit_rms=rms(actual-case['z0']),proxy_exceeded_pixels=bad,
                        proxy_max_abs_error=float((proxy-actual).abs().max()) if len(actual) else 0.,
                        proxy_binary_disagreements=int(((proxy>0)!=(actual>0)).sum()),
                        **pixel_counts(first0[rank],first[rank],case['truth'],case['roi']))
                del first,final,grids,coefficients
                torch.cuda.empty_cache()

            for arm in ('C','E','EK'):
                evaluate_arm(arm)
            for rank,case in cases.items():
                record=case['record']
                alpha,status=equal_scale(record['arms']['E']['actual_delta_logit_rms'],record['arms']['EK']['actual_delta_logit_rms'])
                delta=case['deltas']['E']*alpha
                require(torch.isfinite(delta).all(), 'equal-amplitude coefficients are nonfinite')
                case['deltas']['E_equal']=delta
                record['arms']['E_equal']=dict(scale=alpha,scale_status=status,delta_float64=delta.tolist())
                record['numerical_valid'] &= status!='undefined_zero_direction'
            evaluate_arm('E_equal')
            for rank,case in cases.items():
                record=case['record']
                a,b=(record['arms'][v]['actual_delta_logit_rms'] for v in ('E_equal','EK'))
                tolerance=cfg['numerics']['equal_rms_atol']+cfg['numerics']['equal_rms_rtol']*b
                record['arms']['E_equal']['rms_match_error']=abs(a-b)
                record['numerical_valid'] &= abs(a-b)<=tolerance
                stream.write(json.dumps(record,allow_nan=False)+'\n')
                rows.append(record)
            stream.flush()
            process_seconds += time.monotonic()-started
            run.update(phase='finite_coefficient_comparison',completed_images=image_number+1,completed_pairs=len(rows),
                oracle_optimization_calls=solves,zero_delta_exact_rows=zero_rows)
            print('PROGRESS',image_number+1,'/',len(ids),'pairs',len(rows),'solves',solves,'seconds',round(process_seconds,1),flush=True)
            del cases,basis,cache,native_rows,first0,final0,z0_all
            gc.collect();torch.cuda.empty_cache()

    require(len(rows)==len(panel) and solves==3*len(panel), 'incomplete frozen execution')
    require(native_environment()==environment,'runtime configuration drift after execution')
    summary=summarize(rows,cfg,scientific=scope=='full')
    summary.update(zero_delta_exact_rows=zero_rows,oracle_optimization_calls=solves,
        solve_and_decode_seconds=process_seconds,projected_full_seconds_from_smoke=(process_seconds*256/len(rows)*1.2+120 if scope=='smoke' else None))
    write_new(run.path/'STATISTICS.json',summary)
    independent=verify(run.path)
    write_new(run.path/'INDEPENDENT_LAPTOP.json',independent)
    require(independent['status']=='passed','independent verification failed')
    if scope=='smoke':
        require(summary['numerical_valid'],'smoke numerical validity failed; retain, do not tune science')
        require(summary['projected_full_seconds_from_smoke']<=cfg['budget']['full_seconds'],'projected budget exceeds frozen full budget')
    write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',environment_before_equals_after=True,
        full_shape_native_decoder=True,zero_delta_exact_rows=zero_rows,input_files_verified=len(expected['files']),
        immutable_output_fields=True,non_target_masks_unchanged=True,network_training_steps=0,base_model_forward_calls=0,
        all_numeric_proxy_checks_passed=summary['numerical_valid'],protocol_sha256=sha256(STUDY/'PROTOCOL.md')))
    next_action=('Run the frozen full panel once' if scope=='smoke' else
        'Recommend a separately frozen, same-architecture training ablation; do not auto-start' if summary['decision']=='FINITE_MECHANISM_SUPPORTED' else
        'Close this objective configuration; prioritize conditional prediction of witnessed beneficial directions' if summary['decision']=='CONFIGURATION_REJECTED' else
        'Retain inconclusive evidence; resolve the recorded numerical or precision limitation without tuning this Run')
    run.finish(summary,status='passed' if scope=='smoke' or summary['decision']=='FINITE_MECHANISM_SUPPORTED' else 'failed',
        scientific='engineering_only' if scope=='smoke' else summary['decision'],next_action=next_action)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['smoke','full'],required=True)
    parser.add_argument('--parent-run');args=parser.parse_args()
    parent=BASE
    if args.scope=='full':
        require(args.parent_run is not None and Path(args.parent_run).name==args.parent_run,'full requires exact smoke Run ID')
        parent=STUDY/'runs'/args.parent_run
        meta=read_json(parent/'run.json')
        require(meta['scope']=='smoke' and meta['execution_status']=='completed' and meta['exit_code']==0,'smoke not complete')
        require(read_json(parent/'DECISION.json')['status']=='passed','smoke not passed')
        require(read_json(parent/'INDEPENDENT_LAPTOP.json')['status']=='passed','smoke independent verification missing')
    require(not list((STUDY/'runs').glob('RUN_REPAIR_KEEP_'+args.scope.upper()+'_*')),'a Run already exists for this scope; do not repeat automatically')
    run=MechanismRun(args.scope,parent)
    study=read_json(STUDY/'study.json')
    try:
        with run:
            study.update(execution_status='running',active_run_id=run.run_id,current_phase=args.scope)
            save(STUDY/'study.json',study)
            execute(run)
    finally:
        if (run.path/'run.json').exists():
            meta=read_json(run.path/'run.json')
            study.update(execution_status=meta['execution_status'],active_run_id=None,latest_run_id=run.run_id,
                scientific_status=meta['scientific_status'],artifact_status=meta['artifact_status'],
                current_phase=('full_ready_after_smoke' if args.scope=='smoke' and meta['execution_status']=='completed' else
                               'awaiting_transfer_and_local_verification' if meta['execution_status']=='completed' else 'engineering_failure_retained'))
            save(STUDY/'study.json',study)


if __name__=='__main__':
    main()
