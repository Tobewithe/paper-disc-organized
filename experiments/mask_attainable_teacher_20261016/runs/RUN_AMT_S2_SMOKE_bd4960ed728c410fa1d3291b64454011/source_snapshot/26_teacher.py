"""Train2017-only, original-join finite functional teachers, before student training."""
import argparse
from collections import Counter
import json
from common import *
from targets import mask_hash, accept

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope',choices=['smoke','full'],required=True)
    p.add_argument('--smoke-run',type=Path)
    args=p.parse_args()
    full=args.scope=='full'
    with Run('TEACHER_'+args.scope.upper(),CACHE,14400 if full else 900) as run:
        data=inspect_inputs()
        import torch
        from pycocotools import mask as masks
        from pycocotools.coco import COCO
        from adapter_data import FrozenCache, training_join, projected_targets
        from adapter_model import loss_grids
        from s1_geometry import predicted_roi, full_rle_iou
        from finite_coefficient_solver import solve_coefficients, positive_control
        from native_output_bridge import decode_native, encode_rle
        ids=data['split']['fit_ids'] if full else data['split']['fit_ids'][:8]
        if full:
            require(args.smoke_run is not None,'teacher smoke required')
            require(read_json(args.smoke_run/'DECISION.json')['status']=='passed','teacher smoke failed')
            require(read_json(args.smoke_run/'INDEPENDENT_TEACHER.json')['status']=='passed','teacher smoke independently unverified')
            for path in (STUDY/'runs').glob('RUN_AMT_TEACHER_FULL_*/run.json'):
                if path.parent!=run.path:
                    require(not read_json(path).get('optimization_started'),'formal teacher budget consumed')
        run.freeze(dict(scope=args.scope,stage='train_only_teacher',spec=SPEC,image_ids=ids,
            inputs_sha256=data['inputs_sha256'],environment=data['environment'],wall_seconds=run.wall_seconds,
            adapter_optimizer_steps=0,teacher_annotation_source=str(TRAIN_GT)))
        if full:
            require(read_json(args.smoke_run/'SOURCE_MANIFEST.json')['files']==run.sources,'source changed after smoke')
        write_new(run.path/'ENVIRONMENT.json',data['environment'])
        control=positive_control()
        require(control['passed'],'solver positive control failed')
        write_new(run.path/'SOLVER_CONTROL.json',control)
        source=FrozenCache(CACHE/'trace',ids)
        coco=COCO(str(TRAIN_GT))
        baseline={(r['image_id'],r['annotation_id']):r for r in read_jsonl(FIT/'BASELINE_GT_LEDGER.jsonl')}
        oldjoins=read_json(INITIAL/'TRAINING_JOIN.json')['images']
        results=[]
        packs={}
        total_outputs=0
        checks=[]
        run.update(optimization_started=True,phase='train_only_finite_teacher',images_completed=0)
        with (run.path/'TEACHER_ROWS.jsonl').open('x',encoding='utf8') as output:
            for image_id in ids:
                run.check_budget()
                cache,rows=source.load(image_id)
                join=training_join(rows,coco.imgToAnns.get(image_id,[]))
                require({str(k):v for k,v in join.items()}==oldjoins[str(image_id)]['output_rank_to_annotation_id'],'original training join drift')
                original=decode_native(cache)
                require(all(encode_rle(original[k])==r['segmentation'] for k,r in enumerate(rows)),'zero-delta RLE differs')
                del original
                total_outputs+=len(rows)
                matrix=cache['c0'].clone()
                projected=projected_targets(coco,cache,join)
                pending=[]
                for rank,aid in join.items():
                    run.check_budget()
                    ann=coco.anns[aid]
                    label=baseline[(image_id,aid)]
                    require(not label['is_ignored'] and ann['image_id']==image_id and ann['category_id']==rows[rank]['category_id'],'invalid training association')
                    roi=predicted_roi(cache,rank)
                    g=coco.annToRLE(ann)
                    native_iou=full_rle_iou(rows[rank]['segmentation'],g)
                    eligible=label['task_mask75']!='success' and native_iou<.75 and bool(roi.any())
                    r=dict(image_id=image_id,annotation_id=aid,rank=rank,global_index=cache['prediction_indices'][rank],
                        output_detection_id=source.manifest['predictions_sha256']+':'+str(cache['prediction_indices'][rank]),
                        baseline_iou=native_iou,baseline_task_mask75=label['task_mask75'],
                        native_area=int(masks.area(rows[rank]['segmentation'])),roi_pixels=int(roi.sum()),
                        cache_sha256=source.records[image_id]['cache_sha256'],attempted=eligible,accepted=False,
                        target_kind='original_GT',solver_status='not_attempted')
                    if eligible:
                        solved=solve_coefficients(cache['proto'].permute(1,2,0)[roi],projected[rank][roi],cache['c0'][rank],SPEC['solver'])
                        matrix[rank]=solved.pop('coefficients').to(matrix)
                        solved.pop('delta')
                        r.update(solved,coefficients_fp32=matrix[rank].tolist())
                    else:
                        r['fallback_reason']='original_success' if label['task_mask75']=='success' else 'empty_ROI' if not bool(roi.any()) else 'paired_mask_already_ge_0.75'
                    pending.append((r,g))
                modified=decode_native(cache,matrix)
                # Same full matrix shape/FP32 CUDA path as the student loss, even for ineligible rows.
                with torch.no_grad():
                    logits,_,roi_grid=loss_grids(cache,matrix.cuda())
                    q=logits.gt(0).to(torch.uint8).cpu()
                    roi_grid=roi_grid.cpu()
                del logits
                targets={}
                target_hashes={}
                for r,g in pending:
                    rank=r['rank']
                    require(torch.equal(roi_grid[rank],predicted_roi(cache,rank)),'ROI solver/loss mismatch')
                    if r['attempted']:
                        rle=encode_rle(modified[rank])
                        iou=full_rle_iou(rle,g)
                        foreground=int(q[rank][roi_grid[rank]].sum())
                        accepted=accept(r['baseline_iou'],iou,r['solver_status'],foreground)
                        r.update(oracle_iou=iou,oracle_rle=rle,teacher_roi_foreground=foreground,accepted=accepted,
                            target_kind='attainable_binary_proto' if accepted else 'original_GT',
                            fallback_reason=None if accepted else 'solver_not_converged_or_gain_lt_0.01_or_empty_foreground')
                        if accepted:
                            targets[rank]=q[rank].clone()
                            target_hashes[rank]=mask_hash(targets[rank])
                            r['target_sha256']=target_hashes[rank]
                    results.append(r)
                    output.write(json.dumps(r,allow_nan=False)+'\n')
                del modified,q,roi_grid
                # No coefficient regression: coefficient solutions are evidence, never model inputs/labels.
                package=dict(image_id=image_id,prediction_indices=cache['prediction_indices'],join=join,
                    targets=targets,target_hashes=target_hashes,cache_sha256=source.records[image_id]['cache_sha256'])
                path=run.path/'targets'/f'{image_id}.pt'
                path.parent.mkdir(exist_ok=True)
                torch.save(package,path)
                packs[str(image_id)]=dict(path=path.relative_to(run.path).as_posix(),sha256=sha256(path),accepted=len(targets),joins=len(join))
                checks.append(dict(image_id=image_id,zero_delta_RLE_exact=True,outputs=len(rows)))
                require(sha256(source.trace/source.records[image_id]['cache'])==source.records[image_id]['cache_sha256'],'cache changed')
                output.flush()
                run.update(images_completed=len(packs),solved=sum(x['attempted'] for x in results),accepted=sum(x['accepted'] for x in results))
                if len(packs)%25==0 or len(packs)==len(ids):
                    print('TEACHER',len(packs),'/',len(ids),'solved',run.meta['solved'],'accepted',run.meta['accepted'],flush=True)
                del cache,rows,projected,matrix,targets,package
        source.verify()
        verify_after(data)
        gate=teacher_gate(results)
        require(sum(p['joins'] for p in packs.values())==len(results),'join coverage changed')
        if full:
            require(len(results)==14276 and sum(r['roi_pixels']>0 for r in results)==14268,'original supervised population changed')
            require(total_outputs==218209,'training output count drift')
        require(not any(r['accepted'] and r['baseline_task_mask75']=='success' for r in results),'successful GT target changed')
        write_new(run.path/'TEACHER_MANIFEST.json',dict(scope=args.scope,images=packs,spec=SPEC,rows_sha256=sha256(run.path/'TEACHER_ROWS.jsonl')))
        write_new(run.path/'CONTROLS.json',dict(solver=control,zero_delta=checks))
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',original_join_unchanged=True,
            frozen_inputs_and_runtime_unchanged=True,all_successful_GT_targets_unchanged=True,GT_train_only=True,
            same_native_geometry_and_decoder=True,base_model_forward_calls=0,adapter_optimizer_steps=0))
        status='passed' if not full or gate['passed'] else 'blocked' if not gate['checks']['convergence'] else 'failed'
        run.finish(dict(images=len(ids),outputs=total_outputs,joins=len(results),teacher_gate=gate,
            target_kinds=dict(Counter(r['target_kind'] for r in results)),teacher_manifest_sha256=sha256(run.path/'TEACHER_MANIFEST.json')),
            status=status,scientific='engineering_only' if not full else 'teacher_opportunity_'+status,
            next_action='Independent replay then original-init student training' if status=='passed' else 'Stop before student training; retain failed teacher Gate')
        print('TEACHER_FINISHED',run.run_id,status,flush=True)

if __name__=='__main__':
    main()
