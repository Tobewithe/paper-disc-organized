"""Two prespecified spatial parameterizations, original same-output 256-target panel."""
import argparse
from collections import defaultdict
from common import *
from quadrant import SOLVER,partition,active_quadrants,solve,positive_control,changed_logits,decode_logits
sys.path.insert(0,str(ROOT/'experiments/mask_readout_subspace_repulsion_20261015/scripts'))
from repulsion import choose_neighbors
from s1_geometry import project_gt,predicted_roi


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('smoke','full'),required=True);p.add_argument('--smoke-run',default='');a=p.parse_args()
    with ExperimentRun('SMOKE' if a.scope=='smoke' else 'ORACLE',a.scope,S1_RUN) as run:
        expected,_,_,environment=inspect();import torch
        from pycocotools.coco import COCO
        from pycocotools import mask as masks
        torch.set_num_threads(1);panel=read_json(S1_RUN/'TARGET_PANEL.json');targets=panel['targets'] if a.scope=='full' else panel['targets'][:4]
        require(len(panel['targets'])==256 and len(set(v['output_detection_id'] for v in panel['targets']))==256,'S1 target identity changed')
        original={(r['image_id'],r['annotation_id']):r for r in read_jsonl(S1_RUN/'ORACLE_RESULTS.jsonl')}
        require(sum(r['oracle_mask_iou']>=.75 for r in original.values())==122,'S1 reference changed')
        ids=sorted({r['image_id'] for r in targets});contract=dict(method_id='E3-QUADRANT-v1',scope=a.scope,targets=len(targets),image_ids=ids,
            source_panel_sha256=sha256(S1_RUN/'TARGET_PANEL.json'),source_oracle_sha256=sha256(S1_RUN/'ORACLE_RESULTS.jsonl'),solver=SOLVER,
            input_hashes=expected,environment=environment,main_candidate='Q-all',restricted_candidate='Q-local',minimum_rate=.2,minimum_net_new=13,
            new_training_steps=0,base_model_forward_calls=0,wall_seconds=run.wall_seconds)
        if a.scope=='full':
            smoke=STUDY/'runs'/a.smoke_run;require(read_json(smoke/'run.json')['execution_status']=='completed','smoke required')
            for prior in (STUDY/'runs').glob('RUN_QUADRANT_ORACLE_*/run.json'):
                if prior.parent!=run.path:require(not read_json(prior).get('optimization_started'),'oracle budget already consumed')
        run.freeze(contract)
        if a.scope=='full':require(read_json(smoke/'SOURCE_MANIFEST.json')['files']==run.sources,'source changed after smoke')
        write_new(run.path/'TARGET_PANEL.json',dict(source_panel_sha256=contract['source_panel_sha256'],targets=targets))
        control=positive_control();write_new(run.path/'POSITIVE_CONTROL.json',control);require(control['passed'],'solver positive control failed')
        source=FrozenCache(BASE/'trace',ids);coco=COCO(str(GT));by_image=defaultdict(list)
        for target in targets:by_image[target['image_id']].append(target)
        results=[];checks=[];torch.cuda.reset_peak_memory_stats();run.update(optimization_started=True,phase='finite_spatial_oracles')
        with (run.path/'ORACLE_RESULTS.jsonl').open('x',encoding='utf8') as out:
            for image_id in source.records:
                run.check_budget();cache,native=source.load(image_id);ranks={idx:i for i,idx in enumerate(cache['prediction_indices'])}
                with torch.inference_mode():
                    zero=decode_logits(cache,changed_logits(cache,0,torch.zeros(4,32)))
                    zero_rles=[encode_rle(m) for m in zero]
                require(zero_rles==[r['segmentation'] for r in native],'spatial zero-delta decoder failed native replay')
                checks.append(dict(image_id=image_id,rows=len(native),native_RLE_exact=True));del zero;torch.cuda.empty_cache()
                neighbors,_=choose_neighbors(cache['boxes_input'].tolist(),cache['scores'].tolist(),cache['classes'].tolist(),cache['raw_ids'])
                for target in by_image[image_id]:
                    run.check_budget();idx=int(target['output_detection_id'].rsplit(':',1)[1]);rank=ranks[idx];ann=coco.anns[target['annotation_id']]
                    require(ann['image_id']==image_id and ann['category_id']==target['category_id'],'GT identity mismatch')
                    truth=coco.annToRLE(ann);before=full_rle_iou(native[rank]['segmentation'],truth);require(abs(before-target['baseline_mask_iou'])<1e-12,'baseline IoU changed')
                    old=original[image_id,target['annotation_id']];require(old['output_detection_id']==target['output_detection_id'],'global oracle pairing changed')
                    roi=predicted_roi(cache,rank);require(bool(roi.any()),'empty original ROI');gt=project_gt(masks.decode(truth),cache)[roi].double()
                    _,weights=partition(cache,rank);proto=cache['proto'].double();base=(cache['c0'][rank].double()@proto.flatten(1)).reshape(proto.shape[-2:])[roi]
                    local=active_quadrants(cache,rank,neighbors[rank]);row=dict(target,rank=rank,neighbor_rank=neighbors[rank],active_local_quadrants=local,
                        cache_relative_path=source.records[image_id]['cache'],cache_sha256=source.records[image_id]['cache_sha256'],
                        native_rle=native[rank]['segmentation'],old_global_iou=old['oracle_mask_iou'],arms={})
                    for name,active in [('Q-all',list(range(4))),('Q-local',local)]:
                        design=torch.cat([(proto*weights[q]).permute(1,2,0)[roi] for q in active],1) if active else torch.empty(int(roi.sum()),0,dtype=torch.float64)
                        solved=solve(design,gt,base);delta=solved.pop('delta');all_delta=torch.zeros(4,32,dtype=torch.float64)
                        for position,q in enumerate(active):all_delta[q]=delta[position*32:(position+1)*32]
                        with torch.inference_mode():
                            decoded=decode_logits(cache,changed_logits(cache,rank,all_delta));rle=encode_rle(decoded[rank])
                            for other in range(len(native)):
                                if other!=rank:require(encode_rle(decoded[other])==zero_rles[other],'other output changed')
                        iou=full_rle_iou(rle,truth);row['arms'][name]=dict(**solved,active_quadrants=active,delta_float64=all_delta.tolist(),
                            oracle_rle=rle,oracle_mask_iou=iou,recovered=iou>=.75,roi_pixels=int(roi.sum()))
                        del decoded,design,delta,all_delta;torch.cuda.empty_cache()
                    out.write(json.dumps(row,allow_nan=False)+'\n');out.flush();results.append(row)
                    if len(results)%16==0 or len(results)==len(targets):print('TARGETS',len(results),len(targets),flush=True)
                    run.update(completed_targets=len(results))
                del cache,native,zero_rles;gc.collect();torch.cuda.empty_cache()
        require(len(results)==len(targets),'target coverage changed');source.verify();again=inspect();require(again[0]==expected and again[3]==environment,'input/environment changed')
        arms={}
        for name in ('Q-all','Q-local'):
            found=sum(r['arms'][name]['recovered'] for r in results);old_found=sum(r['old_global_iou']>=.75 for r in results)
            unconverged=sum(r['arms'][name]['solver_status']=='budget_exhausted' for r in results)
            passing=found/len(results)>=.2 and found-old_found>=13
            arms[name]=dict(found=found,total=len(results),rate=found/len(results),old_global_found=old_found,net_new=found-old_found,
                new_over_global=sum(r['arms'][name]['recovered'] and r['old_global_iou']<.75 for r in results),
                lost_from_global=sum(not r['arms'][name]['recovered'] and r['old_global_iou']>=.75 for r in results),
                unconverged=unconverged,status=('passed' if passing else 'blocked_precision' if unconverged else 'failed') if a.scope=='full' else 'engineering_only')
        import psutil
        mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss);require(peak<=4*1024**3 and torch.cuda.max_memory_reserved()<=6.5*1024**3,'resource budget exceeded');run.check_budget()
        write_new(run.path/'CONTROLS.json',dict(solver_positive_control=control,images=checks))
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',peak_rss_bytes=peak,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            new_training_steps=0,base_model_forward_calls=0,checks=dict(original_fixed_outputs_and_GT=True,native_zero_RLE_exact=True,
            unchanged_other_detection_rows=True,same_cache_and_native_environment=True,original_global_oracle_reused=True,no_extra_solver_budget=True)))
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file())<=1024**3,'artifact budget exceeded')
        status=arms['Q-all']['status'] if a.scope=='full' else 'passed'
        run.finish(dict(arms=arms,total=len(results),scientific_scope='GT-assisted finite spatial coefficient opportunity; not deployable S2/AP'),
            status=status,scientific='E3_opportunity_'+status if a.scope=='full' else 'engineering_only',
            next_action='Close E3 then execute independently frozen E4 loss study per user sequence' if a.scope=='full' else 'Run frozen original 256-target panel')
        print('E3_DECISION',json.dumps(arms),flush=True)


if __name__=='__main__':main()
