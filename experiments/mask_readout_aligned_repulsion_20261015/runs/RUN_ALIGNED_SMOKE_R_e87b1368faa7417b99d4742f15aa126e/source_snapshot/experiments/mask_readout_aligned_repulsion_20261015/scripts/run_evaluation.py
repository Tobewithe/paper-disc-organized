"""Frozen E2 checkpoint: complete native output and original S2 populations."""
import argparse
from common import *
from adapter_model import CoefficientAdapter
MODEL=None

def export(run,source):
    import torch
    source_indices=[];stats=Counter();by_index={};started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (run.path/'ALIGNED_predictions.json').open('x',encoding='utf8') as out,(run.path/'RULE_ROWS.jsonl').open('x',encoding='utf8') as rules,(run.path/'IMAGE_METADATA.jsonl').open('x',encoding='utf8') as metadata,torch.inference_mode():
        out.write('[');count=0
        for number,image_id in enumerate(source.records,1):
            run.check_budget();cache,original=source.load(image_id)
            native_masks=decode_native(cache,device='cuda');native_rles=[encode_rle(m) for m in native_masks]
            require(native_rles==[r['segmentation'] for r in original],'native zero-delta RLE mismatch')
            del native_masks;torch.cuda.empty_cache()
            delta=MODEL(cache,'eval:0');coefficients=cache['c0'].to(delta)+delta
            original_c=cache['c0'].tolist();actual_c=coefficients.cpu().tolist();raw=delta.cpu().tolist()
            proof=[dict(output_rank=i,c0=original_c[i],c_new=actual_c[i],raw_delta=raw[i],applied=True,neighbor_rank=-1,omega_pixels=0) for i in range(len(original_c))]
            masks=decode_native(cache,coefficients,device='cuda');rles=[encode_rle(m) for m in masks]
            rows=[dict(r,segmentation=m) for r,m in zip(original,rles)];compare_outputs(original,rows)
            for rank,(row,record,index) in enumerate(zip(rows,proof,cache['prediction_indices'])):
                unchanged=row['segmentation']==native_rles[rank]
                if not record['applied']:require(unchanged,'unchanged coefficient mask differs')
                if count:out.write(',')
                out.write(json.dumps(row,separators=(',',':'),allow_nan=False))
                record.update(image_id=image_id,global_native_index=index,method_index=count,mask_changed=not unchanged,
                    raw_id=cache['raw_ids'][rank],neighbor_raw_id=None if record['neighbor_rank']<0 else cache['raw_ids'][record['neighbor_rank']])
                rules.write(json.dumps(record,allow_nan=False)+'\n')
                source_indices.append(index);by_index[index]=dict(applied=record['applied'],neighbor_rank=record['neighbor_rank'],omega_pixels=record['omega_pixels'])
                count+=1;stats['outputs']+=1;stats['selected_coefficients']+=int(record['applied']);stats['changed_masks']+=int(not unchanged)
                stats['no_eligible_neighbor']+=int(record['neighbor_rank']<0)
                stats['empty_region_with_neighbor']+=int(record['neighbor_rank']>=0 and record['omega_pixels']==0)
            metadata.write(json.dumps(dict(image_id=image_id,cache_relative_path=source.records[image_id]['cache'],
                cache_sha256=source.records[image_id]['cache_sha256'],prediction_indices=cache['prediction_indices'],
                boxes=cache['boxes_input'].tolist(),scores=cache['scores'].tolist(),classes=cache['classes'].tolist(),
                raw_ids=cache['raw_ids'],input_hw=cache['input_hw'],proto_hw=list(cache['proto'].shape[-2:])),allow_nan=False)+'\n')
            stats['images']+=1;stats['images_with_updates']+=int(any(r['applied'] for r in proof))
            del cache,coefficients,masks,rles,rows,native_rles,proof,original;gc.collect();torch.cuda.empty_cache()
            if number%50==0 or number==len(source.records):print('EXPORT',number,len(source.records),dict(stats),flush=True)
    with (run.path/'ALIGNED_predictions.json').open('a',encoding='utf8') as out:out.write(']\n')
    require(source_indices==sorted(source_indices),'source order changed')
    expected=read_json(PARENT/'B_ROW_MAPPING.json')['original_global_indices']
    if run.meta['scope']=='full':require(source_indices==expected and len(source_indices)==185621,'full output denominator changed')
    stats.update(export_seconds=time.monotonic()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    require(stats['cuda_peak_reserved_bytes']<=6.5*1024**3,'GPU budget exceeded')
    write_new(run.path/'ROW_MAPPING.json',dict(source_prediction_sha256=source.manifest['predictions_sha256'],original_global_indices=source_indices))
    return source_indices,by_index,dict(stats)


def fixed_pairs(run,source,ids,indices,actions):
    from pycocotools.coco import COCO
    coco=COCO(str(GT));predictions=read_json(run.path/'ALIGNED_predictions.json');position={g:i for i,g in enumerate(indices)}
    effects=[]
    for old in read_jsonl(PARENT/'B_FIXED_PAIR_EFFECTS.jsonl'):
        if old['image_id'] not in set(ids):continue
        index=int(old['baseline_output_detection_id'].rsplit(':',1)[1]);gt=coco.annToRLE(coco.anns[old['annotation_id']])
        before=full_rle_iou(source.predictions[index]['segmentation'],gt)
        after=full_rle_iou(predictions[position[index]]['segmentation'],gt)
        require(abs(before-old['baseline_iou'])<1e-12,'baseline fixed pair changed')
        effects.append(dict(image_id=old['image_id'],annotation_id=old['annotation_id'],population=old['population'],
            baseline_output_detection_id=old['baseline_output_detection_id'],baseline_iou=before,method_iou=after,
            delta_iou=after-before,same_output_reaches_075=after>=.75,**actions[index]))
    jl(run.path/'FIXED_PAIR_EFFECTS.jsonl',effects)
    del coco,predictions;gc.collect();return effects


def main():
    global MODEL
    p=argparse.ArgumentParser();p.add_argument('--training-run',required=True);a=p.parse_args();training=STUDY/'runs'/a.training_run
    require(read_json(training/'run.json')['execution_status']=='completed','training incomplete')
    audit=read_json(STUDY/('AUDIT_'+a.training_run+'.json'));require(audit['status']=='passed','independent training audit required')
    arm=read_json(training/'EVAL_CONTRACT.json')['experiment_arm']
    completed={read_json(p)['experiment_arm'] for p in (STUDY/'runs').glob('RUN_ALIGNED_TRAIN_*/EVAL_CONTRACT.json') if read_json(p.parent/'run.json')['execution_status']=='completed'}
    require(completed=={'R','REP'},'both fixed arms must finish before any evaluation')
    with ExperimentRun('EVAL_'+arm,'full',training) as run:
        expected,source_contract,panel,environment=inspect();import torch
        train_contract=read_json(training/'EVAL_CONTRACT.json');checkpoint=training/'last.pt';checkpoint_sha=sha256(checkpoint)
        require(checkpoint_sha==audit['checkpoint_sha256']==read_json(training/'SUMMARY.json')['checkpoint_sha256'],'checkpoint changed')
        require(train_contract['method_id']=='E4-ALIGNED-v1' and train_contract['expected_optimizer_steps']==1500,'wrong training contract')
        MODEL=CoefficientAdapter('B').cuda();saved=torch.load(checkpoint,map_location='cpu',weights_only=True);MODEL.load_state_dict(saved['state_dict'],strict=True);MODEL.eval();del saved
        ids=read_json(PARENT/'EVALUATION_IMAGES.json');require(len(ids)==1314,'evaluation images changed')
        contract=dict(method_id='E4-ALIGNED-v1',scope='full',experiment_arm=arm,image_ids=ids,source_contract=source_contract,
            checkpoint_sha256=checkpoint_sha,training_run=a.training_run,training_contract_sha256=sha256(training/'EVAL_CONTRACT.json'),
            original_contract_sha256=sha256(PARENT/'EVAL_CONTRACT.json'),native_profile_sha256=source_contract['native_profile_sha256'],
            input_hashes=expected,environment=environment,original_gates=source_contract['gates'],bootstrap=source_contract['bootstrap'],
            damage_panel_sha256=sha256(PARENT/'DAMAGE_PANEL.json'),damage_panel_reused=True,fresh_independent_95_coverage_guarantee=False,
            new_training_steps=0,base_model_forward_calls=0,wall_seconds=run.wall_seconds)
        run.freeze(contract);require(read_json(training/'SOURCE_MANIFEST.json')['files']==run.sources,'code changed since training')
        shutil.copyfile(PARENT/'DAMAGE_PANEL.json',run.path/'DAMAGE_PANEL.json');write_new(run.path/'EVALUATION_IMAGES.json',ids)
        write_new(run.path/'PANEL_REUSE.json',dict(original_panel_sha256=contract['damage_panel_sha256'],reuse_after_outcomes_seen=True,
            original_contract_sha256=panel['inference_contract_sha256'],new_contract_sha256=sha256(run.path/'EVAL_CONTRACT.json')))
        cache=FrozenCache(BASE/'trace',ids);run.update(phase='native_replay_and_adapter_export')
        indices,actions,coverage=export(run,cache);effects=fixed_pairs(run,cache,ids,indices,actions);run.update(phase='complete_COCO_evaluation')
        with (run.path/'evaluation.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
            ledger,ap=evaluate_ledger(GT,read_json(run.path/'ALIGNED_predictions.json'),ids,sha256(run.path/'ALIGNED_predictions.json'))
        baseline=read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl');wanted=validate_ledger(ledger,baseline,ids,indices,cache.manifest['predictions_sha256'])
        jl(run.path/'METHOD_GT_LEDGER.jsonl',ledger);jl(run.path/'BASELINE_SUBSET_GT_LEDGER.jsonl',wanted.values())
        require(len(effects)==2925 and sum(r['population']=='T' for r in effects)==655,'fixed denominator changed')
        target=bootstrap(image_effects(effects));overall=bootstrap(image_effects(effects,target_only=False));require(target['images']==257 and overall['images']==508,'image denominator changed')
        damage=score_damage(panel,baseline,[r for r in ledger if r['image_id'] in set(panel['image_ids'])]);damage['coverage_interpretation']='nominal original formula, exposed development panel'
        status,gates=adjudicate(target,overall,damage);keyed={(r['image_id'],r['annotation_id']):r for r in ledger};targets=[r for r in effects if r['population']=='T']
        coverage.update(dev_selected=sum(r['applied'] for r in effects),target_selected=sum(r['applied'] for r in targets),
            target_task_repaired=sum(keyed[r['image_id'],r['annotation_id']]['task_mask75']=='success' for r in targets),
            target_fixed_detection_reaches075=sum(r['same_output_reaches_075'] for r in targets))
        cache.verify();again=inspect();require(again==(expected,source_contract,panel,environment),'configuration drift')
        require(sha256(checkpoint)==checkpoint_sha and sha256(run.path/'DAMAGE_PANEL.json')==contract['damage_panel_sha256'],'checkpoint/panel changed')
        import psutil
        mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss);require(peak<=4*1024**3,'RSS exceeded');run.check_budget()
        require(sum(p.stat().st_size for p in run.path.rglob('*') if p.is_file())<=1024**3,'artifact budget exceeded')
        write_new(run.path/'CONFIG_AUDIT.json',dict(status='passed',peak_rss_bytes=peak,cache_images_verified_before_after=len(ids),native_zero_delta_replayed_rows=len(indices),
            new_training_steps=0,base_model_forward_calls=0,contract_sha256=sha256(run.path/'EVAL_CONTRACT.json'),checks=dict(inputs_sources_environment_unchanged=True,
            checkpoint_unchanged=True,native_RLE_exact=True,all_nonmask_fields_and_order_equal=True,unselected_rows_identical=True,
            independent_bbox_segm_matching=True,same_bbox_match_identity=True,original_denominators_and_gate=True,panel_bytes_unchanged=True)))
        run.finish(dict(gates=gates,target=target,all_bbox75=overall,damage=damage,coverage=coverage,secondary_subset_AP=ap,
            scientific_scope='E4 fixed loss arm on exposed development data'),status=status,scientific='E4_development_'+status,
            next_action='Close E4 and compare the completed four-experiment evidence; no automatic S3 or scale-up')
        print('E4_DECISION',status,json.dumps(dict(gates=gates,target=target,all_bbox75=overall,damage=damage)),flush=True)


if __name__=='__main__':main()


