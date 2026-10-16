"""Replay frozen same-detection native/oracle/adapter/gate masks, without training."""
from common import *
import argparse
from collections import Counter, defaultdict
import contextlib
import gc
import psutil


def resources(run):
    run.check_budget()
    require(psutil.Process().memory_info().rss < 6*1024**3, 'RSS limit 6 GiB exceeded')
    require(psutil.virtual_memory().available >= 768*1024**2, 'available RAM below reserve')
    require(shutil.disk_usage(ROOT).free >= 2*1024**3, 'disk reserve below 2 GiB')


def execute(run, ids):
    import torch
    from adapter_data import FrozenCache
    from adapter_model import CoefficientAdapter
    from native_output_bridge import decode_native, encode_rle, native_environment
    from s1_geometry import iter_prediction_array, full_rle_iou
    from pycocotools import mask as masks
    from pycocotools.coco import COCO

    expected, profile, environment = inspect_inputs()
    preparation = read_json(STUDY/'PREPARATION.json')
    require(environment == preparation['environment'], 'prepared environment drift')
    run.freeze(dict(environment=environment, native_profile=profile, image_ids=ids,
        scope=run.meta['scope'], parent_S1=S1.name, parent_S3=S3.name,
        checkpoint=dict(run_id=TRAIN.name, sha256=sha256(TRAIN/'last.pt'), arm='B', phase='eval:0'),
        original_panel_sha256=sha256(S1/'TARGET_PANEL.json'), input_files=expected['files'],
        gate_native_binary_area_gte=9216, mask_success_iou_gte=.75,
        primary_denominator='all original S1 pairs; primary decomposition original oracle-positive 122',
        actual_geometry='original S0 image cache; no new YOLO forward or batching',
        new_training_steps=0, oracle_optimization_calls=0, base_model_forward_calls=0))
    for filename, source in [('SOURCE_TARGET_PANEL.json',S1/'TARGET_PANEL.json'),
                             ('SOURCE_ORACLE_RESULTS.jsonl',S1/'ORACLE_RESULTS.jsonl')]:
        shutil.copyfile(source, run.path/filename)
    panel = [r for r in read_json(S1/'TARGET_PANEL.json')['targets'] if r['image_id'] in set(ids)]
    identity = lambda r: (r['image_id'],r['annotation_id'],r['output_detection_id'])
    require(len({identity(r) for r in panel}) == len(panel), 'duplicate target identity')
    require(len({r['output_detection_id'] for r in panel}) == len(panel), 'duplicate detection in original panel')
    oracle = {identity(r):r for r in read_jsonl(S1/'ORACLE_RESULTS.jsonl')}
    require(set(oracle) == {identity(r) for r in read_json(S1/'TARGET_PANEL.json')['targets']}, 'oracle panel differs')
    ledger = {(r['image_id'],r['annotation_id']):r for r in read_jsonl(BASE/'ledger/PER_GT_FAILURE.jsonl')}
    census = {(r['image_id'],r['annotation_id']):r for r in read_jsonl(S3/'GT_CENSUS.jsonl')}
    source = FrozenCache(BASE/'trace', ids)
    wanted_indices = {i for r in source.records.values() for i in r['prediction_indices']}
    original_mapping = read_json(OLD/'B_ROW_MAPPING.json')['original_global_indices']
    old = {i:r for i,r in zip(original_mapping, iter_prediction_array(OLD/'B_predictions.json'), strict=True) if i in wanted_indices}
    previous = {i:r for i,r in enumerate(iter_prediction_array(S3/'predictions_mixed.json')) if i in wanted_indices}
    require(set(previous)==wanted_indices, 'S3 row coverage missing')
    previous_delta_all = torch.load(S3/'DELTA.pt', map_location='cpu', weights_only=True)
    previous_delta = {i:previous_delta_all[i] for i in ids if i in previous_delta_all}
    del previous_delta_all
    model = CoefficientAdapter('B').cuda().eval()
    checkpoint = torch.load(TRAIN/'last.pt', map_location='cpu', weights_only=True)
    require(checkpoint['arm']=='B' and checkpoint['completed_epochs']==3, 'wrong checkpoint metadata')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.requires_grad_(False)
    model_hash = tensor_state_hash(model)
    with (run.path/'coco_load.log').open('x',encoding='utf8') as log, contextlib.redirect_stdout(log):
        coco = COCO(str(GT))
    by_image = defaultdict(list)
    for target in panel:
        by_image[target['image_id']].append(target)
    write_new(run.path/'INPUT_CACHE_RECORDS.json', [source.records[i] for i in ids])
    write_new(run.path/'SOURCE_GT_ANNOTATIONS.json', [coco.anns[r['annotation_id']] for r in panel])
    write_new(run.path/'SOURCE_LEDGER_ROWS.json', [ledger[(r['image_id'],r['annotation_id'])] for r in panel])
    write_new(run.path/'SOURCE_S3_CENSUS_ROWS.json', [census[(r['image_id'],r['annotation_id'])] for r in panel])
    results, evidence, counters = {}, {}, Counter()
    with torch.inference_mode():
        for number, image_id in enumerate(ids):
            resources(run)
            cache, native_rows = source.load(image_id)
            indices = cache['prediction_indices']
            ranks = {index:rank for rank,index in enumerate(indices)}
            require(len(indices)==len(native_rows)==len(cache['c0']), 'cache row coverage')
            targets = by_image[image_id]
            zero = decode_native(cache)
            for rank,row in enumerate(native_rows):
                require(encode_rle(zero[rank]) == row['segmentation'], 'zero-delta RLE mismatch')
                counters['zero_delta_exact_rows'] += 1
            del zero
            torch.cuda.empty_cache()
            coefficients = cache['c0'].clone()
            for target in targets:
                record = oracle[identity(target)]
                index = int(target['output_detection_id'].rsplit(':',1)[1])
                rank = ranks[index]
                require(rank==record['rank'], 'original oracle rank changed')
                require(record['cache_sha256']==source.records[image_id]['cache_sha256'], 'oracle cache changed')
                coefficients[rank] = torch.tensor(record['coefficients_fp32'], dtype=coefficients.dtype)
            decoded_oracle = decode_native(cache, coefficients)
            oracle_masks = {}
            for target in targets:
                record = oracle[identity(target)]
                replay = encode_rle(decoded_oracle[record['rank']])
                require(replay==record['oracle_rle'], 'saved oracle coefficient replay mismatch')
                oracle_masks[identity(target)] = replay
                counters['oracle_exact_pairs'] += 1
            del decoded_oracle, coefficients
            torch.cuda.empty_cache()
            delta = model(cache, 'eval:0')
            require(torch.isfinite(delta).all().item(), 'nonfinite adapter delta')
            if image_id in previous_delta:
                require(torch.equal(delta.cpu(), previous_delta[image_id]), 'S3 delta differs bitwise')
                counters['S3_delta_exact_images'] += 1
                counters['S3_delta_exact_rows'] += len(indices)
            else:
                require(all(i in old for i in indices), 'no historical ungated comparator')
            adapted = decode_native(cache, cache['c0'].to(delta)+delta)
            ungated = {}
            for rank,(index,row) in enumerate(zip(indices,native_rows,strict=True)):
                rle = encode_rle(adapted[rank])
                area = int(masks.area(row['segmentation']))
                if index in old:
                    require(old[index] == {**row,'segmentation':rle}, 'old S2 ungated prediction differs')
                    counters['S2_ungated_exact_rows'] += 1
                mixed = {**row,'segmentation':rle if area >= 9216 else row['segmentation']}
                require(mixed == previous[index], 'S3 mixed output or immutable field differs')
                counters['S3_mixed_exact_rows'] += 1
                ungated[index] = rle
            del adapted, delta
            for target in targets:
                key = identity(target)
                record = oracle[key]
                index = int(target['output_detection_id'].rsplit(':',1)[1])
                rank = ranks[index]
                native = native_rows[rank]
                ann = coco.anns[target['annotation_id']]
                prior = ledger[(image_id,target['annotation_id'])]
                task = census[(image_id,target['annotation_id'])]
                require(native['image_id']==ann['image_id']==image_id, 'cross-image GT or prediction')
                require(native['category_id']==ann['category_id']==target['category_id'], 'class mismatch')
                require(not ann.get('iscrowd',0) and not ann.get('ignore',0) and not prior['is_ignored'], 'ignored GT')
                require(prior['task_mask75']=='failure' and prior['bbox75_match_output_id']==target['output_detection_id'], 'Box75 identity/Mask75 task mismatch')
                require(not task['ignored'] and not task['baseline_success'], 'S3 task census disagrees with baseline')
                require(target['output_detection_id'].startswith(source.manifest['predictions_sha256']+':'), 'wrong prediction namespace')
                box_iou = float(masks.iou([native['bbox']],[ann['bbox']],[0])[0,0])
                require(box_iou >= .75 and abs(box_iou-prior['box_iou_of_bbox_match']) < 1e-12, 'fixed Box75 identity changed')
                gt = coco.annToRLE(ann)
                if isinstance(gt['counts'],bytes):
                    gt = {**gt,'counts':gt['counts'].decode('ascii')}
                area = int(masks.area(native['segmentation']))
                enabled = area >= 9216
                variants = dict(native=native['segmentation'],oracle=oracle_masks[key],ungated=ungated[index],
                                gated=previous[index]['segmentation'])
                ious = {k:full_rle_iou(v,gt) for k,v in variants.items()}
                require(ious['native'] < .75 and abs(ious['native']-target['baseline_mask_iou']) < 1e-12, 'native panel IoU differs')
                require(abs(ious['native']-prior['mask_iou_of_bbox_match']) < 1e-12, 'native ledger IoU differs')
                require(abs(ious['oracle']-record['oracle_mask_iou']) < 1e-12, 'archived oracle IoU differs')
                require(variants['gated']==variants['ungated' if enabled else 'native'], 'gate RLE logic differs')
                found, repaired = ious['oracle'] >= .75, ious['ungated'] >= .75
                label = ('L' if not repaired else 'R' if enabled else 'G') if found else 'O_unfound'
                row = dict(**target, global_index=index, rank=rank, raw_id=int(cache['raw_ids'][rank]),
                    box_iou=box_iou, native_area=area, native_area_group='S' if area<1024 else 'M' if area<9216 else 'L',
                    gate_enabled=enabled, oracle_found=found, ungated_repaired=repaired,
                    gated_repaired=ious['gated']>=.75, decomposition=label,
                    **{k+'_iou':v for k,v in ious.items()},
                    s3_task_success=task['method_success'], s3_method_dt_id=task['method_dt_id'],
                    ungated_COCO_task_success=None)
                results[key] = row
                evidence[key] = dict(image_id=image_id, annotation_id=ann['id'], output_detection_id=target['output_detection_id'],
                    global_index=index, native_prediction=native, GT_rle=gt, **{k+'_rle':v for k,v in variants.items()})
            counters['images'] += 1
            if (number+1)%20==0 or number+1==len(ids):
                run.update(phase='paired_replay', completed_images=number+1, total_images=len(ids))
                print('PAIRED_REPLAY',number+1,'/',len(ids),flush=True)
            del cache, native_rows, ungated
            torch.cuda.empty_cache()
    ordered = [results[identity(r)] for r in panel]
    jsonl(run.path/'PAIR_RESULTS.jsonl', ordered)
    jsonl(run.path/'PAIRED_MASKS.jsonl', [evidence[identity(r)] for r in panel])
    summary = summarize(ordered)
    if run.meta['scope']=='full':
        require(summary['pairs']==256 and summary['images']==243 and summary['oracle_positive']==122, 'formal denominator changed')
    resources(run)
    source.verify()
    for record in source.records.values():
        image = ROOT/'assets/datasets/coco_native_s0_20261013/images/val2017'/f"{record['image_id']:012d}.jpg"
        require(sha256(image)==record['image_sha256'], 'image changed during execution')
    for relative,digest in expected['files'].items():
        require(sha256(ROOT/relative)==digest, 'source input changed during evaluation')
    require(tensor_state_hash(model)==model_hash, 'adapter state modified')
    require(native_environment()==environment, 'numeric runtime changed')
    write_new(run.path/'CONFIG_AUDIT.json', dict(status='passed', before_after_input_hashes=True,
        input_files=len(expected['files']), cache_and_image_count=len(ids), actual_runtime_equal_original=True,
        checkpoint_sha256=sha256(TRAIN/'last.pt'), model_state_sha256=model_hash,
        base_weight_sha256=sha256(WEIGHT), profile_sha256=sha256(PROFILE),
        gate_native_binary_area_gte=9216, new_training_steps=0, base_model_forward_calls=0,
        oracle_optimization_calls=0, exact_replay=dict(counters),
        all_fixed_detection_fields_unchanged=True, parent_S3_gate_reinterpreted=False))
    summary['engineering'] = dict(counters)
    run.finish(summary, scientific='fixed_panel_diagnostic_complete' if run.meta['scope']=='full' else 'engineering_smoke_only',
               next_action=summary['priority_direction'] if run.meta['scope']=='full' else 'Run frozen full panel after smoke passes')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope',choices=('smoke','full'),required=True)
    p.add_argument('--smoke-run',type=Path)
    a=p.parse_args()
    prep=read_json(STUDY/'PREPARATION.json')
    if a.scope=='full':
        require(a.smoke_run is not None,'smoke Run required')
        require(read_json(a.smoke_run/'run.json')['execution_status']=='completed','smoke incomplete')
        require(read_json(a.smoke_run/'CONFIG_AUDIT.json')['status']=='passed','smoke configuration failed')
        require(read_json(a.smoke_run/'DECISION.json')['status']=='passed','smoke failed')
    ids=prep['smoke_image_ids'] if a.scope=='smoke' else prep['image_ids']
    with AuditRun(a.scope, a.smoke_run or S1) as run:
        study=read_json(STUDY/'study.json')
        study.update(execution_status='running',active_run_id=run.run_id,**{a.scope+'_run_id':run.run_id})
        save(STUDY/'study.json',study)
        execute(run,ids)
    study=read_json(STUDY/'study.json')
    study.update(active_run_id=None,execution_status='smoke_completed' if a.scope=='smoke' else 'completed_pending_independent_verification')
    save(STUDY/'study.json',study)


if __name__=='__main__':
    main()
