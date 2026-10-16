"""Independent teacher selection, native RLE and exact target reconstruction; no solves."""
import argparse
import time
from common import *
from targets import mask_hash

def verify(path):
    data=inspect_inputs()
    import torch
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    from adapter_data import FrozenCache, training_join
    from native_output_bridge import decode_native, encode_rle
    from s1_geometry import full_rle_iou, predicted_roi
    manifest=read_json(path/'TEACHER_MANIFEST.json')
    require(manifest['spec']==SPEC,'teacher policy changed')
    require(sha256(path/'TEACHER_ROWS.jsonl')==manifest['rows_sha256'],'teacher rows drift')
    expected=data['split']['fit_ids'] if manifest['scope']=='full' else data['split']['fit_ids'][:8]
    require(set(map(int,manifest['images']))==set(expected),'teacher image coverage differs')
    rows=read_jsonl(path/'TEACHER_ROWS.jsonl')
    by_image={i:[] for i in expected}
    for r in rows:
        by_image[r['image_id']].append(r)
    source=FrozenCache(CACHE/'trace',expected)
    coco=COCO(str(TRAIN_GT))
    ledger={(r['image_id'],r['annotation_id']):r for r in read_jsonl(FIT/'BASELINE_GT_LEDGER.jsonl')}
    started=time.monotonic()
    checked=accepted_count=0
    for iid in expected:
        require(time.monotonic()-started<3600,'teacher independent verification wall budget exceeded')
        cache,predictions=source.load(iid)
        entry=manifest['images'][str(iid)]
        require(sha256(path/entry['path'])==entry['sha256'],'teacher package hash mismatch')
        pack=torch.load(path/entry['path'],map_location='cpu',weights_only=True)
        join=training_join(predictions,coco.imgToAnns.get(iid,[]))
        records=by_image[iid]
        require(len(records)==len(join) and {r['rank']:r['annotation_id'] for r in records}==join==pack['join'],'teacher pair coverage mismatch')
        require(pack['image_id']==iid and pack['prediction_indices']==cache['prediction_indices'],'package identity mismatch')
        require(pack['cache_sha256']==source.records[iid]['cache_sha256'],'package source cache differs')
        coeff=cache['c0'].clone()
        zero=decode_native(cache)
        require(all(encode_rle(zero[j])==v['segmentation'] for j,v in enumerate(predictions)),'independent zero replay differs')
        del zero
        for r in records:
            rank=r['rank']
            gt=ledger[(iid,r['annotation_id'])]
            roi=predicted_roi(cache,rank)
            baseline_iou=full_rle_iou(predictions[rank]['segmentation'],coco.annToRLE(coco.anns[r['annotation_id']]))
            eligible=gt['task_mask75']!='success' and baseline_iou<.75 and bool(roi.any())
            require(eligible==r['attempted'] and baseline_iou==r['baseline_iou'],'eligibility/baseline differs')
            require(gt['task_mask75']==r['baseline_task_mask75'],'original success status differs')
            require(r['global_index']==cache['prediction_indices'][rank],'teacher detection index mismatch')
            require(r['output_detection_id']==source.manifest['predictions_sha256']+':'+str(r['global_index']),'teacher output ID mismatch')
            require(r['native_area']==int(masks.area(predictions[rank]['segmentation'])),'native area differs')
            require(r['roi_pixels']==int(roi.sum()),'ROI denominator differs')
            if eligible:
                coeff[rank]=torch.tensor(r['coefficients_fp32'],dtype=coeff.dtype)
                require(r['solver_status']==('converged' if r['gradient_inf_norm']<=1e-6 else 'budget_exhausted'),'solver convergence status differs')
        changed=decode_native(cache,coeff)
        # Independent explicit matrix expression, no call to producer's loss_grids.
        hard=(coeff.cuda().float() @ cache['proto'].cuda().float().flatten(1)).reshape(len(coeff),*cache['proto'].shape[-2:]).gt(0).to(torch.uint8).cpu()
        expected_targets=set()
        for r in records:
            rank=r['rank']
            if r['attempted']:
                native_rle=encode_rle(changed[rank])
                iou=full_rle_iou(native_rle,coco.annToRLE(coco.anns[r['annotation_id']]))
                fg=int(hard[rank][predicted_roi(cache,rank)].sum())
                selected=r['solver_status']=='converged' and iou-r['baseline_iou']>=.01 and fg>0
                require(iou==r['oracle_iou'] and native_rle==r['oracle_rle'] and fg==r['teacher_roi_foreground'],'oracle native replay differs')
                require(selected==r['accepted'],'teacher acceptance differs')
                if selected:
                    expected_targets.add(rank)
                    require(torch.equal(pack['targets'][rank],hard[rank]),'functional target replay differs')
                    require(mask_hash(hard[rank])==pack['target_hashes'][rank]==r['target_sha256'],'functional target hash differs')
            else:
                require(not r['accepted'],'noneligible target replaced')
        require(set(pack['targets'])==set(pack['target_hashes'])==expected_targets,'extra or missing teacher target')
        accepted_count+=len(expected_targets)
        checked+=1
        if checked%100==0 or checked==len(expected):
            print('VERIFY_TEACHER',checked,'/',len(expected),flush=True)
        del cache,pack,coeff,changed,hard,predictions
    source.verify()
    verify_after(data)
    gate=teacher_gate(rows)
    require(gate==read_json(path/'SUMMARY.json')['teacher_gate'],'teacher Gate recomputation differs')
    report=dict(status='passed',at=now(),teacher_run_id=path.name,images=checked,joined_rows=len(rows),
        accepted=accepted_count,all_zero_delta_RLE_exact=True,all_solver_native_RLE_exact=True,
        all_binary_functional_targets_exact=True,all_selection_and_success_status_recomputed=True,
        teacher_gate=gate,manifest_sha256=sha256(path/'TEACHER_MANIFEST.json'),
        source_manifest_sha256=sha256(path/'SOURCE_MANIFEST.json'),new_solver_calls=0,new_adapter_steps=0)
    write_new(path/'INDEPENDENT_TEACHER.json',report)
    print('INDEPENDENT_TEACHER_PASSED',path.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);verify(p.parse_args().run)
