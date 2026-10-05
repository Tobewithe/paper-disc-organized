"""Finish interrupted COCO scoring from immutable exported predictions, without inference."""
import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import time


def atomic(path, data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source,out=Path(args.input),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'SUMMARY.json').exists(): raise RuntimeError('Use a fresh Run')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    start=time.perf_counter()
    ids=json.loads((source/'image_ids.json').read_text())
    assert len(ids)==4500 and len(set(ids))==4500
    coco=COCO(args.annotations)
    assert ids==sorted(coco.getImgIds())[500:5000]
    metrics=json.loads((source/'metrics_partial.json').read_text())
    signatures={}
    counts={}
    for variant in ('official_zero','smooth_gated','filter_only','boundary_only'):
        print('Reading '+variant,flush=True)
        atomic(out/'progress.json',dict(stage='verify_exports',variant=variant))
        data=json.loads((source/f'predictions_{variant}.json').read_text())
        identities={}
        for row in data:
            assert set(row)=={'image_id','candidate_index','category_id','score','segmentation'}
            key=(row['image_id'],row['candidate_index'])
            assert key not in identities
            rle=row['segmentation']
            identities[key]=(row['category_id'],row['score'],tuple(rle['size']),hashlib.sha256(rle['counts'].encode('ascii')).digest())
        signatures[variant]=identities
        counts[variant]=dict(exported_nonempty=len(data))
        if variant=='filter_only':
            assert identities.keys()==signatures['smooth_gated'].keys()
            assert all(value==signatures['official_zero'][key] for key,value in identities.items())
        if variant=='boundary_only':
            assert identities.keys()==signatures['official_zero'].keys()
            assert all(value==signatures['smooth_gated'].get(key,signatures['official_zero'][key]) for key,value in identities.items())
        if variant not in metrics:
            print('COCOeval '+variant,flush=True)
            atomic(out/'progress.json',dict(stage='cocoeval',variant=variant))
            with contextlib.redirect_stdout(io.StringIO()) as logs:
                result=coco.loadRes(data)
                evaluator=COCOeval(coco,result,'segm')
                evaluator.params.imgIds=ids
                evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            metrics[variant]=[float(v) for v in evaluator.stats]
            (out/f'cocoeval_{variant}.log').write_text(logs.getvalue(),encoding='utf-8')
            del result,evaluator
        del data
        gc.collect()
    candidate_count=empty0=empty1=0
    with (source/'candidate_records.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            candidate_count+=1
            empty0+=int(row['baseline_empty'])
            empty1+=int(row['smooth_empty'])
    assert counts['official_zero']['exported_nonempty']==candidate_count-empty0
    assert counts['smooth_gated']['exported_nonempty']==candidate_count-empty1
    total_gt=sum(not a.get('iscrowd',0) for iid in ids for a in coco.imgToAnns[iid])
    transitions={key:dict(repaired=0,damaged=0,matched=0) for key in metrics}
    with (source/'instance_records.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            before=float(row['baseline_iou'])>=.75
            after=float(row['iou'])>=.75
            stats=transitions[row['variant']]
            stats['matched']+=1
            stats['repaired']+=int(not before and after)
            stats['damaged']+=int(before and not after)
    ap={key:metrics[key][0] for key in metrics}
    payload=dict(run_id=os.environ.get('RESEARCH_RUN_ID'),study_id='STUDY_8fb3468ebb704682a2225ebed0e16206',
        source_run_id=source.name,source_execution_status='observed interrupted: no process, stale heartbeat, exit code unknown',
        recovery='reuse three completed COCO evaluations; compute missing boundary_only evaluation from saved predictions; no model inference',
        branch='one2one',images=len(ids),start=500,ordinary_gt=total_gt,metrics=metrics,
        delta_vs_official={key:[v-b for v,b in zip(values,metrics['official_zero'])] for key,values in metrics.items()},
        counts=counts,all_candidates=candidate_count,baseline_empty=empty0,smooth_empty=empty1,
        factorial_identity_checks='all exported candidate IDs, categories, exact scores and mask RLEs verified for both counterfactuals',
        fixed_slot_transitions=transitions,
        factorial_effects_ap_points=dict(
            boundary_at_original_candidates=100*(ap['boundary_only']-ap['official_zero']),
            filter_at_original_masks=100*(ap['filter_only']-ap['official_zero']),
            boundary_at_filtered_candidates=100*(ap['smooth_gated']-ap['filter_only']),
            filter_at_calibrated_masks=100*(ap['smooth_gated']-ap['boundary_only']),
            interaction=100*(ap['smooth_gated']-ap['filter_only']-ap['boundary_only']+ap['official_zero'])),
        limitations=['Previously explored COCO val2017, single model','Original job did not finish; its run.json remains untouched',
            'AP point estimates, no AP significance claim','Original three-image parity output was not saved in a completed summary'],
        elapsed_seconds=time.perf_counter()-start)
    atomic(out/'SUMMARY.json',payload)
    atomic(out/'progress.json',dict(stage='completed'))
    print(json.dumps(payload),flush=True)


if __name__=='__main__': main()
