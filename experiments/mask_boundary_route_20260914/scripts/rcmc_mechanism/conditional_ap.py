"""GT-group conditional mask replacement; standard COCOeval on ALL original GT."""
from pathlib import Path
import argparse
import contextlib
import io
import gc
import json
import time
from common import setup, atomic, load_slots, load_predictions, progress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--response-predictions',required=True);args=parser.parse_args();p,out=setup(args.protocol,args.output)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco=COCO(p['annotations']);rows=load_slots(p)
    sets={'target_only':set(),'successful_only':set(),'other_failure_only':set(),'matched_failures_only':set()}
    for r in rows:
        key=(r['image_id'],r['candidate_index'])
        if r['group']=='target_failure':sets['target_only'].add(key)
        if r['baseline_iou']>=.75:sets['successful_only'].add(key)
        else:
            sets['matched_failures_only'].add(key)
            if r['group']!='target_failure':sets['other_failure_only'].add(key)
    base,basekeys=load_predictions(Path(p['bank'])/'predictions_official_zero.json')
    response,reskeys=load_predictions(args.response_predictions)
    assert basekeys.keys()==reskeys.keys(),'Final gate must preserve baseline nonempty exports'
    for key,r in basekeys.items():
        s=reskeys[key];assert r['category_id']==s['category_id'] and r['score']==s['score']
    ids=json.loads((Path(p['bank'])/'image_ids.json').read_text());metrics={};start=time.perf_counter()
    for name in ('baseline','target_only','successful_only','other_failure_only','matched_failures_only','response'):
        progress(out,'cocoeval',variant=name,elapsed_seconds=time.perf_counter()-start)
        changed=0;items=[]
        for r in base:
            key=(r['image_id'],r['candidate_index'])
            use=name=='response' or key in sets.get(name,set())
            s=reskeys[key] if use else r
            changed+=int(use and s['segmentation']!=r['segmentation'])
            items.append({k:s[k] for k in ('image_id','category_id','score','segmentation')})
        with contextlib.redirect_stdout(io.StringIO()) as log:
            detections=coco.loadRes(items);ev=COCOeval(coco,detections,'segm');ev.params.imgIds=ids
            ev.evaluate();ev.accumulate();ev.summarize()
        vals=[float(v) for v in ev.stats];metrics[name]={'stats':vals,'ap':100*vals[0],
            'delta_ap_points':100*(vals[0]-metrics['baseline']['stats'][0]) if name!='baseline' else 0,
            'changed_masks':changed,'selected_slots':len(sets.get(name,())) if name not in ('baseline','response') else len(base) if name=='response' else 0}
        (out/f'cocoeval_{name}.log').write_text(log.getvalue(),encoding='utf-8');atomic(out/'metrics_partial.json',metrics)
        del ev,detections,items;gc.collect()
        if name=='baseline':assert abs(vals[0]-.4337882998311341)<1e-8,('baseline AP mismatch',vals[0])
        if name=='response':assert abs(vals[0]-.4393954420856892)<1e-8,('response AP mismatch',vals[0])
    atomic(out/'SUMMARY.json',{'metrics':metrics,'images':len(ids),'elapsed_seconds':time.perf_counter()-start,
        'diagnostic':'GT-defined baseline fixed-slot groups; actual RCMC mask applied only within group. All original GT and other predictions retained.',
        'limitations':['Not AP computed on a deleted-GT subset.','Not a deployable GT-group selector.','Group AP contributions are nonadditive.','One-to-one 4500-image confirmation; previously explored validation set.']})
    progress(out,'completed',metrics=metrics)


if __name__=='__main__':main()
