"""Complementary official box AP for the same four frozen prediction-slot arms."""
import argparse,contextlib,io,json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from candidate_lineage_probe import ROOT,ARMS,read,need,save_csv,write_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    ids=json.loads((out/'protocol.json').read_text())['images']
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    cats=sorted(gt.cats);pred={arm:[] for arm in ARMS}
    for iid in ids:
        with np.load(ROOT/'diagnostics/structure_main300_20260911/raw'/f'{iid}.npz') as raw, np.load(out/'images'/f'{iid}.npz') as probe:
            source=probe['original_sources'];alt=probe['alternative_sources']
            scores=probe['source_scores'];classes=probe['source_classes']
            for arm in ARMS:
                boxes=raw['boxes_original'][alt if arm in ['box_only','both'] else source].copy()
                boxes[:,2:]-=boxes[:,:2]
                pred[arm].extend(dict(image_id=iid,category_id=cats[int(classes[j])],bbox=list(map(float,b)),score=float(scores[j])) for j,b in zip(source,boxes))
    prior={int(r['annotation_id']):r['bbox75']=='True' for r in read(ROOT/'diagnostics/structure_main300_20260911/instances_same_iou.csv') if int(r['image_id']) in ids}
    rows=[]
    for arm in ARMS:
        with contextlib.redirect_stdout(io.StringIO()):
            dt=gt.loadRes(pred[arm]);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
        if arm=='original':
            t=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.75))[0]);hits={}
            for r in ev.evalImgs:
                if r is None or r['aRng']!=[0,1e10] or r['maxDet']!=100:continue
                hits.update({int(aid):bool(r['gtMatches'][t,j]) for j,aid in enumerate(r['gtIds']) if not r['gtIgnore'][j]})
            need(hits==prior,'Original official bbox75 replay fails')
        rows.append(dict(arm=arm,box_ap=float(ev.stats[0]),box_ap50=float(ev.stats[1]),box_ap75=float(ev.stats[2]),predictions=len(pred[arm])))
    need(rows[0]['box_ap']==rows[1]['box_ap'],'Coefficient edit changed bbox AP')
    save_csv(out/'box_task_summary.csv',rows)
    write_json(out/'BOX_EVAL_COMPLETE.json',dict(status='COMPLETE',script_sha256=sha(__file__),
        results_sha256=sha(out/'box_task_summary.csv'),original_official_box75_exact=True,
        note='Original prediction slots/classes/scores. The four main mask arms all retained 43146 nonempty predictions, so bbox pool is identical.'))
    print(json.dumps(rows))


if __name__=='__main__':main()
