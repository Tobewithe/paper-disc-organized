"""Score one frozen candidate-bank variant with original COCO annotations."""
import argparse,contextlib,io,json,os,time,gc
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline-reference',default=str(Path(__file__).resolve().parents[1]/
        'runs/RUN_2ae67d6556a14848bceb5783a72e5790/predictions_official_zero.json'))
    a=p.parse_args();source=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'SUMMARY.json').exists():raise RuntimeError('Use a fresh Run')
    start=time.perf_counter();ids=json.loads((source/'image_ids.json').read_text());coco=COCO(a.annotations)
    parity=None
    if a.baseline_reference:
        new=json.loads((source/'predictions_official_zero.json').read_text())
        old=json.loads(Path(a.baseline_reference).read_text())
        assert new==old,'Rerun baseline changed; do not combine AP comparisons'
        parity=dict(exact_exported_prediction_match=True,predictions=len(new))
        del new,old;gc.collect()
    records=json.loads((source/f'predictions_{a.variant}.json').read_text())
    canonical=[{k:r[k] for k in ('image_id','category_id','score','segmentation')} for r in records]
    with contextlib.redirect_stdout(io.StringIO()) as log:
        result=coco.loadRes(canonical);ev=COCOeval(coco,result,'segm');ev.params.imgIds=ids
        ev.evaluate();ev.accumulate();ev.summarize()
    (out/'cocoeval.log').write_text(log.getvalue(),encoding='utf-8')
    summary=dict(run_id=os.environ.get('RESEARCH_RUN_ID'),source_run=source.name,variant=a.variant,
        images=len(ids),prediction_count=len(records),metrics=[float(v) for v in ev.stats],baseline_parity=parity,elapsed_seconds=time.perf_counter()-start)
    tmp=out/'SUMMARY.json.tmp';tmp.write_text(json.dumps(summary,indent=2),encoding='utf-8');os.replace(tmp,out/'SUMMARY.json')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
