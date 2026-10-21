"""Independent arithmetic and COCO matching; no model or producer-statistic imports."""
from pathlib import Path
import argparse,collections,contextlib,hashlib,json,time
import numpy as np
from pycocotools import mask as masks
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def read(p): return json.loads(p.read_text(encoding='utf8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def iou(a,b):
    a=masks.decode(a).astype(bool); b=masks.decode(b).astype(bool); u=np.count_nonzero(a|b)
    return np.count_nonzero(a&b)/u if u else 0.

def verify(run):
    started=time.monotonic(); sm=read(run/'SUMMARY.json'); contract=read(run/'EVAL_CONTRACT.json')
    for rel,stored in read(run/'SOURCE_SNAPSHOT_MAP.json').items():
        assert sha(run/stored)==read(run/'SOURCE_MANIFEST.json')['files'][rel],rel
    study=run.parent.parent
    for arm,rid in contract['training_runs'].items():
        p=run.parent/rid
        assert read(p/'run.json')['execution_status']=='completed'
        assert sha(p/'last.pt')==contract['checkpoints'][arm]
        log=rows(p/'training.jsonl'); steps=rows(p/'OPTIMIZER_STEPS.jsonl'); orders=read(p/'IMAGE_ORDERS.json')
        assert len(log)==4800 and len(steps)==1200 and len(orders)==3
        assert [r['image_id'] for r in log]==sum(orders,[])
        assert all(r['step']==i+1 and r['image_ids']==sum(orders,[])[i*4:i*4+4] for i,r in enumerate(steps))
        assert all(np.isfinite(r['loss']) for r in log)
        assert read(p/'ACTUAL_OPTIMIZER.json')==read(p/'EVAL_CONTRACT.json')['optimizer']
    pairs=rows(run/'FIXED_PAIRS.jsonl'); tasks=rows(run/'TASKS.jsonl'); mapping=read(run/'ROW_MAPPING.json')['global_indices']
    pred={k:read(run/f'{k}_predictions.json') for k in ('N','L','U')}
    assert all(len(v)==42167 for v in pred.values()) and len(mapping)==len(set(mapping))==42167
    gi={g:j for j,g in enumerate(mapping)}
    support={r['global_index']:r for r in rows(run/'SUPPORT_RECEIPTS.jsonl')}
    changed={k:0 for k in ('L','U')}
    for j,g in enumerate(mapping):
        n=pred['N'][j]; a=masks.decode(n['segmentation']).astype(bool)
        for k in ('L','U'):
            b=pred[k][j]
            assert {t:v for t,v in n.items() if t!='segmentation'}=={t:v for t,v in b.items() if t!='segmentation'}
            if g not in support: assert n['segmentation']==b['segmentation']
            else:
                band=masks.decode(support[g]['band_rle']).astype(bool)
                edit=a!=masks.decode(b['segmentation']).astype(bool)
                assert not np.any(edit&~band)
                assert int(edit.sum())==support[g]['changed'][k]
                assert int(masks.area(n['segmentation']))>=9216
                changed[k]+=int(edit.sum())
    for r in pairs:
        j=gi[int(r['output_detection_id'].rsplit(':',1)[1])]
        for k in ('N','L','U'): assert abs(iou(pred[k][j]['segmentation'],r['GT_rle'])-r[k])<1e-12
    assert len(pairs)==2246 and sum(r['population']=='T' for r in pairs)==490
    for label,rs in (('target',[r for r in pairs if r['population']=='T']),('overall',pairs)):
        groups=collections.defaultdict(list)
        for r in rs: groups[r['image_id']].append(r)
        ids=sorted(groups); draws=np.random.Generator(np.random.PCG64(1729)).integers(len(ids),size=(2000,len(ids)))
        assert np.array_equal(draws,np.load(run/'BOOTSTRAP.npz')[label])
        for key,result in sm[label].items():
            left,right=key.split('-'); values=np.array([sum(r[left]-r[right] for r in groups[i])/len(groups[i]) for i in ids])
            ci=np.percentile(np.array([np.mean(values[d]) for d in draws]),[2.5,97.5],method='linear')
            assert abs(float(values.mean())-result['mean'])<1e-12 and np.max(np.abs(ci-result['ci95']))<1e-12
    with (run/'independent_coco.log').open('a',encoding='utf8') as log,contextlib.redirect_stdout(log):
        coco=COCO(str(run/'ANNOTATIONS_SUBSET.json')); matches={}
        for key in ('N','L','U'):
            matches[key]={}
            # Full default threshold/area grid, independently select area=all, IoU=.75.
            for kind in ('segm','bbox'):
                ev=COCOeval(coco,coco.loadRes(pred[key]),kind); ev.params.imgIds=sorted(coco.imgs); ev.evaluate()
                ti=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.75))[0]); data={}
                for item in ev.evalImgs:
                    if item is None or list(item['aRng'])!=list(ev.params.areaRng[0]): continue
                    for q,aid in enumerate(item['gtIds']):
                        data[aid]=(bool(item['gtIgnore'][q]),int(item['gtMatches'][ti,q]))
                matches[key][kind]=data
        for k in ('L','U'): assert matches[k]['bbox']==matches['N']['bbox']
        for r in tasks:
            aid=r['annotation_id']
            for k in ('N','L','U'):
                ignored,dt=matches[k]['segm'][aid]
                assert not ignored and bool(dt)==r[k]
        ordinary={aid for aid,(ignored,dt) in matches['N']['segm'].items() if not ignored}
        assert {r['annotation_id'] for r in tasks}==ordinary
    assert sum(r['N'] for r in tasks)==1808
    for k in ('G','L','U'):
        assert sm['census'][k]['damaged']==sum(r['N'] and not r[k] for r in tasks)
        assert sm['census'][k]['recovered']==sum(not r['N'] and r[k] for r in tasks)
    t=sm['target']; a=sm['overall']
    mechanism=t['L-U']['mean']>=.005 and t['L-U']['ci95'][0]>0
    utility=t['L-N']['mean']>=.005 and t['L-N']['ci95'][0]>0 and t['L-G']['ci95'][0]>0 and a['L-N']['mean']>=-.001 and sm['census']['L']['damaged']<=18
    negative=t['L-U']['ci95'][1]<.005 or t['L-N']['ci95'][1]<.005 or t['L-G']['ci95'][1]<=0 or a['L-N']['mean']<-.001 or sm['census']['L']['damaged']>18
    decision='POSITIVE_INTERNAL_SCREEN' if mechanism and utility else 'NEGATIVE_AT_BUDGET' if negative else 'INCONCLUSIVE_AT_BUDGET'
    assert (sm['mechanism_pass'],sm['utility_pass'],sm['decision'])==(mechanism,utility,decision)
    return dict(status='passed',training_arms=2,training_steps_per_arm=1200,heldout_images=400,predictions=42167,fixed_pairs=2246,target=490,native_successes=1808,decision=decision,changed_pixels=changed,outside_changes=0,bootstrap=2000,independent_matching_arms=3,elapsed_seconds=time.monotonic()-started,verifier_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--output',default='INDEPENDENT_LAPTOP.json'); a=p.parse_args()
    result=verify(a.run)
    with (a.run/a.output).open('x',encoding='utf8') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
