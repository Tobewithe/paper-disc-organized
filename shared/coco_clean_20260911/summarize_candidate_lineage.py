"""Official whole-image oracle task evaluation and image-cluster paired estimates."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,gzip,io,json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from frozen_mechanism_probe import ROOT,sha,write_json
from structure_candidate_trace import save_csv
from summarize_relative_ownership import evaluate
from candidate_lineage_probe import ARMS,STAGES,read,need


def boot(rows, value, ids, weights):
    ix={v:j for j,v in enumerate(ids)}
    nums=np.zeros(len(ids));dens=np.zeros(len(ids))
    for r in rows:
        i=ix[int(r['image_id'])];nums[i]+=value(r);dens[i]+=1
    bden=weights@dens
    v=(weights@nums)[bden>0]/bden[bden>0]
    return dict(n=len(rows),difference_pp=100*float(nums.sum()/dens.sum()) if dens.sum() else None,
                ci95_pp=(100*np.quantile(v,[.025,.975])).tolist() if len(v) else None,
                valid_resamples=len(v))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    receipt=json.loads((out/'COMPLETE.json').read_text())
    need(receipt['status']=='COMPLETE','Incomplete inference')
    for name,digest in receipt['hashes'].items():need(sha(out/name)==digest,name)
    protocol=json.loads((out/'protocol.json').read_text());ids=protocol['images']
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    metadata={int(r['annotation_id']):r for r in read(ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv')}
    prior={int(r['annotation_id']):r['official_mask75']=='True' for r in read(out/'gt.csv')}
    tasks=[];allrecords=[];allpairs=[];predhashes={};fps=[]
    for arm in ARMS:
        pp=[]
        for iid in ids:
            path=out/'predictions'/arm/f'{iid}.json.gz';predhashes[str(path.relative_to(out))]=sha(path)
            with gzip.open(path,'rt',encoding='utf-8') as f:pp.extend(json.load(f))
        row,records,pairs=evaluate(gt,metadata,ids,pp,arm)
        if arm=='original':
            need(len(records)==len(prior),'GT count differs from original official evaluator')
            need(all(r['hit75']==prior[r['annotation_id']] for r in records),'Original official mask75 hits do not replay')
        tasks.append(row);allrecords.extend(records);allpairs.extend(pairs)
        # Record official one-to-one duplicate-like unmatched predictions at mask75.
        with contextlib.redirect_stdout(io.StringIO()):
            # loadRes mutates input dictionaries (adds ndarray bbox/id/area).
            # Rebuild segm-only records before the second official evaluator.
            clean=[{k:r[k] for k in ['image_id','category_id','score','segmentation']} for r in pp]
            dt=gt.loadRes(clean);ev=COCOeval(gt,dt,'segm');ev.params.imgIds=ids;ev.evaluate()
        t=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.75))[0])
        for r in ev.evalImgs:
            if r is None or r['aRng']!=[0,1e10] or r['maxDet']!=100:continue
            ious=np.asarray(ev.ious[(r['image_id'],r['category_id'])])
            original_gt=[x for x in ev._gts[(r['image_id'],r['category_id'])]]
            ordinarycols=[j for j,x in enumerate(original_gt) if not x.get('iscrowd',0)]
            good=np.any(ious[:len(r['dtIds']),ordinarycols]>=.75,axis=1) if ordinarycols and ious.size else np.zeros(len(r['dtIds']),dtype=bool)
            valid=~r['dtIgnore'][t].astype(bool);tp=r['dtMatches'][t]>0
            fps.append(dict(arm=arm,image_id=r['image_id'],category_id=r['category_id'],
                evaluated_predictions=len(r['dtIds']),tp75=int((tp&valid).sum()),fp75=int((~tp&valid).sum()),
                duplicate_like_fp75=int((~tp&valid&good).sum())))
        print(json.dumps(row),flush=True)
    save_csv(out/'task_summary.csv',tasks);save_csv(out/'official_gt_recovery.csv',allrecords)
    save_csv(out/'official_pair_recovery.csv',allpairs);save_csv(out/'official_false_positives.csv',fps)
    rng=np.random.default_rng(20260912);weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    spatial=read(out/'spatial.csv');gtrows=read(out/'gt.csv');counts=read(out/'counts.csv');choices=read(out/'choices.csv')
    def group(rows,name,key):return [r for r in rows if name=='all' or (float(r[key])>.5+1e-10)==(name=='high')]
    baseline={int(r['target_annotation']):r for r in spatial if r['arm']=='original'}
    diffs=[]
    for arm in ARMS[1:]:
        for name in ['all','high','low']:
            rr=group([r for r in spatial if r['arm']==arm],name,'target_ici')
            for field in ['mask_iou','coverage','same_neighbor','background']:
                diffs.append(dict(arm=arm,group=name,metric=field,**boot(rr,lambda r:float(r[field])-float(baseline[int(r['target_annotation'])][field]),ids,weights)))
            originalhits={r['annotation_id']:r['hit75'] for r in allrecords if r['arm']=='original'}
            rr=group([r for r in allrecords if r['arm']==arm],name,'ici')
            diffs.append(dict(arm=arm,group=name,metric='official_r75',**boot(rr,lambda r:int(r['hit75'])-int(originalhits[r['annotation_id']]),ids,weights)))
    groups=[]
    for name in ['all','high','low']:
        rr=group(gtrows,name,'ici');failed=[r for r in rr if r['official_mask75']=='False']
        lost=[r for r in rr if r['score_mask75']=='True' and r['nms_mask75']=='False']
        lostbox=[r for r in rr if r['score_box75']=='True' and r['nms_box75']=='False']
        choices_group=group(choices,name,'ici')
        matched=[r for r in spatial if r['arm']=='original' and (name=='all' or (float(r['target_ici'])>.5+1e-10)==(name=='high'))]
        failids={int(r['target_annotation']) for r in matched if float(r['mask_iou'])<.75}
        recover={arm:sum(int(r['target_annotation']) in failids and float(r['mask_iou'])>=.75 for r in spatial if r['arm']==arm) for arm in ARMS}
        harms={arm:sum(int(r['target_annotation']) in baseline and int(r['target_annotation']) in {int(q['target_annotation']) for q in matched} and float(baseline[int(r['target_annotation'])]['mask_iou'])>=.75 and float(r['mask_iou'])<.75 for r in spatial if r['arm']==arm) for arm in ARMS}
        groups.append(dict(group=name,gt=len(rr),official_failed75=len(failed),
            mask75_available={s:sum(r[s+'_mask75']=='True' for r in rr) for s in STAGES},
            official_fail_with_score_mask75=sum(r['score_mask75']=='True' for r in failed),
            official_fail_with_final_mask75=sum(r['eval100_mask75']=='True' for r in failed),
            nms_lost_mask75=len(lost),nms_lost_box75=len(lostbox),
            lost_best_candidate_relation={k:sum(r['best_mask_suppression_relation']==k for r in lost) for k in ['same_gt','different_gt','ambiguous_or_unassigned','not_suppressed']},
            fixed_matched=len(matched),fixed_failed75=len(failids),fixed_rescued75=recover,fixed_harmed75=harms,
            alternatives_selected=len(choices_group),alternatives_changed=sum(r['original_source']!=r['alternative_source'] for r in choices_group)))
    write_json(out/'ANALYSIS.json',dict(groups=groups,paired_differences=diffs,
        whole_image_unique_feasibility={s:sum(int(r[s+'_max_unique75']) for r in counts) for s in STAGES},
        reused_alternative_source_slots=sum(int(r['reused_alternative_sources']) for r in counts),
        all_nms_replay=all(r['nms_order_exact']=='True' for r in counts),
        bootstrap='2000 paired image-cluster multinomial draws, fixed cohorts; exploratory intervals, no multiple-comparison adjustment; not training-seed uncertainty',
        limitation='Best candidate GT selector favors both arm by construction; not a deployable method. Max matching has no confidence/budget/AP meaning. High/low GT availability is nonexclusive; maximum unique matching is reported only for full images.'))
    write_json(out/'ANALYSIS_COMPLETE.json',dict(status='COMPLETE',script_sha256=sha(__file__),
        baseline_official_mask75_replay=True,prediction_hashes=predhashes,
        hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['ANALYSIS_COMPLETE.json']}))


if __name__=='__main__':main()
