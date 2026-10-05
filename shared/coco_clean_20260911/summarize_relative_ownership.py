"""Official COCO task evaluation and paired instance-level intervention effects."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, contextlib, csv, gzip, hashlib, io, json, time
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from frozen_mechanism_probe import ROOT, sha, write_csv, write_json, box_iou


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def evaluate(gt, meta, ids, pp, arm):
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(pp)
        ev=COCOeval(gt,dt,'segm');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    row=dict(arm=arm,mask_ap=float(ev.stats[0]),mask_ap50=float(ev.stats[1]),mask_ap75=float(ev.stats[2]),predictions=len(pp))
    t=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.75))[0]);recovery={};records=[]
    for r in ev.evalImgs:
        if r is None or r['aRng']!=[0,1e10] or r['maxDet']!=100:continue
        for j,aid in enumerate(r['gtIds']):
            if r['gtIgnore'][j]:continue
            aid=int(aid);hit=bool(r['gtMatches'][t,j]);ici=float(meta[aid]['ici_same']);recovery[aid]=hit
            area=gt.anns[aid]['area']
            records.append(dict(arm=arm,image_id=r['image_id'],annotation_id=aid,ici=ici,
                 category_id=gt.anns[aid]['category_id'],area=area,area_bin='small' if area<32**2 else 'medium' if area<96**2 else 'large',
                 hit75=hit))
    pairrecords=[]
    for iid in ids:
        aa=[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0)]
        for j,a in enumerate(aa):
            for b in aa[j+1:]:
                if a['category_id']==b['category_id'] and box_iou(a['bbox'],b['bbox'])>.05:
                    pairrecords.append(dict(arm=arm,image_id=iid,annotation_a=a['id'],annotation_b=b['id'],
                        ici=max(float(meta[a['id']]['ici_same']),float(meta[b['id']]['ici_same'])),
                        hit75=recovery[a['id']] and recovery[b['id']]))
    for group in ['all','high','low']:
        for prefix,rows in [('r75',records),('pair75',pairrecords)]:
            rr=[r for r in rows if group=='all' or (r['ici']>.5+1e-10)==(group=='high')]
            row[f'{prefix}_{group}']=float(np.mean([r['hit75'] for r in rr])) if rr else None
            row[f'{prefix}_{group}_n']=len(rr)
    return row,records,pairrecords


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    receipt=json.loads((out/'EVALUATION_COMPLETE.json').read_text());assert receipt['status']=='COMPLETE'
    for name,digest in receipt['hashes'].items():assert sha(out/name)==digest,name
    ids=json.loads((out/'selection.json').read_text())['evaluation'];lock=json.loads((out/'LOCKED_SETTINGS.json').read_text())
    arms=list(lock['arms']);iidindex={iid:j for j,iid in enumerate(ids)}
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    meta={int(r['annotation_id']):r for r in read(ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv')}
    stats=[];allrecords=[];allpairs=[];hashes={};start=time.monotonic();refpred=None;semantic_cache={};deduplicated={}
    for arm in arms:
        pp=[]
        for iid in ids:
            p=out/'predictions_by_image'/arm/f'{iid}.json.gz';hashes[str(p.relative_to(out))]=sha(p)
            with gzip.open(p,'rt') as f:pp.extend(json.load(f))
        signatures=[(r['image_id'],r['category_id'],r['score']) for r in pp]
        if arm=='initial':refpred=signatures
        elif lock['arms'][arm]['kind']!='threshold':assert signatures==refpred,'Candidate/score changed in exact-area arm'
        digest=hashlib.sha256(json.dumps(pp,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest in semantic_cache:
            oldarm,oldrow,oldrecords,oldpairs=semantic_cache[digest]
            row={**oldrow,'arm':arm};records=[{**r,'arm':arm} for r in oldrecords];pairs=[{**r,'arm':arm} for r in oldpairs]
            deduplicated[arm]=dict(identical_prediction_arm=oldarm,semantic_sha256=digest)
        else:
            row,records,pairs=evaluate(gt,meta,ids,pp,arm)
            semantic_cache[digest]=(arm,row,records,pairs)
        stats.append(row);allrecords.extend(records);allpairs.extend(pairs)
        print(json.dumps(row),flush=True)
        write_json(out/'task_progress.json',dict(completed=len(stats),total=len(arms),seconds=time.monotonic()-start))
    write_csv(out/'task_summary.csv',stats);write_csv(out/'gt_recovery.csv',allrecords);write_csv(out/'pair_recovery.csv',allpairs)
    write_json(out/'PREDICTION_HASHES.json',hashes)
    spatial=read(out/'evaluation_spatial.csv')
    correction=read(out/'evaluation_corrections.csv') if (out/'evaluation_corrections.csv').exists() else []
    assert all(int(r['original_area'])==int(r['resulting_area']) for r in correction)
    draws=np.random.default_rng(20260911).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    fields=['coverage','same_neighbor','neighbor','background','mask_iou','exclusive_same_neighbor','predicted_area_over_gt']
    basetask=sorted([r for r in allrecords if r['arm']=='initial'],key=lambda r:r['annotation_id'])
    basepair=sorted([r for r in allpairs if r['arm']=='initial'],key=lambda r:(r['image_id'],r['annotation_a'],r['annotation_b']))
    basespatial=sorted([r for r in spatial if r['arm']=='initial'],key=lambda r:int(r['target_annotation']))
    assert len(basetask)==sum(sum(not q.get('iscrowd',0) for q in gt.imgToAnns[i]) for i in ids)
    matrices={};cohorts={}
    for name,rows,key,metrics in [('task',allrecords,lambda r:r['annotation_id'],['hit75']),
                                 ('pair',allpairs,lambda r:(r['image_id'],r['annotation_a'],r['annotation_b']),['hit75']),
                                 ('spatial',spatial,lambda r:int(r['target_annotation']),fields)]:
        matrices[name]={}
        for arm in arms:
            rr=sorted([r for r in rows if r['arm']==arm],key=key)
            if arm=='initial':cohorts[name]=rr;expected=[key(r) for r in rr]
            assert [key(r) for r in rr]==expected
            matrices[name][arm]=np.array([[float(r[m]) for m in metrics] for r in rr])
        random_arrays=[matrices[name][f'random_s{s}'] for s in range(3)]
        matrices[name]['random_mean']=(random_arrays[0].copy() if all(np.array_equal(random_arrays[0],v) for v in random_arrays[1:]) else np.mean(random_arrays,axis=0))
    contrasts=[]
    def interval(values,idx,mask):
        totals=np.bincount(idx[mask],weights=values[mask],minlength=len(ids));counts=np.bincount(idx[mask],minlength=len(ids))
        den=draws@counts;ok=den>0;boot=(draws@totals)[ok]/den[ok]*100
        return dict(mean_pp=float(values[mask].mean()*100),ci_low_pp=float(np.quantile(boot,.025)),ci_high_pp=float(np.quantile(boot,.975)),n=int(mask.sum()),images=int((counts>0).sum()))
    comparisons=[('relative','initial'),('relative','geometry'),('relative','threshold'),('relative','random_mean'),('geometry','initial'),('threshold','initial')]
    for name,metrics in [('task',['mask_r75']),('pair',['pair_r75']),('spatial',fields)]:
        rows=cohorts[name];idx=np.array([iidindex[int(r['image_id'])] for r in rows])
        ici=np.array([float(r['target_ici'] if name=='spatial' else r['ici']) for r in rows])
        groups={'all':np.ones(len(rows),dtype=bool),'high':ici>.5+1e-10,'low':ici<=.5+1e-10,
                'ici_zero':ici==0,'ici_0_025':(ici>0)&(ici<=.25),'ici_025_05':(ici>.25)&(ici<=.5+1e-10),
                'ici_05_1':(ici>.5+1e-10)&(ici<=1),'ici_gt1':ici>1}
        for treatment,control in comparisons:
            delta=matrices[name][treatment]-matrices[name][control]
            for group,mask in groups.items():
                if not mask.any():continue
                for j,metric in enumerate(metrics):contrasts.append(dict(domain=name,treatment=treatment,control=control,group=group,metric=metric,**interval(delta[:,j],idx,mask)))
    # Category x size standardized gains: fixed shared GT composition; point estimate only.
    strata={};task=cohorts['task']
    for j,r in enumerate(task):strata.setdefault((r['category_id'],r['area_bin']),{'high':[],'low':[]})['high' if r['ici']>.5+1e-10 else 'low'].append(j)
    common={k:v for k,v in strata.items() if v['high'] and v['low']};den=sum(len(v['high'])+len(v['low']) for v in common.values());standard=[]
    for treatment,control in comparisons:
        delta=(matrices['task'][treatment]-matrices['task'][control])[:,0]
        row=dict(treatment=treatment,control=control,common_strata=len(common),common_gt=den)
        for group in ['high','low']:
            row[group+'_standardized_gain_pp']=float(sum((len(v['high'])+len(v['low']))/den*delta[v[group]].mean() for v in common.values())*100) if den else None
        standard.append(row)
    impact=[]
    for arm in arms:
        rr=[r for r in correction if r['arm']==arm]
        impact.append(dict(arm=arm,recorded_nonzero_intervention_predictions=len(rr),
             eligibility_note='Rows record eligible predictions only when selected alpha is nonzero; zero rows for alpha=0 do NOT mean no eligible neighbors.',
             changed_predictions=sum(int(r['changed_pixels'])>0 for r in rr),
             changed_input_pixels=sum(int(r['changed_pixels']) for r in rr),max_area_error=max([abs(int(r['original_area'])-int(r['resulting_area'])) for r in rr] or [0])))
    write_json(out/'PAIRED_ANALYSIS.json',dict(scope='2000 paired image-cluster draws; parameters train-selected; explored val. Pointwise CIs, no multiplicity correction. Random controls are three spatial shuffles not trained seeds. AP is official point estimate; intervals below concern R75 and fixed-attribution spatial metrics.',
         contrasts=contrasts,category_area_standardization=standard,intervention_impact=impact,
         gt_total=len(basetask),matched_spatial_targets=len(basespatial),unmatched_or_no_valid_pixels=len(basetask)-len(basespatial),pair_total=len(basepair),
         identical_prediction_reuse=deduplicated,
         zero_intervention_selected=all(lock['arms'][arm]['value']==0 for arm in arms)))
    write_json(out/'TASK_COMPLETE.json',dict(status='COMPLETE',training=False,images=len(ids),arms=arms,seconds=time.monotonic()-start,
        source_sha256=sha(__file__),hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='TASK_COMPLETE.json'}))
    for r in contrasts:
        if r['group']=='high' and r['control'] in ['initial','random_mean'] and r['metric'] in ['mask_r75','pair_r75','coverage','same_neighbor','background','mask_iou']:print(json.dumps(r),flush=True)


if __name__=='__main__':main()
