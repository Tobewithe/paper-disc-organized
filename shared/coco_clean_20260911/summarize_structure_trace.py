"""Same-IoU official box/mask matches and descriptive feature summaries."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,gzip,io,json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from frozen_mechanism_probe import ROOT,write_json,sha
from structure_candidate_trace import STAGES,pair_exists,save_csv

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();source=a.source
    require=json.loads((source/'COMPLETE.json').read_text())['status']=='COMPLETE'
    if not require:raise RuntimeError('Incomplete experiment')
    protocol=json.loads((source/'protocol.json').read_text());ids=protocol['images'];wanted=set(ids)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    with gzip.open(ROOT/'diagnostics/full_local_comparison_20260911/predictions/initial.json.gz','rt') as f:pred=[r for r in json.load(f) if r['image_id'] in wanted]
    boxes=[];higher=[];categories=sorted(gt.cats);catidx={v:k for k,v in enumerate(categories)}
    for iid in ids:
        q=np.load(source/'raw'/f'{iid}.npz');score=q['class_scores'];cls=score.argmax(1);conf=score.max(1)
        for j in q['nonempty_indices']:
            b=q['boxes_original'][j];boxes.append(dict(image_id=iid,category_id=categories[cls[j]],score=float(conf[j]),bbox=[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])]))
        pools=dict(raw_geometry=np.arange(len(score)),argmax_class=np.arange(len(score)),score=np.flatnonzero(conf>.001),nms=q['nms_indices'],top300=q['top300_indices'],nonempty=q['nonempty_indices'],eval100=q['eval100_indices'])
        for index,aid in enumerate(q['annotation_ids']):
            label=catidx[gt.anns[int(aid)]['category_id']];row=dict(image_id=iid,annotation_id=int(aid))
            for stage,pool in pools.items():row[stage+'_available75']=bool(((q['bbox_iou'][index,pool]>=.75)&((cls[pool]==label) if stage!='raw_geometry' else True)).any())
            higher.append(row)
    matches={}
    for mode,pp in [('bbox',boxes),('segm',pred)]:
        with contextlib.redirect_stdout(io.StringIO()):
            dt=gt.loadRes(pp);ev=COCOeval(gt,dt,mode);ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
        for r in ev.evalImgs:
            if r is None or r['aRng']!=[0,1e10] or r['maxDet']!=100:continue
            for i,aid in enumerate(r['gtIds']):
                if r['gtIgnore'][i]:continue
                item=matches.setdefault(int(aid),{})
                for tau in [.5,.75]:
                    t=int(np.flatnonzero(np.isclose(ev.params.iouThrs,tau))[0]);item[f'{mode}{int(tau*100)}']=bool(r['gtMatches'][t,i])
    hi={r['annotation_id']:r for r in higher};instances=read(source/'instances.csv')
    for r in instances:
        aid=int(r['annotation_id']);r.update(matches[aid]);r.update({k:v for k,v in hi[aid].items() if k.endswith('75')})
    save_csv(source/'instances_same_iou.csv',instances)
    summary=[]
    for group in ['all','high','low']:
        rows=[r for r in instances if group=='all' or (float(r['ici_same'])>.5+1e-10)==(group=='high')]
        summary.append(dict(group=group,gt=len(rows),**{k:float(np.mean([r[k] for r in rows])*100) for k in ['bbox50','bbox75','segm50','segm75']},
            bbox75_and_mask_fail75=sum(r['bbox75'] and not r['segm75'] for r in rows),mask75_without_bbox75=sum(r['segm75'] and not r['bbox75'] for r in rows),
            **{stage+'_availability75_pct':float(np.mean([r[stage+'_available75'] for r in rows])*100) for stage in STAGES}))
    save_csv(source/'same_iou_summary.csv',summary)
    raw=read(source/'features.csv');keys=['auc','shuffled_train_auc','coordinate_auc','region_mean_cosine','mixed_cell_fraction','actual_a_auc','actual_a_mean_margin','actual_pair_difference_auc','coefficient_cosine']
    grouped={}
    for r in raw:
        key=tuple(r[k] for k in ['image_id','annotation_a','annotation_b','pair_type','task','layer'])
        grouped.setdefault(key,[]).append(r)
    points=[]
    for key,rr in grouped.items():
        if len(rr)!=3:raise RuntimeError('Expected three probe seeds')
        p={k:rr[0][k] for k in ['image_id','annotation_a','annotation_b','pair_type','task','layer','category_a','category_b','area_a','area_b','box_iou','ici_a','ici_b','normalized_center_distance']}
        p['high']=max(float(p['ici_a']),float(p['ici_b']))>.5+1e-10
        for metric in keys:
            valid=[float(r[metric]) for r in rr if r.get(metric,'')!='']
            if valid:p[metric]=float(np.mean(valid))
        points.append(p)
    save_csv(source/'feature_pair_means.csv',points)
    features=[]
    for pair_type in ['same_adjacent','same_separate','different_adjacent']:
        for task in ['own_neighbor','own_background']:
            for group in ['all','high','low']:
                for layer in sorted({r['layer'] for r in points}):
                    rr=[r for r in points if r['pair_type']==pair_type and r['task']==task and r['layer']==layer and (group=='all' or r['high']==(group=='high'))]
                    if not rr:continue
                    row=dict(pair_type=pair_type,task=task,group=group,layer=layer,pairs=len(rr))
                    for metric in keys:
                        vv=[r[metric] for r in rr if metric in r]
                        if vv:row[metric]=float(np.mean(vv));row[metric+'_n']=len(vv)
                    features.append(row)
    save_csv(source/'feature_summary.csv',features)
    # Paired spatial probes: only same instances with proto actual scores available.
    primary=[r for r in points if r['layer']=='proto' and r['task']=='own_neighbor' and r['pair_type']=='same_adjacent' and r['high'] and 'actual_a_auc' in r]
    imageidx={iid:j for j,iid in enumerate(ids)};count=np.zeros(len(ids));delta=np.zeros(len(ids))
    for r in primary:ii=imageidx[int(r['image_id'])];count[ii]+=1;delta[ii]+=r['auc']-r['actual_a_auc']
    rng=np.random.default_rng(20260911);draw=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000);den=draw@count
    boot=(draw@delta)[den>0]/den[den>0]*100
    paired=dict(scope='Within-image GT oracle linear readout minus actual coefficient pixel-ranking AUC. Pair selected using GT; conditioned on eligible pixels and matched A. Pixel task, not AP or instance recall.',pairs=len(primary),mean_pp=float(delta.sum()/max(count.sum(),1)*100),ci_low_pp=float(np.quantile(boot,.025)),ci_high_pp=float(np.quantile(boot,.975)),valid_bootstrap=len(boot))
    write_json(source/'STRUCTURE_SUMMARY.json',dict(stage_same_iou=summary,proto_oracle_minus_actual_high=paired,feature_scope='Three fixed random readout seeds averaged per pair. Pair groups not yet matched on category/area/distance; checkerboard spatial blocks share receptive fields. Probes at full-image pixel locations, actual raw logits BEFORE crop, not deployed masks.',files={p.name:sha(p) for p in [source/'instances_same_iou.csv',source/'feature_pair_means.csv',source/'feature_summary.csv']}))
    print(json.dumps(summary));print(json.dumps(paired))
    for r in features:
        if r['pair_type']=='same_adjacent' and r['task']=='own_neighbor' and r['group']=='high':print(json.dumps(r))

if __name__=='__main__':main()
