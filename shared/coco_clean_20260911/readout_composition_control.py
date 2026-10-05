"""Descriptive common-stratum comparisons of dense-instance mask errors.

Uses existing frozen predictions. No model training, GT interventions or method
selection. Bootstrap recomputes common-support weights by image cluster.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,csv,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def bbox_iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    inter=np.maximum(hi-lo,0).prod()
    return float(inter/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter))


def comparison(rows,keys,metric,images,draws):
    rr=[r for r in rows if r[metric] is not None]
    cnt=Counter((tuple(r[k] for k in keys),r['high']) for r in rr)
    strata=sorted({key for key,high in cnt if min(cnt[key,True],cnt[key,False])>=3})
    if not strata:return dict(status='INSUFFICIENT_COMMON_SUPPORT')
    sidx={s:i for i,s in enumerate(strata)};iidx={i:k for k,i in enumerate(images)}
    count=np.zeros((len(images),len(strata),2));total=np.zeros_like(count)
    for r in rr:
        key=tuple(r[k] for k in keys)
        if key not in sidx:continue
        loc=(iidx[r['image_id']],sidx[key],int(r['high']))
        count[loc]+=1;total[loc]+=r[metric]
    def reduce(c,t):
        means=np.divide(t,c,out=np.zeros_like(t),where=c>0)
        weights=np.minimum(c[...,0],c[...,1]);den=weights.sum(-1)
        avg=(means*weights[...,None]).sum(-2)/den[...,None]
        return avg
    point=reduce(count.sum(0),total.sum(0))
    # Explicit contractions avoid platform BLAS dispatch for these tiny tables.
    c=np.einsum('bi,ijk->bjk',draws,count,optimize=False)
    t=np.einsum('bi,ijk->bjk',draws,total,optimize=False)
    b=reduce(c,t);d=b[:,1]-b[:,0]
    if not np.isfinite(d).all():raise RuntimeError('Empty common-support bootstrap draw')
    return dict(status='COMPLETE',strata=len(strata),
        matched_targets_high=int(count[:,:,1].sum()),matched_targets_other=int(count[:,:,0].sum()),
        eligible_high=sum(r['high'] for r in rr),eligible_other=sum(not r['high'] for r in rr),
        high=float(point[1]),other=float(point[0]),high_minus_other=float(point[1]-point[0]),
        ci95=np.quantile(d,[.025,.975]).tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path);a=ap.parse_args()
    run=a.run;out=a.out or run/'composition_control';out.mkdir(exist_ok=False);start=time.monotonic()
    def progress(stage):
        record=dict(stage=stage,seconds=time.monotonic()-start)
        write_json(out/'progress.json',record);print(json.dumps(record),flush=True)
    progress('starting')
    protocol=json.loads((run/'cache/protocol.json').read_text())
    ann=Path(protocol['data_root'])/'annotations/instances_train2017.json'
    if sha(ann)!=protocol['annotation_sha256']:raise RuntimeError('Changed original annotations')
    selected=json.loads((run/'cache/selection.json').read_text())['transfer']
    subset=run/'cache/conversion_input/instances_probe.json'
    receipt=json.loads((run/'cache/COMPLETE.json').read_text())
    receipt_paths={k.replace('\\','/'):v for k,v in receipt['hashes'].items()}
    if sha(subset)!=receipt_paths['conversion_input/instances_probe.json']:raise RuntimeError('Subset changed')
    data=json.loads(subset.read_text(encoding='utf-8'));anns={q['id']:q for q in data['annotations'] if q['image_id'] in selected}
    del data
    progress('annotations_loaded')
    mapping={}
    for iid in selected:
        with np.load(run/'cache/images'/f'{iid}.npz') as q:
            for aid,j in zip(q['annotation_ids'],q['prediction_indices']):
                aa=anns[int(aid)];box=np.array(aa['bbox']);box[2:]+=box[:2]
                mapping[int(aid)]=(bbox_iou(box,q['detections'][j,:4]),int(q['level'][j]))
    progress('box_mapping_complete')
    with (run/'error_localization_v2/instances.csv').open(encoding='utf-8') as f:raw=list(csv.DictReader(f))
    rows=[]
    for r in raw:
        if int(r['valid_area'])<=0:continue
        aid=int(r['annotation_id']);area=float(r['valid_area']);biou,level=mapping[aid]
        if biou<.5-1e-6:raise RuntimeError('Fixed attribution lost bbox50')
        rows.append(dict(image_id=int(r['image_id']),annotation_id=aid,high=r['density']=='high',
            category_id=int(r['category_id']),size='small' if float(r['area'])<1024 else 'medium' if float(r['area'])<9216 else 'large',
            box_bin='50_75' if biou<.75 else '75_90' if biou<.9 else '90_100',box_iou=biou,level=level,
            exposure=float(r['same_neighbor_available'])/area,
            neighbor_fp_own=float(r['false_positive_same_neighbor'])/area,
            background_fp_own=float(r['false_positive_background'])/area,
            neighbor_fpr=float(r['neighbor_fp_rate']) if r['neighbor_fp_rate'] else None,
            coverage=float(r['own_coverage']),fixed_iou=float(r['fixed_iou']),
            task_failure=float(r['official_hit75']!='True')))
    rng=np.random.default_rng(20260912);draws=rng.multinomial(len(selected),np.ones(len(selected))/len(selected),size=2000)
    results=[]
    for keys in [('category_id','size'),('category_id','size','box_bin')]:
        for metric in ['task_failure','fixed_iou','coverage','neighbor_fp_own','background_fp_own','exposure','neighbor_fpr']:
            progress(':'.join(keys)+':'+metric)
            results.append(dict(stratification=list(keys),metric=metric,**comparison(rows,keys,metric,selected,draws)))
    with (out/'instances.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(out/'SUMMARY.json',dict(results=results,seconds=time.monotonic()-start,
      scope='Post-hoc exploratory comparisons on frozen bbox50-matched targets. No causal effect or standard COCO recall claim. '
      'Task_failure within matched subset differs from whole all-GT recovery. '
      'Common strata require at least3 original targets per group. Shared weights=min(n_high,n_other); '
      'weights recomputed within 2000 paired image-cluster bootstrap draws, original stratum membership fixed. '
      'Neighbor FPR has its own available-neighbor subset and common support. '
      'Category/size coarse bins and box bins leave residual confounding, no equivalence or multiple-comparison adjustment.',
      source_hashes={'spatial':sha(run/'error_localization_v2/instances.csv'),'annotations':sha(ann),'script':sha(Path(__file__))}))
    print(json.dumps(results),flush=True)
    progress('COMPLETE')


if __name__=='__main__':main()
