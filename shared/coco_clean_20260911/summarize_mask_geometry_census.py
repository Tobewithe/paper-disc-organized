"""S048 descriptive census, reproducible stratification and frozen candidates."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd


def rank(a):return hashlib.sha256(f'S049-before-input:20260912:{a}'.encode()).hexdigest()


def cluster_summary(df,value):
    x=df[['image_id',value]].dropna();g=x.groupby('image_id')[value].agg(['sum','count']);n=len(g)
    if not n:return dict(n=0,mean=None,ci95=None)
    sums=g['sum'].to_numpy(float);counts=g['count'].to_numpy(float);rng=np.random.default_rng(20260912);boot=[]
    for _ in range(2000):
        idx=rng.integers(n,size=n);boot.append(float(sums[idx].sum()/counts[idx].sum()))
    return dict(n=int(counts.sum()),images=n,mean=float(sums.sum()/counts.sum()),ci95=np.quantile(boot,[.025,.975]).tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    data=pd.read_csv(out/'instances.csv');d=data.copy();valid=d.geometry_valid.fillna(False)
    d.loc[~valid,'mask_density']='undefined'
    d['boxgood_maskbad']=d.state=='box_good__mask_bad';d['boxbad_maskgood']=d.state=='box_bad__mask_good'
    d['goodbox_maskbad_supported']=d.boxgood_maskbad&d.support_sufficient
    d['strict_unrescued']=d.unrescued_mask_failure&d.strict_box90
    d.to_csv(out/'instances_classified.csv',index=False)
    summary=[]
    for kind in ['mask_density','ici_high']:
        for group,x in d.groupby(kind):
            summary.append(dict(grouping=kind,group=str(group),gt=len(x),images=x.image_id.nunique(),
                states=dict(Counter(x.state)),matched=int(x.matched.sum()),supported_mask_failures=int(x.goodbox_maskbad_supported.sum()),
                unrescued_mask_failures=int(x.unrescued_mask_failure.sum()),strict_box90_unrescued=int(x.strict_unrescued.sum()),
                official_mask75=cluster_summary(x,'official_mask75'),official_box75=cluster_summary(x,'official_box75'),
                same_slot_boxgood_maskbad=cluster_summary(x,'boxgood_maskbad'),
                conditional_mask_failure_given_boxgood=cluster_summary(x[x.state.isin(['box_good__mask_good','box_good__mask_bad'])],'boxgood_maskbad')))
    threshold=[]
    for radius in [2,4,8]:
        for cut in [.1,.2,.3]:
            take=d[f'same_boundary_exposure{radius}']>=cut
            threshold.append(dict(radius=radius,cut=cut,gt=int(take.sum()),mask75=float(d.loc[take,'official_mask75'].mean()),
                unrescued=int(d.loc[take,'unrescued_mask_failure'].sum()),strict90=int(d.loc[take,'strict_unrescued'].sum())))
    pool=d[valid & d.unrescued_mask_failure & (d.same_boundary_exposure4>=.2) & (d.same_exclusive_exposure4>=.2) &
           (d.same_overlap_fraction<=.01) & (d.same_pair_exposure4>=.1) & (d.same_pair_overlap_fraction<=.01)].copy()
    pool['hash_rank']=pool.annotation_id.map(rank);pool=pool.sort_values('hash_rank');pool.to_csv(out/'CROWDED_SUPPORTED_MASK_FAILURE_POOL.csv',index=False)
    # All pools defined before input intervention. Do not expand according to its result.
    success=d[valid & (d.state=='box_good__mask_good') & d.support_sufficient & (d.same_boundary_exposure4>=.2)&
        (d.same_exclusive_exposure4>=.2)&(d.same_overlap_fraction<=.01)&(d.same_pair_exposure4>=.1)].copy()
    other=d[valid & d.unrescued_mask_failure & (d.different_boundary_exposure4>=.2)&
        (d.different_exclusive_exposure4>=.2)&(d.different_overlap_fraction<=.01)&(d.different_pair_exposure4>=.1)&(d.same_boundary_exposure4<=1e-10)].copy()
    for name,frame in [('CROWDED_SUCCESS_POOL',success),('DIFFERENT_CROWDED_MASK_FAILURE_POOL',other)]:
        frame['hash_rank']=frame.annotation_id.map(rank);frame.sort_values('hash_rank').to_csv(out/(name+'.csv'),index=False)
    # Common-category/size descriptive comparison: no conditioning on outcome.
    low=d[d.mask_density=='low'];high=d[d.mask_density=='high'];cells=[]
    for (cat,size),x in d.groupby(['category_id','area_bin']):
        a=x[x.mask_density=='low'];b=x[x.mask_density=='high']
        if len(a)>=10 and len(b)>=10:cells.append(dict(category=int(cat),size=size,low=len(a),high=len(b),
            mask75_low=float(a.official_mask75.mean()),mask75_high=float(b.official_mask75.mean()),
            box75_low=float(a.official_box75.mean()),box75_high=float(b.official_box75.mean())))
    weights=np.array([q['low']+q['high'] for q in cells],float);weights/=weights.sum()
    standardized={metric:float(sum(w*(q[metric+'_high']-q[metric+'_low']) for w,q in zip(weights,cells))) for metric in ['mask75','box75']}
    joint=pd.crosstab(d.ici_high,d.mask_density).reset_index().to_dict('records')
    result=dict(experiment='S048',gt=len(d),images=5000,images_with_ordinary_gt=d.image_id.nunique(),geometry_invalid=int((~valid).sum()),
        counts=dict(Counter(d.state)),supported_goodbox_maskbad=int(d.goodbox_maskbad_supported.sum()),
        unrescued=int(d.unrescued_mask_failure.sum()),strict_box90_unrescued=int(d.strict_unrescued.sum()),groups=summary,
        ici_mask_density_cross=joint,threshold_sensitivity=threshold,primary_failure_pool=len(pool),primary_failure_images=pool.image_id.nunique(),
        strict_failure_pool=int(pool.strict_box90.sum()),success_pool=len(success),different_failure_pool=len(other),
        primary_pool_error_types=dict(Counter(pool.dominant_error)),primary_pool_category_counts={str(int(k)):int(v) for k,v in pool.category_id.value_counts().items()},
        adjusted=dict(cells=cells,pointwise_high_minus_low=standardized,retained_low=sum(q['low'] for q in cells),retained_high=sum(q['high'] for q in cells),
            limits='Pooled category-size cell weights,pointestimate only; not causal, no claim box/visibility/annotationconfounding removed.'),
        limits='Primary geometry threshold fixed before reading outcomes; masks are visible annotations, not amodal occlusion. AllGT detectionrecall differs from same-slot conditionalIoU. Original floatboxcoverage is proxy pending exactcropwitness. Frozen pools precede newimageintervention but are selected from exploredval, no method/AP confirmation.')
    (out/'ANALYSIS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','threshold_sensitivity','adjusted','primary_pool_category_counts']},ensure_ascii=False))
    for r in summary:print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__':main()
