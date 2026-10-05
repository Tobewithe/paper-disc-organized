"""Paired image-cluster intervals for R75/gap; AP remains official point estimate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent

def analyze(dest,prefix):
    df=pd.read_csv(dest/'summary.csv');gt=pd.read_csv(dest/'gt_recovery.csv')
    ids=sorted(gt.image_id.unique());aids=sorted(gt.annotation_id.unique())
    meta=gt.drop_duplicates('annotation_id').set_index('annotation_id').loc[aids]
    high=meta.ici.to_numpy()>.5+1e-10;im=np.searchsorted(ids,meta.image_id.to_numpy())
    draws=np.random.default_rng(20260913).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    vals={arm:g.set_index('annotation_id').loc[aids].hit75.to_numpy(dtype=float) for arm,g in gt.groupby('arm')}
    means={};groups={}
    for arm in vals:
        if arm.endswith(('_s0','_s1','_s2')):groups.setdefault(arm[:-3],[]).append(arm)
        else:groups[arm]=[arm]
    for key,names in groups.items():means[key]=np.mean([vals[n] for n in names],axis=0)
    summary=[]
    for key,names in groups.items():
        rows=df[df.arm.isin(names)]
        row=dict(group=key,seeds=len(names))
        for metric in ['mask_ap','mask_ap50','mask_ap75','r75_high','r75_low','gap','pair75_high']:
            row[metric+'_mean_pp']=rows[metric].mean()*100
            row[metric+'_sd_pp']=rows[metric].std(ddof=1)*100 if len(rows)>1 else None
        summary.append(row)
    pd.DataFrame(summary).to_csv(dest/'group_means.csv',index=False)
    comparisons=[(m,'s032') for m in means if m not in ['s032','original','original_late']]
    if prefix=='existing':comparisons += [('s032','original'),('original_late','original'),('s032_late','s032'),
        ('bsr_center','bsr'),('ada_calib_center','ada_calib'),('full_synergy','full_synergy_nocenter')]
    else:comparisons += [('new_contrast','new_bsr'),('new_ada','new_bsr'),('new_ada','new_scalar'),
        ('new_orth','new_adapter'),('new_full','new_full_noorth'),('new_bsr_center','new_bsr'),('new_ada_center','new_ada')]
    out=[]
    for a,b in comparisons:
        if a not in means or b not in means:continue
        delta=means[a]-means[b];boots={};points={}
        for name,mask in [('high',high),('nonhigh',~high)]:
            numer=np.bincount(im[mask],weights=delta[mask],minlength=len(ids))
            denom=np.bincount(im[mask],minlength=len(ids))
            den=draws@denom
            boots[name]=np.divide(draws@numer,den,out=np.full(len(draws),np.nan),where=den>0)*100
            points[name]=delta[mask].mean()*100
        for name,boot,point in [('high_gain',boots['high'],points['high']),('nonhigh_gain',boots['nonhigh'],points['nonhigh']),
            ('gap_change',boots['nonhigh']-boots['high'],points['nonhigh']-points['high'])]:
            out.append(dict(treatment=a,control=b,metric=name,estimate_pp=point,
                ci_low_pp=float(np.nanquantile(boot,.025)),ci_high_pp=float(np.nanquantile(boot,.975))))
    pd.DataFrame(out).to_csv(dest/'paired_image_intervals.csv',index=False)
    write=dict(scope='2000 paired image-cluster resamples of mean of three seeds; conditional on these trained seeds, pointwise, exploratory, no multiplicity correction. AP intervals not computed.',
        images=len(ids),gt=len(aids),high_gt=int(high.sum()),nonhigh_gt=int((~high).sum()))
    (dest/'statistics_scope.json').write_text(json.dumps(write,indent=2),encoding='utf-8')

if __name__=='__main__':
    for name,prefix in [('existing_aligned','existing'),('corrected_aligned','corrected')]:
        if (OUT/name/'COMPLETE.json').exists():analyze(OUT/name,prefix)
    if (OUT/'s080_corrected_instances.csv').exists():
        cases=pd.read_csv(OUT/'s080_corrected_instances.csv');out=[]
        for arm,sub in cases.groupby('arm'):
            refs=['original']+([f's032_s{arm[-1]}'] if arm.endswith(('_s0','_s1','_s2')) else [])
            for ref in refs:
                if ref==arm:continue
                base=cases[cases.arm==ref]
                paired=sub.merge(base,on=['image_id','annotation_id','residual'],suffixes=('_method','_base'),validate='one_to_one')
                for label in ['all']+sorted(paired.residual.unique()):
                    r=paired if label=='all' else paired[paired.residual==label]
                    out.append(dict(arm=arm,control=ref,stratum=label,n=len(r),
                        rescued=int(((r.hit75_method==1)&(r.hit75_base==0)).sum()),
                        harmed=int(((r.hit75_method==0)&(r.hit75_base==1)).sum()),
                        delta_iou_pp=float((r.iou_method-r.iou_base).mean()*100)))
        pd.DataFrame(out).to_csv(OUT/'s080_rescue_harm.csv',index=False)
