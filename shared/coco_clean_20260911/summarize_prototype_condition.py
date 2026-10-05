"""S045 conditional diagnostic statistics; no new fitting or data selection."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();src=a.source
    receipt=json.loads((src/'COMPLETE.json').read_text());checked=0
    for path,h in receipt['hashes'].items():
        if hashlib.sha256((src/path.replace('\\','/')).read_bytes()).hexdigest()!=h:raise RuntimeError(path)
        checked+=1
    df=pd.read_csv(src/'crossfit.csv');sharing=pd.read_csv(src/'image_sharing.csv');null=pd.read_csv(src/'permutation_null.csv')
    for mode in df['mode'].unique():
        dd=df[df['mode']==mode]
        assert len(dd)==7811*3 and not dd.duplicated(['annotation_id','seed']).any()
        assert set(dd['annotation_id'])==set(df[df['mode']=='original']['annotation_id'])
    assert sharing.groupby(['annotation_id']).size().eq(3).all()
    images=sorted(df.image_id.unique());imlookup={v:k for k,v in enumerate(images)};ni=len(images)
    draws=np.random.default_rng(20260912).multinomial(ni,np.full(ni,1/ni),size=2000)
    def stats(frame,values,weighted=False):
        values=np.asarray(values,dtype=float);ii=np.array([imlookup[v] for v in frame.image_id]);wt=frame.factor.to_numpy() if weighted else np.ones(len(values))
        assert np.isfinite(values).all()
        den=np.bincount(ii,weights=wt,minlength=ni);num=np.bincount(ii,weights=wt*values,minlength=ni)
        ds=np.einsum('bi,i->b',draws,den,optimize=False);ns=np.einsum('bi,i->b',draws,num,optimize=False)
        v=np.divide(ns,ds,out=np.full_like(ns,np.nan),where=ds>0)
        return dict(n=len(values),images=len(set(ii)),mean=float(num.sum()/den.sum()),ci95=np.nanquantile(v,[.025,.975]).tolist())
    def group_rows(d,group):
        return d if group=='all' else d[(d.group!='high') if group=='nonhigh' else (d.group==group)]
    av=df.groupby(['image_id','annotation_id','mode','group'],as_index=False)[['mse','iou','factor']].mean()
    means=[];contrasts=[]
    for group in ['all','high','low','middle','nonhigh']:
        selected=group_rows(av,group)
        for mode in df['mode'].unique():
            dd=selected[selected['mode']==mode]
            means.append(dict(group=group,mode=mode,mse=stats(dd,dd.mse),weighted_mse=stats(dd,dd.mse,True),iou=stats(dd,dd.iou)))
        for alt,ref in [('prototype','base'),('prototype','instance'),('prototype','shuffled'),('shuffled','base'),('base','saved')]:
            left=selected[selected['mode']==alt].set_index('annotation_id');right=selected[selected['mode']==ref].set_index('annotation_id').reindex(left.index)
            assert (left.image_id==right.image_id).all()
            contrasts.append(dict(group=group,alternative=alt,reference=ref,mse=stats(left,left.mse-right.mse),
                weighted_mse=stats(left,left.mse-right.mse,True),iou=stats(left,left.iou-right.iou)))
    ss=sharing.groupby(['image_id','annotation_id','group'],as_index=False)[['before','after','other_targets']].mean();image_results=[]
    for group in ['all','high','low','middle','nonhigh']:
        dd=group_rows(ss,group);delta=dd.after-dd.before
        image_results.append(dict(group=group,before=stats(dd,dd.before),after=stats(dd,dd.after),delta=stats(dd,delta),
            improvement_fraction=float((delta<0).mean()),delta_quantiles=np.quantile(delta,[0,.05,.25,.5,.75,.95,1]).tolist(),
            other_targets_quantiles=np.quantile(dd.other_targets,[0,.25,.5,.75,1]).tolist()))
    null_results=[]
    for group in ['all','high','nonhigh']:
        dd=null[null.group==group].groupby('rep')[['before','after']].mean();actual=next(v for v in image_results if v['group']==group)
        null_results.append(dict(group=group,replicates=20,mean_before=float(dd.before.mean()),mean_after=float(dd.after.mean()),
            mean_delta=float((dd.after-dd.before).mean()),range_delta=[float((dd.after-dd.before).min()),float((dd.after-dd.before).max())],
            fraction_null_delta_le_actual=float(((dd.after-dd.before)<=actual['delta']['mean']).mean()),
            moved_image_fraction=[float(null.moved_image_fraction.min()),float(null.moved_image_fraction.max())]))
    solvers=json.loads((src/'solvers.json').read_text());convergence=[]
    for mode in ['base','instance','prototype','shuffled']:
        dd=[v for v in solvers if v['mode']==mode]
        convergence.append(dict(mode=mode,converged=sum(v['converged'] for v in dd),total=len(dd),
            max_true_residual=max(v['relative_residual'] for v in dd),max_objective_gap_bound=max(v['objective_gap_upper_bound'] for v in dd),
            train_mse_mean=float(np.mean([v['train_mse'] for v in dd]))))
    verify=dict(status='PASS',hashes=checked,targets=7811,seeds=3,all_six_outputs_same_targets=True,
        scope='Receipt hashes and target identities, not an independent GPU rerun. Six pureinstance PCG fits not converged is retained, not corrected by statistics.')
    result=dict(seconds=receipt['seconds'],means=means,contrasts=contrasts,image_sharing=image_results,permutation_null=null_results,
        convergence=convergence,seed_fold_mse=[{k:v[k] for k in ['seed','eval_fold','mode','train_mse','eval_mse','converged']} for v in solvers],
        scope='Existingfit images. NEWmaps twofold imagecrossfit but upstreamfrozenheadsandteachersalreadyusedallfit. '
            'Not end-to-end independent-image performance; allIoUs512sampledpointsnotcomplete masksorAP. '
            'Sameimageleaveoneout is GT-assisted and mayextrapolate outside donor support. Nullstratifies category/GTarea/ICI, '
            'not randomizedcausaltest. Negative prototype/globalmoment recipe doesnotexclude allprototypeorinstanceconditioning.',verification=verify)
    for name,value in [('VERIFICATION.json',verify),('ANALYSIS.json',result)]:
        (src/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(convergence=convergence,means=[v for v in means if v['group'] in ['all','high']],
        contrasts=[v for v in contrasts if v['group'] in ['all','high']],sharing=image_results,null=null_results),ensure_ascii=False))


if __name__=='__main__':main()
