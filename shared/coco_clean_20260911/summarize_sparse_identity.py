"""Summarize S046 without changing target selection or fitting."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();src=a.source
    receipt=json.loads((src/'COMPLETE.json').read_text());checked=0
    for p,h in receipt['hashes'].items():
        assert hashlib.sha256((src/p.replace('\\','/')).read_bytes()).hexdigest()==h,p;checked+=1
    df=pd.read_csv(src/'metrics.csv');wi=pd.read_csv(src/'witness.csv');sol=pd.read_csv(src/'solvers.csv')
    for arm in df.arm.unique():assert len(df[df.arm==arm])==64*3
    assert not df.duplicated(['annotation_id','seed','arm']).any()
    for f in (src/'fits').glob('*.npz'):
        with np.load(f) as z:
            assert not np.intersect1d(z['C'],z['E']).size
            if bool(z['eligible']):
                for arm in ['uniform8','balanced8','own_neighbor8','own_background8','wrong_identity8']:
                    assert len(z[arm+'_coords'])==8 and not np.intersect1d(z[arm+'_coords'],z['E']).size
                assert np.array_equal(z['own_neighbor8_coords'],z['wrong_identity8_coords'])
                assert np.array_equal(1-z['own_neighbor8_labels'],z['wrong_identity8_labels'])
                for arm in ['balanced8','own_neighbor8','own_background8']:assert z[arm+'_labels'].sum()==4
    protocol=json.loads((src/'protocol.json').read_text());old=pd.read_csv(Path(protocol['prior'])/'full_masks.csv')
    ref=old[old.arm=='A'].set_index(['annotation_id','seed']);base=df[df.arm=='baseline'].set_index(['annotation_id','seed']).reindex(ref.index)
    replay=float(np.abs(base.iou_raw-ref.iou_raw_before).max());assert replay<1e-10
    cols=['iou_raw','coverage','neighbor','background','E_iou','E_bce','E_own_positive','E_neighbor_positive','E_background_positive']
    avg=df.groupby(['image_id','annotation_id','group','eligible','arm'],as_index=False)[cols].mean()
    images=sorted(wi.image_id.unique());lookup={v:k for k,v in enumerate(images)};ni=len(images)
    draws=np.random.default_rng(20260912).multinomial(ni,np.full(ni,1/ni),size=2000)
    def stats(dd,values):
        val=np.asarray(values);good=np.isfinite(val);dd=dd[good];val=val[good]
        if not len(val):return dict(n=0,mean=None,ci95=None)
        idx=np.array([lookup[i] for i in dd.image_id]);den=np.bincount(idx,minlength=ni);num=np.bincount(idx,weights=val,minlength=ni)
        ds=np.einsum('bi,i->b',draws,den,optimize=False);ns=np.einsum('bi,i->b',draws,num,optimize=False)
        vv=np.divide(ns,ds,out=np.full_like(ns,np.nan),where=ds>0)
        return dict(n=len(val),images=len(set(idx)),mean=float(val.mean()),ci95=np.nanquantile(vv,[.025,.975]).tolist())
    means=[];diffs=[]
    for scope in ['eligible','all64']:
        sel=avg[avg.eligible] if scope=='eligible' else avg
        for group in ['high','nonhigh','all']:
            groupdf=sel if group=='all' else sel[sel.group==group]
            for arm in df.arm.unique():
                rr=groupdf[groupdf.arm==arm];means.append(dict(scope=scope,group=group,arm=arm,metrics={k:stats(rr,rr[k]) for k in cols}))
            for alt,refname in [('uniform8','baseline'),('balanced8','baseline'),('own_neighbor8','baseline'),('own_background8','baseline'),
                ('wrong_identity8','baseline'),('dense_pool','baseline'),('own_neighbor8','balanced8'),('own_neighbor8','own_background8'),('own_neighbor8','wrong_identity8')]:
                left=groupdf[groupdf.arm==alt].set_index('annotation_id');right=groupdf[groupdf.arm==refname].set_index('annotation_id').reindex(left.index)
                diffs.append(dict(scope=scope,group=group,alternative=alt,reference=refname,metrics={k:stats(left,left[k]-right[k]) for k in cols}))
    counts=[]
    for group in ['high','nonhigh']:
        ww=wi[wi.group==group];counts.append(dict(group=group,targets=len(ww),eligible=int(ww.eligible.sum()),images=int(ww.image_id.nunique()),
            reasons=ww[~ww.eligible].reason.value_counts().to_dict(),pool_quantiles=np.quantile(ww[ww.eligible].cuepool,[0,.5,1]).tolist()))
    verify=dict(status='PASS',hashes=checked,coordinate_archives=64,each_sparse_arm8_unique_C_points=True,commonE_disjoint=True,
        identity_flip_exact=True,baseline_S044_maxIoUdifference=replay,all7arms64targets3seeds=True,scope='Artifact/coordinate/replay checks, not independentGPU repeat.')
    result=dict(seconds=receipt['seconds'],counts=counts,solves=len(sol),converged=int(sol.converged.sum()),means=means,contrasts=diffs,verification=verify,
        scope='GTcue selection+labels on preselectedFIT targets. Feasibilitysubset explicit and all64 unchangedfallback reported. '
            'All sparse8 cues with commonlabel-freeCresponse penalty. Densepool unequalbudgetreference. No AP/sharedtraining/deployablegain. '
            'Seedmean then2000imagebootstrap pointwiseCI, smalleligiblegroup and multiplicitynotadjusted.')
    for name,obj in [('ANALYSIS.json',result),('VERIFICATION.json',verify)]:
        (src/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    compact=[]
    for d in diffs:
        if d['scope']=='eligible' and d['group']=='high':compact.append({k:v for k,v in d.items() if k!='metrics'}|{'metrics':{k:d['metrics'][k] for k in ['iou_raw','coverage','neighbor','background','E_iou']}})
    print(json.dumps(dict(counts=counts,high_eligible=compact,verification=verify),ensure_ascii=False))

if __name__=='__main__':main()
