"""S049 image-paired branch effects and selection accountability."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd


def stat(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)];n=len(v)
    if not n:return dict(n=0,mean=None,ci95=None)
    r=np.random.default_rng(20260912);b=v[r.integers(n,size=(2000,n))].mean(1)
    return dict(n=n,mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),improved=int((v>0).sum()),worse=int((v<0).sum()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out
    d=pd.read_csv(out/'metrics.csv');manifest=json.loads((out/'manifest.json').read_text());witness=json.loads((out/'WITNESS.json').read_text())
    comp=json.loads((out/'COMPLETE.json').read_text());source_count=0
    # Verify main measurements, manifests and perimage tensors; all saved bytes hashed.
    for name,h in comp['hashes'].items():
        hh=hashlib.sha256()
        with (out/name).open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):hh.update(chunk)
        if hh.hexdigest()!=h:raise RuntimeError('Changedartifact '+name)
        source_count+=1
    base=d[(d['mode']=='original')&(d.combination=='c0p0')].set_index('image_id')
    metrics=['mask_iou','coverage','old_neighbor_error','background_error'];rows=[]
    for k in metrics:d['d_'+k]=d[k]-d.image_id.map(base[k])
    by=d[d.fill!='none'].groupby(['group','image_id','fill','mode','combination'])[['d_'+k for k in metrics]].mean().reset_index()
    for (group,fill,combo),q in by.groupby(['group','fill','combination']):
        neighbor=q[q['mode']=='neighbor'].set_index('image_id');control=q[q['mode']=='background_control'].set_index('image_id')
        for contrast,v in [('neighbor_minus_original',neighbor),('control_minus_original',control),('neighbor_minus_control',neighbor[['d_'+k for k in metrics]]-control[['d_'+k for k in metrics]])]:
            rows.append(dict(group=group,fill=fill,combination=combo,contrast=contrast,metrics={k:stat(v['d_'+k]) for k in metrics}))
    interactions=[]
    for (group,fill),q in by.groupby(['group','fill']):
        effects=[]
        for (iid,mode),x in q.groupby(['image_id','mode']):
            xx=x.set_index('combination');effects.append(dict(image_id=iid,mode=mode,**{k:xx.loc['c1p1','d_'+k]-xx.loc['c1p0','d_'+k]-xx.loc['c0p1','d_'+k] for k in metrics}))
        e=pd.DataFrame(effects);delta=e[e['mode']=='neighbor'].set_index('image_id')[metrics]-e[e['mode']=='background_control'].set_index('image_id')[metrics]
        interactions.append(dict(group=group,fill=fill,metrics={k:stat(delta[k]) for k in metrics}))
    # Dominance paired in each image, not inferred from overlapping marginal CIs.
    dominance=[]
    for (group,fill),q in by.groupby(['group','fill']):
        z=q.pivot(index='image_id',columns=['mode','combination'],values='d_mask_iou')
        c=z[('neighbor','c1p0')]-z[('background_control','c1p0')];p=z[('neighbor','c0p1')]-z[('background_control','c0p1')]
        dominance.append(dict(group=group,fill=fill,coefficient_minus_prototype=stat(c-p)))
    normal=[];nd=d.drop_duplicates(['image_id','mode','fill','fill_seed']).copy()
    for k in ['normal_bbox75','normal_segm75']:
        nd[k]=nd[k].astype(float);nd['d_'+k]=nd[k]-nd.image_id.map(base[k].astype(float))
    for (group,fill),q in nd[nd.fill!='none'].groupby(['group','fill']):
        v=q.groupby(['image_id','mode'])[['d_normal_bbox75','d_normal_segm75']].mean();n=v.xs('neighbor',level='mode');b=v.xs('background_control',level='mode')
        normal.append(dict(group=group,fill=fill,bbox75=stat(n.d_normal_bbox75-b.d_normal_bbox75),mask75=stat(n.d_normal_segm75-b.d_normal_segm75),
            neighbor_mask75_delta=stat(n.d_normal_segm75),control_mask75_delta=stat(b.d_normal_segm75)))
    counts=[]
    for group in ['same_failure','same_success','different_failure']:
        ids=[q for q in manifest['pairs'] if q['group']==group];valid=base[base.group==group]
        counts.append(dict(group=group,selected=len(ids),evaluated=len(valid),images=len(valid),strictbox90=int((valid.original_box_iou>=.9).sum()),
            original_mean_maskiou=float(valid.original_mask_iou.mean()) if len(valid) else None,
            exact_crop_min=float(valid.exact_crop_coverage.min()) if len(valid) else None,
            rejected_during_selection=dict(Counter(q['reason'] for q in manifest['rejected'] if q['group']==group)),
            categories={str(k):v for k,v in Counter(q['category'] for q in ids).items()},
            radial_mismatch_max=max(q['placement']['radial_mismatch_input'] for q in ids)))
    result=dict(experiment='S049',seconds=comp['seconds'],counts=counts,contrasts=rows,interaction=interactions,branch_dominance=dominance,normal=normal,
        verification=dict(artifacts=source_count,own_input_difference=float(d.own_input_difference.max()),original_replay_all=True,
            exact_crop_exclusions=[q for q in witness if q['status']!='COMPLETE'],fp64_logit_identity=True,noop=True),
        scope='Selected failure/success instances on exploredCOCOval;fixed model and original source,crop>=95%.3fillseeds averaged perimage then2000pairedbootstrap,pointwise no multiplicityadjustment;notnewmethodAP. CrossinputcP hybridsoffmanifoldpossible,upstreamcouplingnotexcluded. Backgroundcontrolgeometry constrained buttextures/protectedboundary/partialneighborresidual remain.')
    (out/'ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');by.to_csv(out/'image_effects.csv',index=False)
    print('COUNTS',json.dumps(counts));print('VERIFY',json.dumps(result['verification']))
    for r in rows:
        if r['group']=='same_failure' and r['contrast']=='neighbor_minus_control':print('KEY',r['fill'],r['combination'],json.dumps(r['metrics']))
    print('DOMINANCE',json.dumps(dominance));print('NORMAL',json.dumps(normal))


if __name__=='__main__':main()
