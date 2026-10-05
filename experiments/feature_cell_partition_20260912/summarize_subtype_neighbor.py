"""S051 primary absolute AND paired outcomes by frozen failure/size groups."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,hashlib,shutil
import numpy as np,pandas as pd
from collections import Counter

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def stat(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)];n=len(x)
    if not n:return dict(n=0,mean=None,ci95=None)
    b=x[np.random.default_rng(20260912).integers(n,size=(2000,n))].mean(1)
    return dict(n=n,mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),positive=int((x>0).sum()),negative=int((x<0).sum()))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    comp=json.loads((out/'COMPLETE.json').read_text());manifest=json.loads((out/'manifest.json').read_text());witness=json.loads((out/'WITNESS.json').read_text())
    for name,h in comp['hashes'].items():assert sha(out/name)==h,name
    d=pd.read_csv(out/'metrics.csv');meta=pd.DataFrame(manifest['pairs']).set_index('image_id');base=d[(d['mode']=='original')&(d.combination=='c0p0')].set_index('image_id')
    for k in ['error_type','area_bin','chosen_neighbor_leak_share']:d[k]=d.image_id.map(meta[k])
    metrics=['mask_iou','coverage','old_neighbor_error','background_error'];d['whole_error_per_gt']=1-d.coverage+d.old_neighbor_error+d.background_error
    for k in metrics:d['d_'+k]=d[k]-d.image_id.map(base[k])
    v=d[d.fill!='none'].groupby(['image_id','error_type','area_bin','fill','mode','combination'])[['d_'+k for k in metrics]].mean().reset_index()
    v.to_csv(out/'image_effects.csv',index=False);summaries=[]
    for grouping in ['error_type','error_size']:
        v['group_label']=v.error_type if grouping=='error_type' else v.error_type+'__'+v.area_bin
        for (label,fill,combo),q in v.groupby(['group_label','fill','combination']):
            n=q[q['mode']=='neighbor'].set_index('image_id');b=q[q['mode']=='background_control'].set_index('image_id')
            for contrast,z in [('neighbor_minus_original',n),('background_minus_original',b),('neighbor_minus_background',n[['d_'+k for k in metrics]]-b[['d_'+k for k in metrics]])]:
                summaries.append(dict(grouping=grouping,group=label,fill=fill,combination=combo,contrast=contrast,metrics={k:stat(z['d_'+k]) for k in metrics}))
    dom=[]
    for (kind,fill),q in v.groupby(['error_type','fill']):
        z=q.pivot(index='image_id',columns=['mode','combination'],values='d_mask_iou')
        for contrast in ['neighbor_minus_original','neighbor_minus_background']:
            c=z[('neighbor','c1p0')];p=z[('neighbor','c0p1')]
            if contrast.endswith('background'):c=c-z[('background_control','c1p0')];p=p-z[('background_control','c0p1')]
            dom.append(dict(error_type=kind,fill=fill,contrast=contrast,coefficient_minus_prototype=stat(c-p)))
    normal=[];nd=d.drop_duplicates(['image_id','mode','fill','fill_seed']).copy()
    for k in ['normal_bbox75','normal_segm75']:nd['d_'+k]=nd[k].astype(float)-nd.image_id.map(base[k].astype(float))
    for (kind,fill),q in nd[nd.fill!='none'].groupby(['error_type','fill']):
        z=q.groupby(['image_id','mode'])[['d_normal_bbox75','d_normal_segm75']].mean();n=z.xs('neighbor',level='mode');b=z.xs('background_control',level='mode')
        normal.append(dict(error_type=kind,fill=fill,neighbor_mask75=stat(n.d_normal_segm75),background_mask75=stat(b.d_normal_segm75),paired_mask75=stat(n.d_normal_segm75-b.d_normal_segm75),neighbor_box75=stat(n.d_normal_bbox75),paired_box75=stat(n.d_normal_bbox75-b.d_normal_bbox75)))
    counts=[]
    for (kind,size),q in meta.groupby(['error_type','area_bin']):
        bs=base[base.index.isin(q.index)];shares=q.chosen_neighbor_leak_share.dropna()
        counts.append(dict(error_type=kind,area_bin=size,selected=len(q),evaluated=len(bs),box90=int((bs.original_box_iou>=.9).sum()),mask_iou_mean=float(bs.original_mask_iou.mean()) if len(bs) else None,
            crop_min=float(bs.exact_crop_coverage.min()) if len(bs) else None,edit_fraction_median=float(q.edit_fraction.median()),leak_share_min=float(shares.min()) if len(shares) else None,
            categories={str(k):int(n) for k,n in q.category.value_counts().items()},rejected=dict(Counter(r['reason'] for r in manifest['rejected'] if r['group']==kind+'__'+size))))
    result=dict(experiment='S051',seconds=comp['seconds'],counts=counts,summaries=summaries,branch_dominance=dom,normal=normal,
        verification=dict(original_artifacts=len(comp['hashes']),own_input_difference=float(d.own_input_difference.max()),exact_crop_excluded=[r for r in witness if r['status']!='COMPLETE']),
        scope='New S051 local-input observations on already explored val; originalerror,size quota frozen. Quotas characterize these cells, notpopulation or category-adjusted cross-subtype causality. Threefillseeds averaged/image then2000pairedbootstrap. Absolute and backgroundcontrasts bothrequired, pointwise exploratory CIs, no training/methodAP.')
    dump(out/'ANALYSIS.json',result);shutil.copy2(__file__,out/Path(__file__).name)
    print('COUNTS',json.dumps(counts));print('VERIFY',json.dumps(result['verification']))
    for r in summaries:
        if r['grouping']=='error_type' and r['contrast']!='background_minus_original' and r['combination'] in ['c1p0','c0p1','c1p1']:
            print('MAIN',r['group'],r['fill'],r['combination'],r['contrast'],json.dumps(r['metrics']))
    print('NORMAL',json.dumps(normal))
if __name__=='__main__':main()
