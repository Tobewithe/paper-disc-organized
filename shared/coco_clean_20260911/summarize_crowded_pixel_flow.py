"""S050 descriptive subtypes and paired pixel-flow statistics from all saved rows."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,shutil
import numpy as np,pandas as pd

REGIONS=['own_interior','own_boundary_neighbor','own_boundary_other','chosen_neighbor','other_gt','background_near','background_far']
def stat(x):
    v=np.asarray(x,float);v=v[np.isfinite(v)];n=len(v)
    if not n:return dict(n=0,mean=None,ci95=None)
    b=v[np.random.default_rng(20260912).integers(n,size=(2000,n))].mean(1)
    return dict(n=n,mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]).tolist())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    d=pd.read_csv(out/'pixel_flows.csv');meta=d.drop_duplicates('image_id').set_index('image_id')[['group','target','neighbor','dominant_error','area_bin','box90','original_raw_iou']]
    metrics=['raw_iou_delta','valid_iou_delta','all_corrected','all_harmed']+[r+'_'+s for r in REGIONS for s in ['corrected','harmed','net_corrected','iou_term']]
    v=d.groupby(['image_id','fill','mode','combination'])[metrics].mean();effects=v.xs('neighbor',level='mode')-v.xs('background_control',level='mode')
    effects=effects.reset_index().merge(meta,left_on='image_id',right_index=True,validate='many_to_one');effects.to_csv(out/'paired_image_effects.csv',index=False)
    absolute=[]
    absolute_v=v.reset_index().merge(meta,left_on='image_id',right_index=True,validate='many_to_one')
    for (group,fill,mode,combo),q in absolute_v.groupby(['group','fill','mode','combination']):
        absolute.append(dict(group=group,fill=fill,mode=mode,combination=combo,metrics={k:stat(q[k]) for k in metrics}))
    groups=[]
    for (group,fill,combo),q in effects.groupby(['group','fill','combination']):
        groups.append(dict(group=group,fill=fill,combination=combo,metrics={k:stat(q[k]) for k in metrics}))
    subgroups=[]
    for grouping in ['dominant_error','area_bin','box90']:
        for (subgroup,fill,combo),q in effects[effects.group=='same_failure'].groupby([grouping,'fill','combination']):
            subgroups.append(dict(grouping=grouping,subgroup=str(subgroup),fill=fill,combination=combo,metrics={k:stat(q[k]) for k in metrics}))
    consistency=[];caserows=[]
    for (group,combo),q in effects.groupby(['group','combination']):
        pair=q.pivot(index='image_id',columns='fill',values='raw_iou_delta');labels=np.where((pair.texture>.01)&(pair.local_color>.01),'gain_both',np.where((pair.texture<-.01)&(pair.local_color<-.01),'harm_both','mixed_or_small'))
        for iid,label in zip(pair.index,labels):
            caserows.append(dict(group=group,combination=combo,image_id=int(iid),status=label,texture=float(pair.loc[iid,'texture']),local_color=float(pair.loc[iid,'local_color']),**meta.loc[iid].drop('group').to_dict()))
        consistency.append(dict(group=group,combination=combo,images=len(pair),gain_both=int((labels=='gain_both').sum()),harm_both=int((labels=='harm_both').sum()),mixed_or_small=int((labels=='mixed_or_small').sum()),cross_fill_correlation=float(pair.texture.corr(pair.local_color))))
    cases=pd.DataFrame(caserows);cases.to_csv(out/'fill_consistency.csv',index=False)
    # Original failure typing predates the treatment. Response classifications remain exploratory.
    cross=[]
    for key in ['dominant_error','area_bin','box90']:
        z=cases[cases.group=='same_failure'].groupby(['combination',key,'status']).size()
        cross.extend(dict(grouping=key,combination=c,subgroup=str(s),status=t,n=int(n)) for (c,s,t),n in z.items())
    result=dict(experiment='S050',scope='Same S049 63 images, 32 primary failures; saved-tensor spatial replay only. All subgroups and consistency are exploratory, pointwise bootstrap CI uncorrected. No independent causal or method confirmation.',
        n_images=len(meta),n_primary=int((meta.group=='same_failure').sum()),primary_composition={key:{str(k):int(v) for k,v in meta[meta.group=='same_failure'][key].value_counts().items()} for key in ['dominant_error','area_bin','box90']},
        groups=groups,absolute=absolute,subgroups=subgroups,consistency=consistency,consistency_by_subgroup=cross,
        exact_identity='valid IoU change = sum own deltaTP/(A+FP1) - sum other TP0*deltaFP/((A+FP0)*(A+FP1)); endpoint-dependent accounting, not unique causal responsibility. Raw COCO IoU keeps crowd pixels as in S049, valid excludes crowd for regional accounting.')
    dump(out/'ANALYSIS.json',result);shutil.copy2(__file__,out/Path(__file__).name)
    print('COMPOSITION',json.dumps(result['primary_composition']));print('CONSISTENCY',json.dumps([r for r in consistency if r['group']=='same_failure']))
    for row in groups:
        if row['group']=='same_failure':
            print('PRIMARY',row['fill'],row['combination'],json.dumps({k:v for k,v in row['metrics'].items() if k in ['raw_iou_delta','all_corrected','all_harmed'] or k.endswith('_net_corrected') or k.endswith('_iou_term')}))
    for r in subgroups:
        if r['grouping']=='dominant_error':print('SUBTYPE',r['subgroup'],r['fill'],r['combination'],json.dumps(r['metrics']['raw_iou_delta']))
    for r in absolute:
        if r['group']=='same_failure' and r['combination']=='c0p1':print('ABSOLUTE_P',r['fill'],r['mode'],json.dumps({k:v for k,v in r['metrics'].items() if k in ['raw_iou_delta','own_boundary_neighbor_net_corrected','own_boundary_other_net_corrected','own_boundary_neighbor_corrected','own_boundary_neighbor_harmed']}))
if __name__=='__main__':main()
