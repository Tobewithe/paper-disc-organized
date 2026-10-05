"""S051 all-image spatial contrasts and descriptive both-fill repair cases."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
import argparse,json,shutil
import numpy as np,pandas as pd
from summarize_subtype_neighbor import stat,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();src=a.source;out=src/'regional_followup';d=pd.read_csv(out/'regions.csv')
    metrics=['delta_iou']+[c for c in d if c.endswith(('_corrected','_harmed','_net'))]
    avg=d.groupby(['error_type','area_bin','image_id','combination','fill','mode'])[metrics].mean();paired=avg.xs('neighbor',level='mode')-avg.xs('background_control',level='mode')
    rows=[]
    for contrast,z in [('neighbor_minus_original',avg.xs('neighbor',level='mode')),('background_minus_original',avg.xs('background_control',level='mode')),('neighbor_minus_background',paired)]:
        for (kind,combo,fill),q in z.reset_index().groupby(['error_type','combination','fill']):rows.append(dict(error_type=kind,combination=combo,fill=fill,contrast=contrast,metrics={k:stat(q[k]) for k in metrics}))
    candidates=[];summary=[]
    for (kind,iid,combo),q in d.groupby(['error_type','image_id','combination']):
        av=q.groupby(['mode','fill'])[metrics].mean();conditions={}
        for fill in ['texture','local_color']:
            n=av.loc[('neighbor',fill)];b=av.loc[('background_control',fill)]
            conditions[fill]=dict(absolute_iou=float(n.delta_iou),paired_iou=float(n.delta_iou-b.delta_iou),own_net=float(n.own_near_net+n.own_other_net),neighbor_net=float(n.neighbor_edited_net+n.neighbor_untouched_net))
        gain=all(v['absolute_iou']>.01 and v['paired_iou']>.01 for v in conditions.values());harm=all(v['absolute_iou']<-.01 for v in conditions.values())
        region_right=all((v['neighbor_net']>0 and v['own_net']>=-.01) if kind=='same_neighbor' else (v['own_net']>0) if kind=='target_fn' else True for v in conditions.values())
        candidates.append(dict(error_type=kind,image_id=int(iid),combination=combo,bothfill_gain=bool(gain),bothfill_harm=bool(harm),region_direction=bool(region_right),**{f'{f}_{k}':v for f,val in conditions.items() for k,v in val.items()}))
    cases=pd.DataFrame(candidates);cases.to_csv(out/'case_responses.csv',index=False)
    for (kind,combo),q in cases.groupby(['error_type','combination']):summary.append(dict(error_type=kind,combination=combo,n=len(q),bothfill_actual_and_paired_gain=int(q.bothfill_gain.sum()),also_expected_region=int((q.bothfill_gain&q.region_direction).sum()),bothfill_actual_harm=int(q.bothfill_harm.sum())))
    normal=pd.read_csv(src/'metrics.csv').drop_duplicates(['image_id','mode','fill','fill_seed']);normal=normal[normal['mode']=='neighbor']
    stable=[]
    for iid,q in normal.groupby('image_id'):
        if len(q)==6 and q.normal_segm75.all():stable.append(dict(image_id=int(iid),group=q.group.iloc[0]))
    originals=d.drop_duplicates('image_id');leak=originals[originals.error_type=='same_neighbor'];fractions=(leak.neighbor_edited_original_errors/(leak.neighbor_edited_original_errors+leak.neighbor_untouched_original_errors).clip(lower=1))
    result=dict(experiment='S051_REGION_FOLLOWUP',groups=rows,case_counts=summary,stable_normal_mask75_all6=stable,
        selected_leak_inside_edit_original_fraction=dict(n=len(fractions),mean=float(fractions.mean()),median=float(fractions.median()),min=float(fractions.min()),max=float(fractions.max())),
        caveat='Exploratory same S051 outputs. >1pp consistency and region-tolerance classifications chosen during followup, not prespecified confirmatory subgroup. Allcases retained; illustrative candidate list only. Normal6fills share a fixed model andimage, not6independent replicates.')
    dump(out/'ANALYSIS.json',result);shutil.copy2(__file__,out/Path(__file__).name)
    print('CASECOUNTS',json.dumps(summary));print('NORMAL6',json.dumps(stable));print('EDIT_FRACTION',json.dumps(result['selected_leak_inside_edit_original_fraction']))
    for r in rows:
        if r['error_type']!='background' and r['combination']=='c0p1' and r['contrast']!='background_minus_original':print('REGION',r['error_type'],r['fill'],r['contrast'],json.dumps({k:v for k,v in r['metrics'].items() if k.endswith('_net')}))
if __name__=='__main__':main()
