"""S052 absolute and controlled insertion/reversal effects, no outcome exclusions."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,shutil
import numpy as np,pandas as pd
from summarize_subtype_neighbor import stat,dump,sha

METRICS=['mask_iou','coverage','neighbor_error','background_error','own_near_tp','neighbor_untouched_fp']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    comp=json.loads((out/'COMPLETE.json').read_text());d=pd.read_csv(out/'metrics.csv');wit=json.loads((out/'WITNESS.json').read_text())
    for name,h in comp['hashes'].items():assert sha(out/name)==h,name
    endpoint=d[d.stage=='all_inputs'];checks=[]
    for (iid,mode,fill),q in endpoint.groupby(['image_id','mode','fill']):
        ins=q[q.direction=='insert'].iloc[0];rev=q[q.direction=='restore'].iloc[0]
        assert abs(ins.delta_mask_iou+rev.delta_mask_iou)<1e-12
        for stage in ['fused','decoder']:
            x=d[(d.image_id==iid)&(d['mode']==mode)&(d.fill==fill)&(d.stage==stage)&(d.region=='full')]
            assert np.allclose(x[x.direction=='insert'][METRICS].iloc[0],ins[METRICS].astype(float),atol=1e-12,rtol=0)
            assert np.allclose(x[x.direction=='restore'][METRICS].iloc[0],rev[METRICS].astype(float),atol=1e-12,rtol=0)
        checks.append(dict(image_id=int(iid),mode=mode,fill=fill,full_downstream_exact=True))
    result=[]
    for grouping in ['error_type','error_size']:
        d['group_label']=d.error_type if grouping=='error_type' else d.error_type+'__'+d.area_bin
        for (group,fill,stage,region,direction),q in d.groupby(['group_label','fill','stage','region','direction']):
            n=q[q['mode']=='neighbor'].set_index('image_id');b=q[q['mode']=='background_control'].set_index('image_id')
            for contrast,z in [('neighbor',n),('background',b),('neighbor_minus_background',n[['delta_'+k for k in METRICS]]-b[['delta_'+k for k in METRICS]])]:
                result.append(dict(grouping=grouping,group=group,fill=fill,stage=stage,region=region,direction=direction,contrast=contrast,
                    metrics={k:stat(z['delta_'+k]) for k in METRICS}))
    geom=[]
    for p in sorted((out/'pairs').glob('*/patch_geometry.json')):
        iid=int(p.parent.name);meta=d[d.image_id==iid].iloc[0]
        for r in json.loads(p.read_text()):geom.append(dict(image_id=iid,error_type=meta.error_type,area_bin=meta.area_bin,**r))
    g=pd.DataFrame(geom);g['target_overlap_fraction']=g.overlap_cells/g.target_cells.clip(lower=1);g['edit_overlap_fraction']=g.overlap_cells/g.edit_cells.clip(lower=1);g.to_csv(out/'PATCH_GEOMETRY.csv',index=False)
    geometry=[]
    for (kind,mode,stage),q in g[g.fill=='texture'].groupby(['error_type','mode','stage']):geometry.append(dict(error_type=kind,mode=mode,stage=stage,n=len(q),images_with_overlap=int((q.overlap_cells>0).sum()),mean_target_overlap=float(q.target_overlap_fraction.mean()),mean_edit_overlap=float(q.edit_overlap_fraction.mean())))
    # Cases are all reported, but consistency classification is descriptive after observation.
    case_rows=[]
    for (iid,stage,region),q in d[d.stage!='all_inputs'].groupby(['image_id','stage','region']):
        if len(q)!=8:raise RuntimeError('incomplete4inputs2directions')
        z=q.set_index(['mode','fill','direction']);passing=[]
        r=dict(image_id=int(iid),error_type=q.error_type.iloc[0],area_bin=q.area_bin.iloc[0],stage=stage,region=region)
        for fill in ['texture','local_color']:
            ni=z.loc[('neighbor',fill,'insert'),'delta_mask_iou'];nr=z.loc[('neighbor',fill,'restore'),'delta_mask_iou'];bi=z.loc[('background_control',fill,'insert'),'delta_mask_iou'];br=z.loc[('background_control',fill,'restore'),'delta_mask_iou']
            r.update({fill+'_insert':float(ni),fill+'_restore':float(nr),fill+'_paired_insert':float(ni-bi),fill+'_paired_restore':float(nr-br)})
            passing.append(ni>.01 and nr<-.01 and ni-bi>.01 and nr-br<-.01)
        r['both_fill_reversible_over1pp']=bool(all(passing));case_rows.append(r)
    cases=pd.DataFrame(case_rows);cases.to_csv(out/'CASE_RESPONSES.csv',index=False)
    counts=[dict(error_type=k,stage=s,region=r,n=len(q),bothfill_reversible=int(q.both_fill_reversible_over1pp.sum())) for (k,s,r),q in cases.groupby(['error_type','stage','region'])]
    output=dict(experiment='S052',images=comp['images'],seconds=comp['seconds'],rows=comp['rows'],summary=result,geometry=geometry,reversible_case_counts=counts,
        verification=dict(original_files=len(comp['hashes']),checks=len(checks),plain_module_replay=all(x['plain_module_replay'] for x in wit),selfswap_total=sum(x['stage_selfswaps_exact'] for x in wit)),
        interpretation='All S051targets, original c0/b0 fixed, one fillseed each, exploratory pointwise imageCI. Restore delta is restored-minus-edited, so negative cancels an edited gain. Full fused/decoder equal endpoint byconstruction; do not call them independent evidence. Target cells can include neighbor pixels and receptive fields overlap. Reversible stage identifies a carrier, not root cause or method.')
    dump(out/'ANALYSIS.json',output);dump(out/'REPLAY_CHECKS.json',checks);shutil.copy2(__file__,out/Path(__file__).name)
    print('VERIFY',json.dumps(output['verification']));print('REVERSIBLE',json.dumps([x for x in counts if x['bothfill_reversible']>0]))
    for r in result:
        if r['grouping']=='error_type' and r['group'] in ['same_neighbor','target_fn'] and r['region']=='target' and r['contrast']=='neighbor' and r['direction']=='insert':print('TARGET_INSERT',r['group'],r['fill'],r['stage'],json.dumps(r['metrics']['mask_iou']))
if __name__=='__main__':main()
