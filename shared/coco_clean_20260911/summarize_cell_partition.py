"""S053 image-level paired summaries; subset seeds do not multiply sample size."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
import argparse,json,shutil
import numpy as np,pandas as pd
from summarize_subtype_neighbor import stat,dump,sha

METRICS=['mask_iou','coverage','neighbor_error','background_error','own_near_tp','neighbor_untouched_fp']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;comp=json.loads((out/'COMPLETE.json').read_text());d=pd.read_csv(out/'metrics.csv');manifest=json.loads((out/'manifest.json').read_text())['pairs']
    for p,h in comp['hashes'].items():assert sha(out/p)==h,p
    assert set(d.image_id)=={p['image_id'] for p in manifest};assert len(d)==15*52
    # Retain image means of the three geometry subsets; each image has weight one.
    cols=['delta_'+m for m in METRICS];v=d.groupby(['image_id','mode','fill','region','direction'],sort=True)[cols+['delta_norm','patch_cells']].mean().reset_index();v.to_csv(out/'image_effects.csv',index=False)
    absolute=[]
    for (mode,fill,region,direction),q in v.groupby(['mode','fill','region','direction']):absolute.append(dict(mode=mode,fill=fill,region=region,direction=direction,metrics={m:stat(q['delta_'+m]) for m in METRICS},norm=stat(q.delta_norm),cells=stat(q.patch_cells)))
    comparisons=[]
    for fill in ['texture','local_color']:
        for direction in ['insert','restore']:
            q=v[(v.fill==fill)&(v.direction==direction)].pivot(index='image_id',columns=['mode','region'],values=cols)
            for label,terms in [
                ('equal_mixed_minus_exclusive',[(1,'neighbor','equal_mixed'),(-1,'neighbor','equal_exclusive')]),
                ('equal_mixed_minus_background',[(1,'neighbor','equal_mixed'),(-1,'background_control','equal_background')]),
                ('equal_exclusive_minus_background',[(1,'neighbor','equal_exclusive'),(-1,'background_control','equal_background')]),
                ('full_mixed_minus_exclusive',[(1,'neighbor','mixed_all'),(-1,'neighbor','exclusive_all')]),
                ('partition_interaction',[(1,'neighbor','all_edit'),(-1,'neighbor','mixed_all'),(-1,'neighbor','exclusive_all')]),
                ('full_neighbor_minus_background',[(1,'neighbor','all_edit'),(-1,'background_control','all_edit')])]:
                comparisons.append(dict(fill=fill,direction=direction,contrast=label,metrics={m:stat(sum(coef*q[('delta_'+m,mode,region)] for coef,mode,region in terms)) for m in METRICS}))
    # Individual outcomes are descriptive, and fixed cohort includes adverse cases.
    cases=[]
    for iid in sorted(v.image_id.unique()):
        q=v[v.image_id==iid].set_index(['mode','fill','region','direction']);r=dict(image_id=int(iid));pass_regions={}
        for region in ['equal_mixed','equal_exclusive']:
            flags=[]
            for fill in ['texture','local_color']:
                ni=q.loc[('neighbor',fill,region,'insert'),'delta_mask_iou'];nr=q.loc[('neighbor',fill,region,'restore'),'delta_mask_iou'];bi=q.loc[('background_control',fill,'equal_background','insert'),'delta_mask_iou'];br=q.loc[('background_control',fill,'equal_background','restore'),'delta_mask_iou']
                flags.append(ni>.01 and nr<-.01 and ni-bi>.01 and nr-br<-.01)
                r.update({f'{region}_{fill}_insert':float(ni),f'{region}_{fill}_restore':float(nr),f'{region}_{fill}_paired_insert':float(ni-bi),f'{region}_{fill}_paired_restore':float(nr-br)})
            pass_regions[region]=all(flags);r[region+'_bothfill_reversible_over1pp']=bool(all(flags))
        cases.append(r)
    pd.DataFrame(cases).to_csv(out/'CASE_RESPONSES.csv',index=False)
    geometry=[dict(image_id=p['image_id'],k=p['k'],mixed=p['mixed_cells'],exclusive=p['exclusive_cells'],background=p['background_cells']) for p in manifest]
    witness=json.loads((out/'WITNESS.json').read_text());checks=dict(original_hashes=len(comp['hashes']),images=len(manifest),selfswaps=sum(p['selfswaps'] for p in witness),S052_mask_replays=sum(p['S052_local_mask_replays'] for p in witness),all_image_weights_equal=True)
    result=dict(experiment='S053',seconds=comp['seconds'],rows=comp['rows'],absolute=absolute,comparisons=comparisons,geometry=geometry,verification=checks,
        descriptive_counts={r:sum(x[r+'_bothfill_reversible_over1pp'] for x in cases) for r in ['equal_mixed','equal_exclusive']},
        scope='One fixed15image cohort, one saved fillseed perfilltype. Three geometry subsets averaged withinimage. Pointwise2000pairedimagebootstrap exploratory; no multiplicity correction. Equalcells not equal activationchange/position; target-disjoint grid not disjoint RF. Restore reference is each edited endpoint.')
    dump(out/'ANALYSIS.json',result);shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(Path(__file__).parent/'summarize_subtype_neighbor.py',out/'summarize_subtype_neighbor.py')
    print('VERIFICATION',json.dumps(checks))
    for r in absolute:
        if r['region'].startswith('equal_'):print('EQUAL',r['fill'],r['region'],r['direction'],json.dumps(r['metrics']['mask_iou']))
    for r in comparisons:
        if r['contrast'].startswith('equal_'):print('CONTRAST',r['fill'],r['direction'],r['contrast'],json.dumps(r['metrics']['mask_iou']))
if __name__=='__main__':main()
