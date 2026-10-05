"""Analyze fixed S047 manifest, retaining harm/missing controls and tiny groups."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def stat(v,seed=20260912):
    v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
    if not len(v):return dict(n=0,mean=None,ci95=None)
    if len(v)==1:return dict(n=1,mean=float(v[0]),ci95=None)
    r=np.random.default_rng(seed);boot=v[r.integers(len(v),size=(2000,len(v)))].mean(1)
    return dict(n=len(v),mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),positive=int((v>0).sum()),negative=int((v<0).sum()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out
    comp=json.loads((out/'COMPLETE.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    checks={k:sha(out/k)==v for k,v in comp['hashes'].items()}
    if not all(checks.values()):raise RuntimeError([k for k,v in checks.items() if not v])
    df=pd.read_csv(out/'metrics.csv');pairs=pd.DataFrame([{k:v for k,v in q.items() if k!='placement'} for q in manifest['pairs']])
    normal=df.drop_duplicates(['image_id','mode','fill','fill_seed']).copy()
    targetinfo=normal[normal['mode']=='original'].set_index('image_id')
    fixed=df[df['combination']!='no_source'].copy();reference=fixed[fixed['mode']=='original'].drop_duplicates('image_id').set_index('image_id')
    metrics=['mask_iou','coverage','old_neighbor_error','background_error']
    normalmetrics=['normal_bbox75','normal_segm75','normal_box_iou','normal_mask_iou_at_box_match']
    for k in normalmetrics:
        normal[k]=normal[k].astype(float);normal['d_'+k]=normal[k]-normal.image_id.map(targetinfo[k].astype(float))
    for k in metrics:fixed['d_'+k]=fixed[k]-fixed.image_id.map(reference[k])
    strata={}
    for relation in ['all','same_near','different_near','same_far']:
        for state in ['all','good_box_bad_mask','good_both','bad_box','no_original_match','high','box90_badmask']:
            sub=targetinfo if relation=='all' else targetinfo[targetinfo.relation==relation]
            if state=='high':sub=sub[sub.ici>.5]
            elif state=='box90_badmask':sub=sub[(sub.original_box_iou>=.9)&(sub.original_mask_iou<.75)]
            elif state!='all':sub=sub[sub.status==state]
            strata[(relation,state)]=set(sub.index)
    comparisons=[];perimage=[]
    for fill in ['texture','local_color']:
        for combination in ['z0_b1','z1_b0','z1_b1','interaction']:
            if combination=='interaction':
                src=fixed[fixed.fill==fill];values=[]
                for key,x in src.groupby(['image_id','mode','fill_seed']):
                    x=x.set_index('combination')
                    values.append(dict(image_id=key[0],mode=key[1],fill_seed=key[2],**{k:x.loc['z1_b1',k]-x.loc['z1_b0',k]-x.loc['z0_b1',k]+x.loc['z0_b0',k] for k in metrics}))
                part=pd.DataFrame(values)
            else:
                part=fixed[(fixed.fill==fill)&(fixed.combination==combination)][['image_id','mode','fill_seed']+['d_'+k for k in metrics]].rename(columns={'d_'+k:k for k in metrics})
            means=part.groupby(['image_id','mode'])[metrics].mean()
            modes={name:means.xs(name,level='mode') for name in ['neighbor','background_control'] if name in means.index.get_level_values('mode')}
            for contrast in ['neighbor_minus_original','control_minus_original','neighbor_minus_control']:
                if contrast=='neighbor_minus_control':
                    use=modes['neighbor'].index.intersection(modes['background_control'].index);effects=modes['neighbor'].loc[use]-modes['background_control'].loc[use]
                else:effects=modes['neighbor' if contrast=='neighbor_minus_original' else 'background_control']
                for iid,r in effects.iterrows():perimage.append(dict(image_id=int(iid),fill=fill,combination=combination,contrast=contrast,**r.to_dict()))
                for (relation,state),ids in strata.items():
                    sub=effects.loc[effects.index.intersection(sorted(ids))]
                    if not len(sub):continue
                    comparisons.append(dict(fill=fill,combination=combination,contrast=contrast,relation=relation,state=state,
                        metrics={k:stat(sub[k]) for k in metrics}))
    normalcomp=[]
    for fill in ['texture','local_color']:
        x=normal[normal.fill==fill].groupby(['image_id','mode'])[['d_'+k for k in normalmetrics]].mean()
        for mode in ['neighbor','background_control','paired_difference']:
            v=x.xs('neighbor',level='mode') if mode=='neighbor' else x.xs('background_control',level='mode')
            if mode=='paired_difference':
                n=x.xs('neighbor',level='mode');keys=n.index.intersection(v.index);v=n.loc[keys]-v.loc[keys]
            for (relation,state),ids in strata.items():
                sub=v.loc[v.index.intersection(sorted(ids))]
                if len(sub):normalcomp.append(dict(fill=fill,mode=mode,relation=relation,state=state,metrics={k:stat(sub['d_'+k]) for k in normalmetrics}))
    # Cross-relation comparisons restricted to measured shared composition cells.
    effects=pd.DataFrame(perimage);matchresults=[]
    infos=targetinfo.reset_index()[['image_id','relation','status','original_box_iou','ici']].merge(pairs[['image_id','category','area','edit_pixels']],on='image_id')
    infos['area_bin']=pd.cut(infos.area,[0,1024,9216,float('inf')],labels=['small','medium','large'],right=False).astype(str)
    infos['box_bin']=pd.cut(infos.original_box_iou,[-.01,.5,.75,.9,1.001],labels=['no_or_low','mid','good','verygood'],right=False).astype(str)
    # Exact category/area/box cells; report retained count instead of pretending balance.
    infos['cell']=infos.category.astype(str)+'/'+infos.area_bin+'/'+infos.box_bin
    for fill in ['texture','local_color']:
        e=effects[(effects.fill==fill)&(effects.combination=='z1_b0')&(effects.contrast=='neighbor_minus_control')].merge(infos,on='image_id')
        for comparator in ['different_near','same_far']:
            one=e[e.relation=='same_near'];two=e[e.relation==comparator];common=set(one.cell)&set(two.cell)
            cellrows=[]
            for cell in sorted(common):
                x=one[one.cell==cell];y=two[two.cell==cell]
                cellrows.append(dict(cell=cell,n_same=len(x),n_other=len(y),delta=float(x.mask_iou.mean()-y.mask_iou.mean())))
            matchresults.append(dict(fill=fill,comparator=comparator,common_cells=cellrows,
                retained_same=int(one.cell.isin(common).sum()),retained_other=int(two.cell.isin(common).sum()),
                limits='Exact category/size/box cells only; edit amount/distance/texture remain unmatched. Very small support, no causal relation interaction claim.'))
    counts=[]
    for relation in ['same_near','different_near','same_far']:
        q=targetinfo[targetinfo.relation==relation]
        counts.append(dict(relation=relation,images=len(q),states=dict(Counter(q.status)),high=int((q.ici>.5).sum()),
            fixed_source=int((q.original_source>=0).sum()),paired_control=int(q.control_available.sum()),
            edited_fraction_quantiles=np.quantile(q.edit_fraction,[0,.5,1]).tolist()))
    # Descriptive feature movement versus actual fixed-box response change.
    feature_summary=[]
    featurekeys=[k for k in fixed if k.endswith('_relative_change') or k=='coefficient_cosine']
    for (relation,fill),sub in fixed[(fixed['mode']=='neighbor')&(fixed.combination=='z1_b0')].groupby(['relation','fill']):
        byimg=sub.groupby('image_id')[featurekeys+['d_mask_iou']].mean()
        feature_summary.append(dict(relation=relation,fill=fill,n=len(byimg),means={k:float(byimg[k].mean()) for k in featurekeys},
            limits='Magnitude of feature movement is descriptive, not evidence that moved feature is the failure cause.'))
    analysis=dict(experiment='S047',seconds=comp['seconds'],counts=counts,comparisons=comparisons,normal=normalcomp,
        composition_common_support=matchresults,feature_movement=feature_summary,
        verification=dict(hashes=len(checks),all_match=True,images=64,own_input_max=float(df.own_input_max_difference.max()),
            no_op_raw_proto_exact=True,stock_decoder_exact=True),
        limits='64 GT-geometry selected train2017 images, not pristine pretrained holdout. 3 fillseeds averaged perimage. Pointwise imagebootstrap2000, no multiplicitycorrection, no AP claim. Normal GT omits manipulated neighbor B in all arms, preserves other GT. Missing exact background controls retained but excluded from paired difference. Tiny goodboxbadmask groups and weak shared composition support prevent broad causal conclusion.')
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    effects.to_csv(out/'paired_image_effects.csv',index=False);infos.to_csv(out/'sample_composition.csv',index=False)
    print('COUNTS',json.dumps(counts))
    for c in comparisons:
        if c['relation']=='same_near' and c['state'] in ['all','good_box_bad_mask'] and c['contrast']=='neighbor_minus_control' and c['combination'] in ['z0_b1','z1_b0','z1_b1']:
            print('KEY',c['fill'],c['combination'],c['state'],json.dumps(c['metrics']))
    for c in normalcomp:
        if c['relation']=='same_near' and c['state']=='all':print('NORMAL',c['fill'],c['mode'],json.dumps(c['metrics']))


if __name__=='__main__':main()
