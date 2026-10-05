"""S019 fixed-cohort factorial effects; paired image-cluster uncertainty."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

ARMS=['original','A160_GT','B160_PRED','C640_GT','D640_PRED']
CONTRASTS={
    'grid_GT':{'C640_GT':1,'A160_GT':-1},
    'grid_PRED':{'D640_PRED':1,'B160_PRED':-1},
    'support_160':{'B160_PRED':1,'A160_GT':-1},
    'support_640':{'D640_PRED':1,'C640_GT':-1},
    'both_D_minus_A':{'D640_PRED':1,'A160_GT':-1},
    'grid_support_interaction':{'D640_PRED':1,'C640_GT':-1,'B160_PRED':-1,'A160_GT':1}}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):
    with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    receipt=json.loads((out/'COMPLETE.json').read_text());assert receipt['status']=='COMPLETE'
    for name,h in receipt['hashes'].items():assert sha(out/name)==h,name
    rows=read(out/'targets.csv');protocol=json.loads((out/'protocol.json').read_text());aids=protocol['target_ids']
    lookup={(int(r['annotation_id']),r['arm']):r for r in rows}
    assert len(rows)==len(lookup)==len(aids)*5
    assert set(aids)=={int(r['annotation_id']) for r in rows}
    paired=[]
    for aid in aids:
        arms={arm:lookup[(aid,arm)] for arm in ARMS};base=arms['original']
        assert len({r['image_id'] for r in arms.values()})==1
        r={k:base[k] for k in ['image_id','annotation_id','density','ici','category_id','area','mixed_proto_cell_fraction','loss_support_iou','pred_support_outside_gt_fraction','neighbor_exposure_over_gt']}
        r['area_bin']=0 if float(r['area'])<1024 else 1 if float(r['area'])<9216 else 2
        r['baseline_bin']=min(int(float(base['coco_iou'])/.15),4)
        r['A_minus_S018']=float(arms['A160_GT']['coco_iou'])-float(base['s018_official_iou'])
        r['D_minus_old_oracle']=float(arms['D640_PRED']['coco_iou'])-float(base['previous_fullinput_iou'])
        for contrast,weights in CONTRASTS.items():
            for field in ['coco_iou','coverage','same_neighbor','background','valid_iou']:
                r[contrast+'_'+field]=sum(w*float(arms[arm][field]) for arm,w in weights.items()) if all(arms[arm][field]!='' for arm in weights) else None
        for arm in ARMS:r[arm+'_coco_iou']=float(arms[arm]['coco_iou'])
        paired.append(r)
    with open(out/'PAIRED_TARGETS.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    ids=sorted({int(r['image_id']) for r in paired});index={v:j for j,v in enumerate(ids)}
    rng=np.random.default_rng(20260912);boot=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    def estimate(rr,field):
        num=np.zeros(len(ids));den=np.zeros(len(ids))
        for r in rr:
            if r[field] is not None:num[index[int(r['image_id'])]]+=r[field];den[index[int(r['image_id'])]]+=1
        bd=boot@den;bn=boot@num;ok=bd>0
        values=bn[ok]/bd[ok] if ok.any() else np.array([])
        return dict(n=int(den.sum()),mean_pp=100*float(num.sum()/den.sum()) if den.sum() else None,
            ci95_pp=(100*np.quantile(values,[.025,.975])).tolist() if len(values) else None)
    contrasts=[];subgroups=[]
    for density in ['high','other']:
        selected=[r for r in paired if r['density']==density]
        for name in CONTRASTS:
            for field in ['coco_iou','coverage','same_neighbor','background']:
                contrasts.append(dict(density=density,contrast=name,metric=field,**estimate(selected,name+'_'+field)))
        for dimension in ['area','mixed_proto_cell_fraction','pred_support_outside_gt_fraction']:
            # Fixed physical thresholds, not optimized for observed outcomes.
            threshold=1024 if dimension=='area' else .5 if dimension=='mixed_proto_cell_fraction' else .1
            for high in [False,True]:
                rr=[r for r in selected if (float(r[dimension])>=threshold)==high]
                subgroups.append(dict(density=density,dimension=dimension,threshold=threshold,above_or_equal=high,
                    effects={key:estimate(rr,key+'_coco_iou') for key in ['grid_GT','support_640','both_D_minus_A']}))
    interactions=[]
    for contrast in CONTRASTS:
        field=contrast+'_coco_iou';num=np.zeros((len(ids),2));den=np.zeros_like(num)
        for r in paired:
            j=index[int(r['image_id'])];g=int(r['density']=='high');num[j,g]+=r[field];den[j,g]+=1
        bd=boot@den;bn=boot@num;ok=(bd>0).all(1);values=bn[ok]/bd[ok]
        interactions.append(dict(contrast=contrast,high_minus_other_pp=100*float(num[:,1].sum()/den[:,1].sum()-num[:,0].sum()/den[:,0].sum()),
            ci95_pp=(100*np.quantile(values[:,1]-values[:,0],[.025,.975])).tolist()))
    checks=[]
    for density in ['high','other']:
        pp=[r for r in paired if r['density']==density]
        delta=np.array([r['A_minus_S018'] for r in pp])
        checks.append(dict(density=density,n=len(pp),A_minus_S018_mean_pp=100*float(delta.mean()),A_minus_S018_max_abs_pp=100*float(abs(delta).max()),
            A_S018_recovery_disagreements=sum((r['A160_GT_coco_iou']>=.75)!=(float(lookup[(int(r['annotation_id']),'original')]['s018_official_iou'])>=.75) for r in pp),
            D_minus_old_oracle_mean_pp=100*float(np.mean([r['D_minus_old_oracle'] for r in pp]))))
    objective=[]
    for density in ['high','other']:
        for arm in ARMS[1:]:
            q=[r for r in rows if r['density']==density and r['arm']==arm]
            objective.append(dict(density=density,arm=arm,n=len(q),
                loss_lower_iou_worse=sum(float(r['loss_after'])<float(r['loss_before'])-1e-6 and float(r['coco_iou'])<float(r['original_iou'])-1e-6 for r in q),
                nonzero_gradient_cosine_min=min((float(r['official_gradient_cosine']) for r in q if r['official_gradient_cosine']!=''),default=None),
                official_loss_error_max=max(float(r['official_value_abs_error']) for r in q)))
    common=[]
    for dims in [('category_id','area_bin'),('category_id','area_bin','baseline_bin')]:
        cells={}
        for r in paired:cells.setdefault(tuple(r[k] for k in dims),{'high':[],'other':[]})[r['density']].append(r)
        both=[v for v in cells.values() if len(v['high'])>=3 and len(v['other'])>=3];total=sum(len(v['high'])+len(v['other']) for v in both)
        effects={key:{g:100*sum((len(v['high'])+len(v['other']))*np.mean([r[key+'_coco_iou'] for r in v[g]]) for v in both)/total if total else None for g in ['high','other']} for key in ['grid_GT','support_640','both_D_minus_A']}
        common.append(dict(strata=dims,min_per_group=3,common_strata=len(both),high_included=sum(len(v['high']) for v in both),other_included=sum(len(v['other']) for v in both),effects_pp=effects))
    doc=dict(contrasts=contrasts,density_interactions=interactions,subgroups=subgroups,solver_controls=checks,objective_diagnostics=objective,common_stratum_controls=common,
        bootstrap='2000 shared image-cluster resamples; percentile pointwise intervals; exploratory selected failure cohort, no multiplicity correction.',
        subgroup_scope='Pre-analysis physical threshold splits; common-stratum controls post-hoc descriptive with exclusions and no adjusted CI. Not density causality.',
        receipt_files_verified=len(receipt['hashes']),csv_sha256=sha(out/'targets.csv'),paired_csv_sha256=sha(out/'PAIRED_TARGETS.csv'),script_sha256=sha(__file__))
    (out/'FACTORIAL_ANALYSIS.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
    print(json.dumps(dict(contrasts=[r for r in contrasts if r['metric']=='coco_iou'],density_interactions=interactions,solver_controls=checks,objective_diagnostics=objective),indent=2))

if __name__=='__main__':main()
