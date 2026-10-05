"""Image-clustered paired summaries of fixed parameter-update diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np


def estimate(rows, values, seed=20260920):
    keep=[(r,v) for r,v in zip(rows,values) if v is not None and np.isfinite(v)]
    if not keep:
        return {'n':0}
    ids=sorted({r['image_id'] for r,_ in keep});lookup={v:i for i,v in enumerate(ids)}
    sums=np.zeros(len(ids));counts=np.zeros(len(ids))
    for r,v in keep:
        i=lookup[r['image_id']];sums[i]+=v;counts[i]+=1
    rng=np.random.default_rng(seed)
    pick=rng.integers(0,len(ids),(2000,len(ids)))
    sample=sums[pick].sum(1)/counts[pick].sum(1)
    return {'n':len(keep),'images':len(ids),'mean':float(sums.sum()/counts.sum()),
            'ci95':np.quantile(sample,[.025,.975]).tolist()}


def matched_contrast(rows,pairs,field):
    lookup={r['annotation_id']:r for r in rows}
    pairs=[(lookup[a],lookup[b]) for a,b in pairs if a in lookup and b in lookup]
    ids=sorted({r['image_id'] for pair in pairs for r in pair});index={v:i for i,v in enumerate(ids)}
    a=np.zeros(len(ids));b=a.copy();na=a.copy();nb=a.copy()
    for ra,rb in pairs:
        ia=index[ra['image_id']];ib=index[rb['image_id']]
        a[ia]+=float(ra['parameter_gradient'][field]);b[ib]+=float(rb['parameter_gradient'][field]);na[ia]+=1;nb[ib]+=1
    rng=np.random.default_rng(20260920);pick=rng.integers(0,len(ids),(2000,len(ids)))
    ca=na[pick].sum(1);cb=nb[pick].sum(1);valid=(ca>0)&(cb>0)
    diffs=(a[pick].sum(1)[valid]/ca[valid]-b[pick].sum(1)[valid]/cb[valid])*100
    return {'pairs':len(pairs),'images':len(ids),'failure_pct':100*a.sum()/na.sum(),
            'success_pct':100*b.sum()/nb.sum(),'difference_pp':100*(a.sum()/na.sum()-b.sum()/nb.sum()),
            'ci95_pp':np.quantile(diffs,[.025,.975]).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert (a.run/'COMPLETE.json').exists()
    rows=[json.loads(s) for s in (a.run/'instances.jsonl').read_text().splitlines()]
    images=[json.loads(s) for s in (a.run/'images.jsonl').read_text().splitlines()]
    primary=[r for r in rows if r['primary']]
    groups={'primary_failure':[r for r in primary if r['geometry_state']=='box_good_mask_unavailable'],
            'primary_success':[r for r in primary if r['geometry_state']=='joint_good'],
            'primary_all':primary,'all_fixed_witnesses':rows}
    result={'source_run':a.run.name,'images':len(images),'scope':'same-image fixed-candidate one-step; no AP or generalization',
            'groups':{},'matched':{},'projection':{'max_negative_constraint':min(r['projection'].get('min_constraint',0) for r in images),
            'solver_unsuccessful':sum(not r['projection']['success'] for r in images),
            'mean_protected_cosine':float(np.mean([r['protected_cosine'] for r in images])),
            'median_step_l2':float(np.median([r['step_l2'] for r in images]))}}
    arms=[k for k in rows[0]['metrics'] if k!='baseline']
    for name,rs in groups.items():
        g={'n':len(rs),'images':len({r['image_id'] for r in rs}),'baseline':{},'arms':{},'contrasts':{}}
        for metric in ['iou','coverage','purity']:
            g['baseline'][metric]=estimate(rs,[100*r['metrics']['baseline'][metric] for r in rs])
        for arm in arms:
            d={}
            for metric in ['iou','coverage','purity','fg_bce','bg_bce','crop_bce']:
                factor=100 if metric in ['iou','coverage','purity'] else 1
                values=[factor*(r['metrics'][arm][metric]-r['metrics']['baseline'][metric])
                        if r['metrics'][arm][metric] is not None and r['metrics']['baseline'][metric] is not None else None for r in rs]
                d[metric+'_change']=estimate(rs,values)
            d['repaired75']=sum(r['metrics']['baseline']['iou']<.75<=r['metrics'][arm]['iou'] for r in rs)
            d['damaged75']=sum(r['metrics'][arm]['iou']<.75<=r['metrics']['baseline']['iou'] for r in rs)
            d['foreground_bce_worsened_pct']=estimate(rs,[100*float(r['metrics'][arm]['fg_bce']>r['metrics']['baseline']['fg_bce']+1e-7)
                  if r['metrics'][arm]['fg_bce'] is not None and r['metrics']['baseline']['fg_bce'] is not None else None for r in rs])
            fg_changes=[r['metrics'][arm]['fg_bce']-r['metrics']['baseline']['fg_bce'] for r in rs
                        if r['metrics'][arm]['fg_bce'] is not None and r['metrics']['baseline']['fg_bce'] is not None]
            d['fg_change_quantiles_posthoc']=dict(zip(['min','q25','median','q75','q90','max'],np.quantile(fg_changes,[0,.25,.5,.75,.9,1]).tolist()))
            g['arms'][arm]=d
        for scale in ['1','3']:
            for control in ['ordinary','angle_control','balanced']:
                contrast={}
                for metric in ['iou','coverage','purity','fg_bce']:
                    factor=100 if metric!='fg_bce' else 1
                    contrast[metric]=estimate(rs,[factor*(r['metrics']['foreground_protected:'+scale][metric]-r['metrics'][control+':'+scale][metric])
                         if r['metrics']['foreground_protected:'+scale][metric] is not None and r['metrics'][control+':'+scale][metric] is not None else None for r in rs])
                g['contrasts']['protected_minus_'+control+':'+scale]=contrast
            g['contrasts']['balanced_minus_ordinary:'+scale]={metric:estimate(rs,
                [100*(r['metrics']['balanced:'+scale][metric]-r['metrics']['ordinary:'+scale][metric]) for r in rs])
                for metric in ['iou','coverage','purity']}
        if name!='all_fixed_witnesses':
            g['parameter_gradient']={field:estimate(rs,[100*float(r['parameter_gradient'][field]) for r in rs])
                   for field in ['own_joint_harms_fg','image_joint_harms_fg','optimizer_harms_fg']}
            g['parameter_gradient']['fg_bg_cosine']=estimate(rs,[r['parameter_gradient']['fg_bg_cosine'] for r in rs])
        result['groups'][name]=g
    pairs=json.loads(a.pairs.read_text())['current_one2one_positive_class_area_box_fill']
    for field in ['own_joint_harms_fg','image_joint_harms_fg','optimizer_harms_fg']:
        result['matched'][field]=matched_contrast(primary,pairs,field)
    (a.out/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'groups':{k:v['n'] for k,v in result['groups'].items()},'matched':result['matched'],'projection':result['projection']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
