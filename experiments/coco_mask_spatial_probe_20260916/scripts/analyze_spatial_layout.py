"""Paired field interventions and outcome-conditioned object descriptions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def estimate(rows,values):
    valid=[(r,float(v)) for r,v in zip(rows,values) if v is not None and np.isfinite(v)]
    if not valid:return {'n':0}
    ids=sorted({r['image_id'] for r,v in valid});idx={v:i for i,v in enumerate(ids)}
    sums=np.zeros(len(ids));counts=sums.copy()
    for r,v in valid:i=idx[r['image_id']];sums[i]+=v;counts[i]+=1
    pick=np.random.default_rng(20260920).integers(0,len(ids),(2000,len(ids)))
    samples=sums[pick].sum(1)/counts[pick].sum(1)
    return {'n':len(valid),'images':len(ids),'mean':float(sums.sum()/counts.sum()),'ci95':np.quantile(samples,[.025,.975]).tolist()}


def match(left,right):
    if not left or not right:return []
    cost=np.full((len(left),len(right)),1e6)
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            if a['category_id']!=b['category_id']:continue
            da=abs(np.log(max(a['area'],1)/max(b['area'],1)))
            di=abs(a['metrics']['baseline']['iou']-b['metrics']['baseline']['iou'])
            db=abs(a['box_iou']-b['box_iou']);df=abs(a['fill']-b['fill'])
            if da<=np.log(2) and di<=.08 and db<=.1 and df<=.15:cost[i,j]=da/np.log(2)+di/.08+db/.1+df/.15
    ii,jj=linear_sum_assignment(cost)
    return [(left[i],right[j]) for i,j in zip(ii,jj) if cost[i,j]<1e5]


def paired_groups(pairs,extract):
    pairs=[(a,b) for a,b in pairs if extract(a) is not None and extract(b) is not None]
    if not pairs:return {'pairs':0}
    ids=sorted({r['image_id'] for p in pairs for r in p});idx={x:i for i,x in enumerate(ids)}
    sa=np.zeros(len(ids));sb=sa.copy();na=sa.copy();nb=sa.copy()
    for a,b in pairs:
        i=idx[a['image_id']];j=idx[b['image_id']];sa[i]+=extract(a);sb[j]+=extract(b);na[i]+=1;nb[j]+=1
    pick=np.random.default_rng(20260920).integers(0,len(ids),(2000,len(ids)));n1=na[pick].sum(1);n2=nb[pick].sum(1);v=(n1>0)&(n2>0)
    delta=sa[pick].sum(1)[v]/n1[v]-sb[pick].sum(1)[v]/n2[v]
    return {'pairs':len(pairs),'left':float(sa.sum()/na.sum()),'right':float(sb.sum()/nb.sum()),
            'difference':float(sa.sum()/na.sum()-sb.sum()/nb.sum()),'ci95':np.quantile(delta,[.025,.975]).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert (a.run/'COMPLETE.json').exists()
    rows=[json.loads(s) for s in (a.run/'instances.jsonl').read_text().splitlines()]
    names=['baseline','scalar','local','field_mean','field_rms_scalar','field_rotated']
    if 'shared_template' in rows[0]['metrics']:names.append('shared_template')
    controls=[name for name in names if name not in ['baseline','local']]
    groups={k:[r for r in rows if r['group']==k] for k in sorted({r['group'] for r in rows})}
    groups['any_repair']=[r for r in rows if 'repair' in r['group']];groups['any_damage']=[r for r in rows if 'damage' in r['group']]
    result={'scope':'fixed baseline GT-best identity; groups defined by historical COCO matches; no newAP','groups':{},'matched':{}}
    for group,rs in groups.items():
        g={'n':len(rs),'images':len({r['image_id'] for r in rs}),'arms':{},'contrasts':{},'layout':{}}
        for name in names:
            arm={'mask75':sum(r['metrics'][name]['iou']>=.75 for r in rs)}
            for field in ['iou','coverage','purity']:
                arm[field]=estimate(rs,[100*r['metrics'][name][field] for r in rs])
                arm[field+'_change']=estimate(rs,[100*(r['metrics'][name][field]-r['metrics']['baseline'][field]) for r in rs])
            for field in ['fn_fixed','tp_lost','fp_fixed','tn_lost']:
                arm[field+'_area_pp']=estimate(rs,[100*r['metrics'][name][field]/max(r['metrics']['baseline']['tp']+r['metrics']['baseline']['fn'],1) for r in rs])
            for field in ['fp_near','fp_far','fn_near','fn_interior','fp_same_neighbor','fp_other_neighbor','fp_background']:
                arm[field+'_change_area_pp']=estimate(rs,[100*(r['metrics'][name][field]-r['metrics']['baseline'][field])/
                    max(r['metrics']['baseline']['tp']+r['metrics']['baseline']['fn'],1) for r in rs])
            arm['simultaneous_net_tp_gain_fp_reduction_pct']=estimate(rs,[100*float(r['metrics'][name]['tp']>r['metrics']['baseline']['tp'] and
                                    r['metrics'][name]['fp']<r['metrics']['baseline']['fp']) for r in rs])
            g['arms'][name]=arm
        for control in controls:
            g['contrasts']['local_minus_'+control]={field:estimate(rs,[100*(r['metrics']['local'][field]-r['metrics'][control][field]) for r in rs]) for field in ['iou','coverage','purity']}
            g['contrasts']['local_minus_'+control]['mask75_pp']=estimate(rs,[100*(float(r['metrics']['local']['iou']>=.75)-float(r['metrics'][control]['iou']>=.75)) for r in rs])
        for field in ['fp_fraction_largest_cell','fn_fraction_largest_cell','fn_fp_spatial_tv','field_mean','field_std','field_rms','field_positive_fraction']:
            g['layout'][field]=estimate(rs,[r['layout'][field] for r in rs])
        g['layout']['mixed_field_pct']=estimate(rs,[100*float(.05<r['layout']['field_positive_fraction']<.95) for r in rs])
        result['groups'][group]=g
    pairs_saved={}
    for other in ['scalar_only_repair','both_repair']:
        pairs=match(groups['local_only_repair'],groups[other]);pairs_saved[other]=[[a['annotation_id'],b['annotation_id']] for a,b in pairs]
        desc={field:paired_groups(pairs,lambda r,k=field:r['layout'][k]) for field in
             ['fp_fraction_largest_cell','fn_fraction_largest_cell','fn_fp_spatial_tv','field_mean','field_std']}
        desc['rotated_loss_pp']=paired_groups(pairs,lambda r:100*(r['metrics']['local']['iou']-r['metrics']['field_rotated']['iou']))
        desc['mixed_field_pct']=paired_groups(pairs,lambda r:100*float(.05<r['layout']['field_positive_fraction']<.95))
        result['matched']['local_only_vs_'+other]=desc
    (a.out/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False));(a.out/'MATCHED_PAIRS.json').write_text(json.dumps(pairs_saved))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    labels=['Learned scalar','Mean field','Uniform RMS','Rotated field']
    if 'shared_template' in controls:labels.append('Shared training shape')
    for ax,group,title in zip(axs,['local_only_repair','any_repair'],['Local-only COCO repairs (582)','All COCO repairs (1472)']):
        vals=[result['groups'][group]['contrasts']['local_minus_'+k]['iou'] for k in controls];means=np.array([v['mean'] for v in vals]);ci=np.array([v['ci95'] for v in vals])
        ax.errorbar(means,np.arange(len(controls)),xerr=np.maximum(np.vstack([means-ci[:,0],ci[:,1]-means]),0),fmt='o',capsize=4,color='#237b91')
        ax.set_yticks(np.arange(len(controls)),labels);ax.invert_yaxis();ax.axvline(0,color='gray',lw=1);ax.grid(axis='x',alpha=.2)
        ax.set_title(title);ax.set_xlabel('Local minus control: mask IoU percentage points')
    fig.suptitle('Fixed original prediction identity; conditional diagnostic, not AP')
    fig.savefig(a.out/'spatial_layout_controls.png',dpi=180);fig.savefig(a.out/'spatial_layout_controls.pdf');plt.close(fig)
    print(json.dumps({'groups':{k:v['n'] for k,v in result['groups'].items()},'matched_pairs':{k:len(v) for k,v in pairs_saved.items()}}))


if __name__=='__main__':main()
