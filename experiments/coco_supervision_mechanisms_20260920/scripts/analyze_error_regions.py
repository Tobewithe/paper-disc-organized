"""Paired, image-clustered analysis; all intervals are exploratory."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_parameter_probe import estimate, matched_contrast


def main():
    ap=argparse.ArgumentParser()
    for k in ['run','out']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--pairs',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert (a.run/'COMPLETE.json').exists()
    rows=[json.loads(s) for s in (a.run/'instances.jsonl').read_text().splitlines()]
    primary=[r for r in rows if r['primary']]
    groups={'primary_failure':[r for r in primary if r['geometry_state']=='box_good_mask_unavailable'],
            'primary_success':[r for r in primary if r['geometry_state']=='joint_good'],
            'primary_all':primary,'all_fixed_witnesses':rows}
    arms=sorted({k for r in rows for k in r['metrics'] if k!='baseline'})
    result={'source_run':a.run.name,'groups':{},'scope':'fixed-candidate same-image mechanism diagnosis, not AP',
            'ci':'2000 image-cluster bootstrap, instance-weighted means, exploratory no multiplicity correction'}
    for group,rs in groups.items():
        g={'n':len(rs),'images':len({r['image_id'] for r in rs}),'baseline':{},'arms':{},'contrasts':{}}
        for field in ['tp','fp','fn','fp_near_boundary','fp_far_boundary','fn_near_boundary','fn_interior',
                      'fp_same_neighbor','fp_other_neighbor','fp_background']:
            g['baseline'][field+'_total']=sum(r['metrics']['baseline'][field] for r in rs)
        for field in ['iou','coverage','purity']:
            g['baseline'][field]=estimate(rs,[100*r['metrics']['baseline'][field] for r in rs])
        for field in ['fp_near_boundary','fp_far_boundary','fn_near_boundary','fn_interior','fp_same_neighbor','fp_other_neighbor','fp_background']:
            g['baseline'][field+'_per_gt_area_pp']=estimate(rs,[100*r['metrics']['baseline'][field]/
                max(r['metrics']['baseline']['tp']+r['metrics']['baseline']['fn'],1) for r in rs])
        for arm in arms:
            if group=='all_fixed_witnesses' and arm.startswith('coefficient/'):continue
            valid=[r for r in rs if arm in r['metrics']]
            if not valid:continue
            d={'n':len(valid)}
            for field in ['iou','coverage','purity','crop_bce']:
                multiplier=1 if field=='crop_bce' else 100
                d[field+'_change']=estimate(valid,[multiplier*(r['metrics'][arm][field]-r['metrics']['baseline'][field]) for r in valid])
            for field in ['fn_fixed','tp_lost','fp_fixed','tn_lost']:
                d[field+'_total']=sum(r['metrics'][arm][field] for r in valid)
                d[field+'_per_gt_area_pp']=estimate(valid,[100*r['metrics'][arm][field]/max(r['metrics']['baseline']['tp']+r['metrics']['baseline']['fn'],1) for r in valid])
            for field in ['fp_near_boundary','fp_far_boundary','fn_near_boundary','fn_interior',
                          'fp_same_neighbor','fp_other_neighbor','fp_background']:
                d[field+'_change_per_gt_area_pp']=estimate(valid,[100*(r['metrics'][arm][field]-r['metrics']['baseline'][field])/
                       max(r['metrics']['baseline']['tp']+r['metrics']['baseline']['fn'],1) for r in valid])
            for region in ['tp','fn','fp','tn']:
                v=[r for r in valid if r['metrics']['baseline']['region_bce'][region] is not None]
                d[region+'_bce_change']=estimate(v,[r['metrics'][arm]['region_bce'][region]-r['metrics']['baseline']['region_bce'][region] for r in v])
            d['repaired75']=sum(r['metrics']['baseline']['iou']<.75<=r['metrics'][arm]['iou'] for r in valid)
            d['damaged75']=sum(r['metrics'][arm]['iou']<.75<=r['metrics']['baseline']['iou'] for r in valid)
            g['arms'][arm]=d
        for domain,scales in [('head',['1','3']),('coefficient',['0.1','0.3'])]:
            if group=='all_fixed_witnesses' and domain=='coefficient':continue
            for scale in scales:
                comparisons=[(method,'ordinary') for method in ['errors_only','correct_only','random_equal_pixels','balanced','focal2','soft_iou']]
                comparisons += [('errors_only',control) for control in ['random_equal_pixels','balanced','focal2','soft_iou']]
                for method,control in comparisons:
                    ma=f'{domain}/{method}:{scale}';ca=f'{domain}/{control}:{scale}'
                    valid=[r for r in rs if ma in r['metrics'] and ca in r['metrics']]
                    if not valid:continue
                    g['contrasts'][f'{domain}/{method}_minus_{control}:{scale}']={field:estimate(valid,
                       [100*(r['metrics'][ma][field]-r['metrics'][ca][field]) for r in valid]) for field in ['iou','coverage','purity']}
        if group!='all_fixed_witnesses':
            comparable=[r for r in rs if r['competition']['counts']['fn']+r['competition']['counts']['fp']>0]
            g['competition']={'zero_training_errors_excluded':len(rs)-len(comparable)}
            for field in ['own_correct_error_cosine','own_correct_error_norm_ratio','own_correct_gradient_opposes_errors','ordinary_optimizer_harms_error_bce']:
                factor=100 if field in ['own_correct_gradient_opposes_errors','ordinary_optimizer_harms_error_bce'] else 1
                g['competition'][field]=estimate(comparable,[factor*float(r['competition'][field]) for r in comparable])
            for domain in ['image_region_dot_own_errors','coefficient_region_dot_own_errors']:
                g['competition'][domain+'_opposes_pct']={region:estimate(comparable,[100*float(r['competition'][domain][region]<0) for r in comparable])
                                                       for region in ['tp','fn','fp','tn']}
        result['groups'][group]=g
    if a.pairs:
        pairs=json.loads(a.pairs.read_text())['current_one2one_positive_class_area_box_fill']
        comparable=[dict(r,parameter_gradient=r['competition']) for r in primary
                    if r['competition']['counts']['fn']+r['competition']['counts']['fp']>0]
        result['matched']={field:matched_contrast(comparable,pairs,field) for field in
            ['own_correct_gradient_opposes_errors','ordinary_optimizer_harms_error_bce']}
    (a.out/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    methods=['correct_only','random_equal_pixels','balanced','focal2','soft_iou','errors_only']
    labels=['Correct pixels only','Random equal count','FG/BG balanced','Focal (gamma=2)','Soft IoU','Error pixels only']
    for ax,domain,scale,title in zip(axs,['coefficient','head'],['0.3','1'],['Independent coefficient update','Actual shared head update']):
        for offset,group,color in [(-.14,'primary_failure','#D96735'),(.14,'primary_success','#3476AF')]:
            vals=[result['groups'][group]['contrasts'][f'{domain}/{m}_minus_ordinary:{scale}']['iou'] for m in methods]
            means=np.array([v['mean'] for v in vals]);ci=np.array([v['ci95'] for v in vals])
            ax.errorbar(means,np.arange(len(methods))+offset,xerr=np.maximum(np.vstack([means-ci[:,0],ci[:,1]-means]),0),
                        fmt='o',color=color,capsize=3,label='Failure (up to 273)' if group=='primary_failure' else 'Success (415)')
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(np.arange(len(methods)),labels);ax.invert_yaxis()
        ax.set_xlabel('Mask IoU change vs ordinary BCE (percentage points)');ax.set_title(title);ax.grid(axis='x',alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axs[0].legend(loc='best',frameon=False)
    fig.suptitle('Same-image diagnostic; 95% image-cluster intervals; not COCO AP')
    fig.savefig(a.out/'error_region_comparison.png',dpi=180);fig.savefig(a.out/'error_region_comparison.pdf');plt.close(fig)
    print(json.dumps({'groups':{k:v['n'] for k,v in result['groups'].items()},'source':a.run.name}))


if __name__=='__main__':main()
