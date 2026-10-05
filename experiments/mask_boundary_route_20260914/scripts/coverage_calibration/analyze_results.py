"""Prespecified paired uncertainty, component analysis and hypothesis decision."""
from pathlib import Path
import argparse
import csv
import json
from collections import Counter
from common import setup, atomic, progress


def main():
    parser=argparse.ArgumentParser()
    for arg in ['protocol','output','evaluation','fit','selection','inference']:
        parser.add_argument('--'+arg,required=True)
    args=parser.parse_args();p,out=setup(args.protocol,args.output)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evaluation=Path(args.evaluation)
    summary=json.loads((evaluation/'SUMMARY.json').read_text())
    rows=list(csv.DictReader((evaluation/'instance_decisions.csv').open(newline='',encoding='utf-8')))
    assert summary['outcomes']['frozen_rcmc']['target_repairs']==457
    assert summary['outcomes']['frozen_rcmc']['all_repairs']==801
    assert summary['outcomes']['frozen_rcmc']['damages']==445
    ids=sorted(json.loads((Path(p['val_bank'])/'image_ids.json').read_text())); pos={iid:i for i,iid in enumerate(ids)}
    annotations=json.loads(Path(p['val_annotations']).read_text())['annotations']
    total=Counter(a['image_id'] for a in annotations if not a.get('iscrowd',0))
    # Counts per complete image. Zero-target/zero-slot images are retained during resampling.
    denominators=np.zeros((len(ids),3))
    denominators[:,2]=[total[i] for i in ids]
    names=p['variants'];numbers=np.zeros((len(ids),len(names),4))
    grouping={};corrected_coverage={'coverage_high':{'success':0,'damage':0},'coverage_low':{'success':0,'damage':0}}
    for r in rows:
        i=pos[int(r['image_id'])];base=float(r['baseline_iou']);target=r['target']=='True'
        success=base>=.75;coverage=float(r['baseline_coverage']);area=float(r['gt_area'])
        denominators[i,0]+=int(target);denominators[i,1]+=int(success)
        size='small' if area<32**2 else 'medium' if area<96**2 else 'large'
        cov='coverage_high' if coverage>=.95 else 'coverage_low'
        if success:
            corrected_coverage[cov]['success']+=1
            use=r['use_frozen_rcmc']=='1'
            after=float(r['trial_iou']) if use else base
            corrected_coverage[cov]['damage']+=int(after<.75)
        for j,name in enumerate(names):
            use=r['use_'+name]=='1';after=float(r['trial_iou']) if use else base
            repair=base<.75<=after;damage=after<.75<=base
            numbers[i,j]=numbers[i,j]+[int(target and repair),int(damage),int(repair)-int(damage),after-base]
            for group in ['all_matched',size,cov,'target' if target else 'other']:
                st=grouping.setdefault(group,{}).setdefault(name,dict(n=0,targets=0,successes=0,repairs=0,target_repairs=0,damages=0,iou_delta_sum=0.,coverage_loss_sum=0.))
                st['n']+=1;st['targets']+=int(target);st['successes']+=int(success)
                st['repairs']+=int(repair);st['target_repairs']+=int(target and repair);st['damages']+=int(damage)
                st['iou_delta_sum']+=after-base;st['coverage_loss_sum']+=float(r['coverage_cost']) if use else 0.
    rng=np.random.default_rng(20260915);boot=np.empty((2000,len(names),3))
    for k in range(2000):
        sample=rng.integers(0,len(ids),len(ids));den=denominators[sample].sum(axis=0)
        boot[k]=100*numbers[sample,:,:3].sum(axis=0)/den
    estimates=100*numbers[:,:,:3].sum(axis=0)/denominators.sum(axis=0)
    comparisons={}
    for method in ['direct','decomposed','decomposed_protected']:
        for reference in ['frozen_rcmc','direct','decomposed']:
            if method==reference:continue
            a=names.index(method);b=names.index(reference);samples=boot[:,a]-boot[:,b]
            means=estimates[a]-estimates[b]
            comparisons[method+'_vs_'+reference]={key:{'difference_pp':float(means[k]),
                'paired_image_bootstrap_ci95_pp':np.quantile(samples[:,k],[.025,.975]).tolist()}
                for k,key in enumerate(['target_repair_rate','all_success_damage_rate','net_r75_all_gt'])}
    verdicts={}
    for method in ['decomposed','decomposed_protected']:
        checks={}
        for ref in ['direct','frozen_rcmc']:
            m=summary['outcomes'][method];r=summary['outcomes'][ref];delta_ap=summary['metrics'][method]['ap']-summary['metrics'][ref]['ap']
            pareto=(m['target_repairs']>=r['target_repairs'] and m['damages']<=r['damages'] and
                    (m['target_repairs']>r['target_repairs'] or m['damages']<r['damages']))
            ci=comparisons[method+'_vs_'+ref]
            resolved=(ci['target_repair_rate']['paired_image_bootstrap_ci95_pp'][0]>0 or
                      ci['all_success_damage_rate']['paired_image_bootstrap_ci95_pp'][1]<0)
            checks[ref]={'target_repair_count_delta':m['target_repairs']-r['target_repairs'],
                         'all_success_damage_count_delta':m['damages']-r['damages'],
                         'delta_ap_points':delta_ap,'pareto_point_condition':pareto,
                         'ap_nondecrease':delta_ap>=-1e-10,'better_dimension_resolved':resolved}
        point=all(v['pareto_point_condition'] and v['ap_nondecrease'] for v in checks.values())
        verdict=('supported_in_this_setting' if all(v['better_dimension_resolved'] for v in checks.values()) else 'insufficient_evidence') if point else 'not_supported_in_this_setting'
        verdicts[method]={'verdict':verdict,'checks':checks}
    result={'verdicts':verdicts,'comparisons':comparisons,'groups':grouping,
            'corrected_coverage_only_success_groups':corrected_coverage,'denominators':dict(zip(['target_failures','original_successes','ordinary_gt'],denominators.sum(axis=0).astype(int).tolist())),
            'metrics':summary['metrics'],'outcomes':summary['outcomes'],'policies':summary['policies'],
            'fit':json.loads((Path(args.fit)/'SUMMARY.json').read_text()),
            'inference':json.loads((Path(args.inference)/'SUMMARY.json').read_text()),
            'bootstrap':{'n':2000,'seed':20260915,'unit':'complete images, paired across all variants; fixed original groups','scope':'pointwise exploratory intervals on repair/damage/net R75, not AP confidence intervals'},
            'termination':'Single-seed fixed comparison is complete; no validation-driven retuning, more seeds, YOLO training or new architecture in this goal.',
            'limitations':p['scope_limits']}
    # Both policies were independently selected on train data. Hold the cutoff
    # equal here to isolate the protection switch, without selecting a new rule.
    fixed_cutoff_diagnostics={}
    for source in ['decomposed','decomposed_protected']:
        cutoff=summary['policies'][source]['cutoff']
        for protected in [False,True]:
            stats={'cutoff':cutoff,'protection':protected,'target_repairs':0,'all_repairs':0,
                   'damages':0,'iou_gain_sum':0.,'coverage_loss_sum':0.}
            for r in rows:
                use=float(r['predicted_gain'])>cutoff and (not protected or float(r['predicted_coverage_cost'])<=.02)
                base=float(r['baseline_iou']);after=float(r['trial_iou']) if use else base
                repair=base<.75<=after;damage=after<.75<=base
                stats['target_repairs']+=int(r['target']=='True' and repair)
                stats['all_repairs']+=int(repair);stats['damages']+=int(damage)
                stats['iou_gain_sum']+=after-base
                stats['coverage_loss_sum']+=float(r['coverage_cost']) if use else 0.
            fixed_cutoff_diagnostics[source+('_protected' if protected else '_unprotected')]=stats
    result['same_cutoff_protection_diagnostic']=fixed_cutoff_diagnostics
    result['same_cutoff_diagnostic_scope']='Fixed-slot descriptive switch only at already train-selected cutoffs; no new AP selection or deployed variant.'
    atomic(out/'SUMMARY.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    labels=['Baseline','RCMC','Direct','Decomposed','+ coverage\nprotection']
    colors=['#a6abb0','#477ea2','#e1b25a','#c56b51','#679477']
    for ax,key,ylabel in [(axes[0],'target_repairs','Target failures repaired'),(axes[1],'damages','Original successes damaged')]:
        v=[summary['outcomes'][name][key] for name in names]
        bars=ax.bar(range(len(names)),v,color=colors);ax.bar_label(bars,padding=3);ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(names)),labels,fontsize=8);ax.margins(y=.18)
    values=[summary['metrics'][name]['ap'] for name in names]
    bars=axes[2].bar(range(len(names)),values,color=colors)
    axes[2].bar_label(bars,labels=[f'{x:.3f}' for x in values],padding=3)
    axes[2].set_ylim(min(values)-.15,max(values)+.2);axes[2].set_ylabel('COCO Mask AP (0-100)')
    axes[2].set_xticks(range(len(names)),labels,fontsize=8)
    fig.suptitle('Frozen policies: the same 4,500 COCO images and one fixed smooth action')
    fig.savefig(out/'method_comparison.png',dpi=180);fig.savefig(out/'method_comparison.pdf');plt.close(fig)
    progress(out,'completed',verdicts=verdicts)


if __name__=='__main__':main()
