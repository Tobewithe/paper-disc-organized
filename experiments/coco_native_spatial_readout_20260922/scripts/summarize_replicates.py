"""Three-seed and boundary evidence; image-cluster intervals for fixed-model GT recall."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')


def main(args):
    root=args.root;ids=read(root/'replication_ids.json');pilot=root/'runs/RUN_1bb359da7b4942b7a7ab3f0be194ff49'
    initial=Path('/root/coco_spatial_calibration_mechanism_20260922/runs/RUN_984bb9ff1cd2432bae2c2c08c3042c66')
    prior=read('/root/coco_template_source_20260921/runs/RUN_9de7e2ec9f5643d3874bbd4422262def/ANALYSIS.json')['summary']
    baseline_metrics=prior['baseline']['metrics_0_100'];results0=read(pilot/'RESULTS.json');match0=read(pilot/'MATCHED_GT75.json')
    methods=['native_scalar','native_spatial','native_finetune']
    results={k:[] for k in methods};matched={k:[] for k in methods};baseline=set(match0['baseline'])
    for name in methods:
        results[name].append(read(initial/'RESULTS.json') if name=='native_finetune' else results0[name])
        matched[name].append(set(read(initial/'MATCHED_GT75.json')['native_finetune']) if name=='native_finetune' else set(match0[name]))
    sources={'seed0_new':str(pilot),'seed0_native':str(initial)}
    for seed in [1,2]:
        folder=root/'runs'/ids[f'evaluate_s{seed}'];assert read(folder/'run.json')['status']=='completed'
        res=read(folder/'RESULTS.json');mt=read(folder/'MATCHED_GT75.json');assert set(mt['baseline'])==baseline
        sources[f'seed{seed}']=str(folder)
        for name in methods:
            key='native_finetune_selected' if name=='native_finetune' else name
            results[name].append(res[key]);matched[name].append(set(mt[key]))
    table={}
    for name in methods:
        table[name]={'seeds':results[name], 'metrics_0_100':{k:{'mean':float(np.mean([r['metrics'][k]*100 for r in results[name]])),
                     'sd':float(np.std([r['metrics'][k]*100 for r in results[name]],ddof=1))} for k in baseline_metrics},
                     'alpha':[r['alpha'] for r in results[name]]}
        for key in ['repaired75','damaged75','matched75']:
            v=[r[key] for r in results[name]];table[name][key]={'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1))}
        table[name]['net75']={'mean':float(np.mean([r['repaired75']-r['damaged75'] for r in results[name]]))}
    anns=read(args.annotations);rows=[a for a in anns['annotations'] if not a.get('iscrowd',0) and not a.get('ignore',0)]
    assert len(rows)==36335,len(rows);gt_ids={a['id'] for a in rows};assert baseline<=gt_ids
    image_ids=sorted(v['id'] for v in anns['images']);index={iid:i for i,iid in enumerate(image_ids)}
    image_of=np.array([index[a['image_id']] for a in rows]);gids=np.array([a['id'] for a in rows]);area=np.array([a['area'] for a in rows])
    box_area=np.array([a['bbox'][2]*a['bbox'][3] for a in rows]);fill=area/np.maximum(box_area,1e-12)
    groups={'all':np.ones(len(rows),dtype=bool),'small':area<32**2,'medium':(area>=32**2)&(area<96**2),'large':area>=96**2,
            'fill_le025':fill<=.25,'fill_025_050':(fill>.25)&(fill<=.5),'fill_050_075':(fill>.5)&(fill<=.75),'fill_gt075':fill>.75}
    success={'baseline':np.isin(gids,list(baseline)).astype(float)}
    for name in methods:success[name]=np.mean([np.isin(gids,list(s)) for s in matched[name]],axis=0)
    rng=np.random.default_rng(20260922);draws=rng.integers(0,len(image_ids),size=(2000,len(image_ids)),dtype=np.int32)
    comparisons=[('native_spatial','baseline'),('native_spatial','native_scalar'),('native_spatial','native_finetune')]
    recall={}
    for group,select in groups.items():
        denominator=np.bincount(image_of[select],minlength=len(image_ids));bootstrap_den=denominator[draws].sum(1)
        entry={'gt':int(select.sum()),'micro_R75':{k:float(v[select].mean()*100) for k,v in success.items()},'paired_differences_pp':{}}
        for a,b in comparisons:
            d=success[a]-success[b];numerator=np.bincount(image_of[select],weights=d[select],minlength=len(image_ids))
            boot=numerator[draws].sum(1)/bootstrap_den*100
            entry['paired_differences_pp'][a+'-minus-'+b]={'estimate':float(d[select].mean()*100),'ci95':np.quantile(boot,[.025,.975]).tolist()}
        recall[group]=entry
    categories=sorted({a['category_id'] for a in rows});cat=np.array([a['category_id'] for a in rows])
    macro={k:float(np.mean([v[cat==c].mean() for c in categories])*100) for k,v in success.items()}
    boundary_id=read(root/'boundary_retry_ids.json')['boundary_evaluation'];boundary=read(root/'runs'/boundary_id/'RESULTS.json')
    assert read(root/'runs'/boundary_id/'run.json')['status']=='completed';assert len(boundary)==6
    bmetrics={k:{m:v*100 for m,v in r['boundary_metrics'].items()} for k,r in boundary.items()}
    summary={'sources':sources,'baseline_metrics_0_100':baseline_metrics,'three_seed':table,'boundary_seed0_0_100':bmetrics,
       'matched_GT_recall':recall,'macro_category_R75_point_estimate':macro,
       'statistics_scope':'AP mean/sampleSD over3trainingseeds, noAPsignificanceclaim. RecallCI resamples5000images2000times, conditioning on these3trainedmodels averagedperGT; it doesnot accountforunobservedtrainingseeds. Subgroups exploratory, no multiplicityadjustment. Recall is GTmicro atIoU.75 usingofficialmatchedsets, not AP or theCOCOclassaveragedAR statistic.',
       'fill_definition':'ProvidedCOCOannotationarea /providedbboxarea, not a visible-occlusion groundtruth. Ratiosnotclipped; invalidbboxarea count recorded.',
       'invalid_bbox_area':int((box_area<=0).sum()),'fill_above_one':int((fill>1).sum())}
    write(args.out/'RESULTS.json',summary)
    lines=['# 三种子与边界评价','', 'AP为三种子均值±样本标准差；Boundary AP为seed0。','',
        '|方法|Mask AP|AP75|AP small|平均修复/损伤/净修复75|','|---|---:|---:|---:|---:|']
    for name in methods:
        row=table[name];m=row['metrics_0_100'];cell=lambda k:f"{m[k]['mean']:.4f} ± {m[k]['sd']:.4f}"
        lines.append(f"|{name}|{cell('AP')}|{cell('AP75')}|{cell('APS')}|{row['repaired75']['mean']:.1f}/{row['damaged75']['mean']:.1f}/{row['net75']['mean']:.1f}|")
    lines+=['','## Boundary AP（固定预测，seed0）','','|方法|Boundary AP|相对基线|','|---|---:|---:|']
    for name,m in bmetrics.items():lines.append(f"|{name}|{m['AP']:.4f}|{m['AP']-bmetrics['baseline']['AP']:+.4f}|")
    lines+=['','## GT微平均R75（3个固定模型的平均）','','|子组|GT数|baseline|空间|空间−标量及95%图像簇区间|空间−原生微调及区间|','|---|---:|---:|---:|---:|---:|']
    for name,r in recall.items():
        def contrast(key):
            d=r['paired_differences_pp'][key];return f"{d['estimate']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]"
        lines.append(f"|{name}|{r['gt']}|{r['micro_R75']['baseline']:.3f}|{r['micro_R75']['native_spatial']:.3f}|{contrast('native_spatial-minus-native_scalar')}|{contrast('native_spatial-minus-native_finetune')}|")
    lines+=['','上述区间是在现有三个模型固定的条件下重采样图像，不是AP置信区间，也不覆盖所有未来训练随机性。子组是探索性分析，未作多重比较校正。填充率不能等同遮挡率。','']
    (args.out/'TABLES.md').write_text('\n'.join(lines),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.3),layout='constrained');labels=['Scalar','Spatial','Native fine-tune'];colors=['#7f8c8d','#2166ac','#d68027']
    gain=[table[n]['metrics_0_100']['AP']['mean']-baseline_metrics['AP'] for n in methods];sd=[table[n]['metrics_0_100']['AP']['sd'] for n in methods]
    axes[0].bar(labels,gain,color=colors,yerr=sd,capsize=4);axes[0].set(ylabel='Mask AP gain (points)',title='A. Mean ± SD over three seeds');axes[0].tick_params(axis='x',rotation=18)
    x=np.arange(3);repair=[table[n]['repaired75']['mean'] for n in methods];harm=[table[n]['damaged75']['mean'] for n in methods]
    axes[1].bar(x-.18,repair,.36,color='#2a9d8f',label='Repaired');axes[1].bar(x+.18,harm,.36,color='#c65a49',label='Lost')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='GT matches at Mask75',title='B. Gains and losses (seed mean)');axes[1].tick_params(axis='x',rotation=18);axes[1].legend(frameon=False)
    bm=['ROI_scalar','ROI_spatial_residual','native_coefficient_finetune','native_spatial'];bl=['ROI scalar','ROI spatial','Native fine-tune','Native spatial']
    axes[2].barh(bl,[bmetrics[n]['AP']-bmetrics['baseline']['AP'] for n in bm],color=['#7f8c8d','#6baed6','#d68027','#2166ac'])
    axes[2].set(xlabel='Boundary AP gain (points)',title='C. Published boundary metric (seed 0)');axes[2].invert_yaxis()
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for suffix in ['png','pdf','svg']:fig.savefig(args.out/f'REPLICATION_AND_BOUNDARY.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig);write(args.out/'COMPLETE.json',{'seeds':[0,1,2],'images':5000,'boundary_models':6,'GT':len(rows)})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','out','annotations']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);main(a)
