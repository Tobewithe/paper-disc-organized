"""Reproduce the consolidated mechanism tables and figure from immutable runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(path.read_text())


def main(args):
    root=args.root;out=args.out;out.mkdir(parents=True,exist_ok=True)
    phase1=root/'runs/RUN_89e14901533a4493a9617a16414e68e6'
    phase2=root/'runs/RUN_c1b150dc71c54c3e9abcfe07d6da9303'
    reference=read(args.template_summary)
    support=read(root/'support_ids.json');native=read(root/'native_ids.json');within=read(root/'within_instance_ids.json')
    table={k:v for k,v in reference['summary'].items()}
    for name,rid in [('gt_support',support['gt_support_eval']),('matched_random',support['matched_random_eval']),('native_readout',native['evaluate'])]:
        res=read(root/'runs'/rid/'RESULTS.json');item=res['shared2'] if name!='native_readout' else res
        table[name]={'metrics_0_100':{k:100*v for k,v in item['metrics'].items()},'matched75':item['matched75'],
            'repairs75':item['repaired75'],'harms75':item['damaged75'],'net75':item['repaired75']-item['damaged75'],
            'alpha':item['alpha'],'source_run':rid}
    contrasts={}
    for a,b in [('fit_residual','baseline'),('fit_residual','scalar'),('gt_support','analytic_center'),('matched_random','analytic_center'),
                ('gt_support','matched_random'),('native_readout','baseline'),('fit_residual','native_readout')]:
        contrasts[a+'_minus_'+b]={k:table[a]['metrics_0_100'][k]-table[b]['metrics_0_100'][k] for k in table[a]['metrics_0_100']}
    phase1_summary=read(phase1/'SUMMARY.json');phase2_summary=read(phase2/'SUMMARY.json')
    calibration=read(phase2/'CALIBRATION.json');rows=[r for r in calibration['images'] if r['split']=='confirm']
    bins=np.asarray([r['hard_inside_gt_bins'] for r in rows],dtype=float);totals=bins.sum(0)
    rng=np.random.default_rng(20260926);weights=rng.multinomial(len(rows),np.ones(len(rows))/len(rows),size=2000)
    boot=np.einsum('bi,irfc->brfc',weights,bins)
    reliability=[]
    for bi in range(10):
        c,r=totals[:,bi];prob=c[2]/c[0]-r[2]/r[0];gap=c[1]/c[0]-r[1]/r[0]
        draw=boot[:,:,bi];g=draw[:,0,1]/draw[:,0,0]-draw[:,1,1]/draw[:,1,0]
        reliability.append({'center_count':int(c[0]),'ring_count':int(r[0]),'center_foreground':float(c[1]/c[0]),
            'ring_foreground':float(r[1]/r[0]),'center_probability':float(c[2]/c[0]),'ring_probability':float(r[2]/r[0]),
            'foreground_gap_pp':float(100*gap),'model_probability_gap_pp':float(100*prob),'image_ci95_pp':(100*np.quantile(g,[.025,.975])).tolist()})
    instances=[json.loads(x) for x in (phase2/'instances.jsonl').read_text().splitlines()]
    confirm=[r for r in instances if r['split']=='confirm']
    projection={'retained_energy_median':float(np.median([r['projection']['retained_energy'] for r in confirm])),
        'a_positive_fraction':float(np.mean([r['a']>0 for r in confirm])),
        'b_mean':float(np.mean([r['b'] for r in confirm])),'a_mean':float(np.mean([r['a'] for r in confirm]))}
    groups={}
    # Descriptive, after-the-fact failure subsets; never used to train or select a checkpoint.
    for label,subset in [('all',confirm),('initial_mask75_fail',[r for r in confirm if r['metrics']['base_byte']['mask75']==0]),
        ('initial_mask75_pass',[r for r in confirm if r['metrics']['base_byte']['mask75']==1]),
        ('box75_mask75_fail',[r for r in confirm if r['box_iou']>=.75 and r['metrics']['base_byte']['mask75']==0])]:
        groups[label]={'instances':len(subset),'iou_gain_pp':float(100*np.mean([r['metrics']['shared_byte']['iou']-r['metrics']['base_byte']['iou'] for r in subset])),
            'repairs75':sum(r['metrics']['shared_byte']['mask75']>r['metrics']['base_byte']['mask75'] for r in subset),
            'harms75':sum(r['metrics']['shared_byte']['mask75']<r['metrics']['base_byte']['mask75'] for r in subset)}
    within_results=read(root/'runs'/within['evaluate']/'RESULTS.json')
    result={'scope':'Completed runs; single-seed training controls. AP point estimates, not AP confidence intervals.',
        'full_val5000':table,'ap_contrasts':contrasts,'phase1':phase1_summary,'phase2':phase2_summary,
        'hard_gt_inside_box_reliability_confirm':reliability,'within_instance':within_results,
        'projection_confirm':projection,'descriptive_failure_subsets_confirm':groups,
        'sources':{'template':str(args.template_summary),'phase1':str(phase1),'phase2':str(phase2),
            'support':support,'native':native,'within':within}}
    (out/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    names={'baseline':'官方权重','scalar':'仅标量修正','local16':'原16格修正头','learned_template':'旧学习模板',
        'analytic_center':'解析中心模板，完整区域监督','shuffled_learned':'打乱模板','fit_residual':'训练残差模板',
        'gt_support':'解析模板，仅GT框内监督','matched_random':'解析模板，等量同标签比例随机删监督','native_readout':'仅原生系数末层微调'}
    lines=['# 机制实验汇总','', '所有 AP 按0–100显示。诊断IoU点数与标准COCO AP不是同一指标。新训练对照为seed0；不能把单次小差值说成统计显著。','',
        '|方法|AP|AP75|AP小目标|修复/损伤/净修复@75|','|---|---:|---:|---:|---:|']
    for key,item in table.items():
        m=item['metrics_0_100'];lines.append(f"|{names[key]}|{m['AP']:.4f}|{m['AP75']:.4f}|{m['APS']:.4f}|{item['repairs75']}/{item['harms75']}/{item['net75']}|")
    lines+=['','## 同分数、不同位置','',f"第二组确认面板{len(confirm)}个固定候选；GT框内原始硬标签。z∈[0,0.5]时中心真实前景率{100*reliability[5]['center_foreground']:.3f}%，外围{100*reliability[5]['ring_foreground']:.3f}%；各自平均sigmoid分数{100*reliability[5]['center_probability']:.3f}%和{100*reliability[5]['ring_probability']:.3f}%。",'',
        f"同一实例内配对（至少3个中心、外围ROI样本）结果：{json.dumps(within_results['confirm']['bins'][5],ensure_ascii=False)}",'',
        '## 原型投影与受控空间干预','','|干预差值，第二组确认|IoU提升/百分点|图像簇bootstrap95%区间|','|---|---:|---|']
    pc=phase2_summary['confirm']['all']['contrasts']
    for name in ['space_given_bias_half','projection_retains_space_half','outside_span_space_half','full_minus_projected_half','space_inside_given_bias_half','space_outside_given_bias_half']:
        if name not in pc:continue
        x=pc[name]['iou'];lines.append(f"|{name}|{x['delta_pp']:.4f}|{x['image_ci95_pp']}|")
    lines+=['',f"空间项投影能量保留的实例中位数：{100*projection['retained_energy_median']:.3f}%。该投影只针对aT，b保持未投影；不是证明全部修正都能由原系数表达。",'',
        '## 训练对照的差值','', '```json',json.dumps(contrasts,ensure_ascii=False,indent=2),'```','',
        '## 适用范围','',
        '- 两个诊断面板来自COCO train2017、排除修正头fit/dev图像；官方COCO预训练模型此前见过train2017。其独立性仅相对于修正头学习，不称为原模型未见数据。',
        '- 机制诊断仅覆盖有同类BoxIoU≥0.5候选的代表实例，不诊断全体漏检；固定候选诊断不等同官方匹配。',
        '- GT框内/外干预是机制工具；正常模板和原系数末层方法在val推理不使用GT。',
        '- 监督区域训练消融固定alpha=0.5；原系数末层在原200张train selection上选alpha。原生末层6240参数，模板头94498参数，不是参数量匹配。',
        '- 支持范围对照使用新增修正头BCE+0.5Dice，未重放官方TAL、实例面积归一化或端到端预训练；不能据此断言官方训练发生具体错误。',
        '- 同实例位置差异仍可能反映形状先验和特征损失；它提供有效条件信息，不自动说明唯一历史根因。','']
    (out/'TABLES.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    a=axes[0];bs=[3,4,5,6];labels=['[-1,-.5]','[-.5,0]','[0,.5]','[.5,1]'];xx=np.arange(4)
    for offset,key,label,color in [(-.19,'center_foreground','Center GT','#2274a5'),(.19,'ring_foreground','Periphery GT','#de8f05')]:
        a.bar(xx+offset,[100*reliability[b][key] for b in bs],.36,label=label,color=color)
    a.plot(xx,[100*(reliability[b]['center_probability']+reliability[b]['ring_probability'])/2 for b in bs],'ko--',label='Mean model probability',markersize=4)
    a.set(xticks=xx,xticklabels=labels,xlabel='Original mask logit bin',ylabel='Foreground rate / probability (%)',title='A. Same score, different reliability')
    a.legend(fontsize=8);a.set_ylim(0,100)
    a=axes[1];keys=['space_given_bias_half','projection_retains_space_half','outside_span_space_half']
    # Use stored contrasts rather than recomputing or assigning significance from the figure.
    available=[k for k in keys if k in pc]
    values=np.array([pc[k]['iou']['delta_pp'] for k in available]);ci=np.array([pc[k]['iou']['image_ci95_pp'] for k in available]).T
    a.bar(np.arange(len(values)),values,color=['#2274a5','#3ca370','#a9b5c0'][:len(values)])
    a.errorbar(np.arange(len(values)),values,yerr=np.stack((values-ci[0],ci[1]-values)),fmt='none',color='black',capsize=4)
    a.set(xticks=np.arange(len(values)),xticklabels=['Full spatial','Prototype span','Complement'][:len(values)],ylabel='Mean instance IoU gain (pp)',title='B. Spatial gain after adding the same bias')
    a=axes[2];keys=['baseline','scalar','fit_residual','native_readout'];values=[table[k]['metrics_0_100']['AP'] for k in keys]
    a.barh(np.arange(4),values,color=['#a9b5c0','#91bed4','#2274a5','#3ca370'])
    a.set(yticks=np.arange(4),yticklabels=['Original','Scalar','Spatial residual','Native readout'],xlabel='COCO Mask AP',title='C. Full val2017 (5,000 images)',xlim=(min(values)-.5,max(values)+.5));a.invert_yaxis()
    for i,v in enumerate(values):a.text(v+.025,i,f'{v:.3f}',va='center',fontsize=9)
    fig.savefig(out/'MECHANISM.png',dpi=180);plt.close(fig)
    (out/'COMPLETE.json').write_text(json.dumps({'status':'completed','tables':len(table),'source_run_ids':[support,native,within]}))
    print(json.dumps({'ap_contrasts':contrasts,'within_instance_confirm_near_threshold':within_results['confirm']['bins'][5]},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','out','template-summary']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
