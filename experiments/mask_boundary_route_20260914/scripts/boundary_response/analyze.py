"""Paired uncertainty and a finite decision after the frozen full evaluation."""
import argparse
import csv
import json
from pathlib import Path
from common import setup,atomic,progress


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();p,out=setup(args.protocol,args.output)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(p['evaluation']['bank']).parent;source=root/p['evaluation']['run_id']
    summary=json.loads((source/'SUMMARY.json').read_text())
    refs=json.loads((Path(p['evaluation']['reference_evaluation'])/'SUMMARY.json').read_text())
    # Refit reproduces all leaf values/structure; a few fractional-area split
    # thresholds differ at 3e-17. Verify no val input lies between those values
    # before treating historical direct outputs as this cell's exact comparator.
    direct=json.loads((root/'RUN_6b816e23401b401eaebdb9a59604de19/direct.json').read_text())
    shape=json.loads((root/p['evaluation']['model_run']/'single_shape.json').read_text())
    assert direct['baseline']==shape['baseline']
    for key in direct['nodes']:
        if key!='threshold':assert direct['nodes'][key]==shape['nodes'][key]
    gaps=[]
    for i,(a,b) in enumerate(zip(direct['nodes']['threshold'],shape['nodes']['threshold'])):
        for j,(lo,hi) in enumerate(zip(a,b)):
            if lo!=hi:
                assert direct['nodes']['feature'][i][j]==4
                gaps.append((min(lo,hi),max(lo,hi)))
    values=np.asarray([float(r['removed_fraction']) for r in csv.DictReader((Path(p['evaluation']['bank'])/'candidate_records.csv').open(newline='',encoding='utf-8'))])
    ambiguous=sum(int(((values>lo)&(values<=hi)).sum()) for lo,hi in gaps)
    assert ambiguous==0,'Historical comparator must be rescored; floating split changed a decision'
    comparator_parity=dict(equal_structure_and_leaf_values=True,threshold_rounding_gaps=len(gaps),inputs_in_gaps=ambiguous)
    old=list(csv.DictReader((Path(p['evaluation']['reference_evaluation'])/'instance_decisions.csv').open(newline='',encoding='utf-8')))
    lookup={(int(r['image_id']),int(r['annotation_id'])):r for r in old}
    rows=list(csv.DictReader((source/'instance_decisions.csv').open(newline='',encoding='utf-8')))
    ids=sorted(json.loads((Path(p['evaluation']['bank'])/'image_ids.json').read_text()));pos={i:j for j,i in enumerate(ids)}
    names=['frozen_rcmc','direct','single_local','multi_local'];nums=np.zeros((len(ids),len(names),3));den=np.zeros((len(ids),2))
    groups={}
    for r in rows:
        key=(int(r['image_id']),int(r['annotation_id']));b=lookup[key];i=pos[key[0]]
        base=float(r['baseline_iou']);coverage=float(r['baseline_coverage']);target=r['target']=='True'
        assert abs(base-float(b['baseline_iou']))<1e-12
        success=base>=.75;den[i]+=[target,success]
        after={n:float(b['trial_iou']) if b['use_'+n]=='1' else base for n in names[:2]}
        after.update({n:float(r['iou_'+n]) for n in names[2:]})
        for j,n in enumerate(names):
            repair=base<.75<=after[n];damage=after[n]<.75<=base
            nums[i,j]+=[target and repair,damage,int(repair)-int(damage)]
            if success:
                g='success_high_coverage' if coverage>=.95 else 'success_low_coverage'
                s=groups.setdefault(g,{}).setdefault(n,{'n':0,'damages':0});s['n']+=1;s['damages']+=int(damage)
    assert len(rows)==30426 and den.sum(0).tolist()==[1701,19065]
    rng=np.random.default_rng(20260921);boot=np.empty((2000,len(names),3))
    for k in range(2000):
        ix=rng.integers(0,len(ids),len(ids));ds=den[ix].sum(0)
        boot[k]=nums[ix].sum(0);boot[k,:,:2]=100*boot[k,:,:2]/ds
    point=nums.sum(0);point[:,:2]=100*point[:,:2]/den.sum(0)
    comparisons={}
    for method in names[2:]:
        for ref in names[:2]:
            a=names.index(method);b=names.index(ref)
            comparisons[method+'_vs_'+ref]={metric:dict(difference=float(point[a,k]-point[b,k]),
                ci95=np.quantile(boot[:,a,k]-boot[:,b,k],[.025,.975]).tolist())
                for k,metric in enumerate(['target_repair_pp','success_damage_pp','net_repair_count'])}
    result=dict(evaluation=summary,reference_outcomes=refs['outcomes'],reference_metrics=refs['metrics'],
        paired_image_comparisons=comparisons,coverage_groups=groups,comparator_parity=comparator_parity,bootstrap={'repeats':2000,'seed':20260921,'images':len(ids),
        'scope':'Complete paired images including zero-target images; fixed baseline groups; not AP intervals.'})
    atomic(out/'SUMMARY.json',result)
    allout={n:refs['outcomes'][n] for n in names[:2]}|summary['outcomes']
    allmetrics={n:refs['metrics'][n] for n in names[:2]}|summary['metrics']
    lines=['# 冻结边界响应策略：完整验证结果','',
        '本次结果来自历史已探索的 4500 张 COCO val2017 图片，不是新盲测。模型、阈值在训练选择集上固定后未修改。','',
        '| 方法 | 目标修复 /1701 | 全部修复 | 原成功误伤 /19065 | Mask AP | AP75 |',
        '|---|---:|---:|---:|---:|---:|']
    for n in names:
        s=allout[n];m=allmetrics[n]
        lines.append(f"| {n} | {s['target_repairs']} | {s['all_repairs']} | {s['damages']} | {m['ap']:.4f} | {m['ap75']:.4f} |")
    lines+=['',f"执行判断：**{summary['decision']}**。",'',
        '通过意味着值得进入独立确认；未通过意味着本轮局部响应统计／动作组合未满足预设权衡要求，停止调参，不推论所有局部信息无用。',
        '完整逐图配对区间、覆盖分组与比较条件保存在 SUMMARY.json。GT 上限不能当作可部署收益。',
        f"源评价 Run：{source.name}。全量原预测 RLE 核对：{summary['full_baseline_rle_parity']} 个。"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    labels=['RCMC','Direct','Local / single','Local / multi'];colors=['#888888','#5287ac','#64a87c','#b78552']
    for ax,key,title in [(axes[0],'target_repairs','Target failures repaired'),(axes[1],'damages','Original successes damaged')]:
        bars=ax.bar(labels,[allout[n][key] for n in names],color=colors);ax.bar_label(bars,padding=3);ax.set_title(title);ax.tick_params(axis='x',labelrotation=20);ax.margins(y=.15)
    vals=[allmetrics[n]['ap'] for n in names];bars=axes[2].bar(labels,vals,color=colors);axes[2].bar_label(bars,fmt='%.3f',padding=3)
    axes[2].set_ylim(min(vals)-.15,max(vals)+.15);axes[2].set_title('COCO Mask AP');axes[2].tick_params(axis='x',labelrotation=20)
    fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf');plt.close(fig)
    progress(out,'completed',decision=summary['decision'],verdict=summary['verdict'])


if __name__=='__main__':main()
