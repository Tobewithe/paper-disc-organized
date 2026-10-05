"""Describe observed pilot/scaled results without selecting models on holdout."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,stable_seed,write_json


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def unpack(arm):
    mode,rest=arm.rsplit('_s',1);seed,draw=rest.split('_d')
    return mode,int(seed),int(draw)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();task=args.run/'task';summaries=read(task/'task_summary.csv');gtrows=read(task/'gt_recovery.csv')
    metrics=['mask_ap','mask_ap50','mask_ap75','r75_all','r75_low_threeway','r75_middle_threeway','r75_high_threeway','nonhigh_minus_high_gap']
    seedgroup=defaultdict(list)
    for r in summaries:
        mode,seed,draw=unpack(r['arm']);seedgroup[(mode,seed)].append(r)
    seedmeans=[]
    for (mode,seed),rr in seedgroup.items():
        record=dict(mode=mode,seed=seed,draws=len(rr))
        record.update({m:float(np.mean([float(r[m]) for r in rr if r[m]!=''])) if any(r[m]!='' for r in rr) else None for m in metrics})
        seedmeans.append(record)
    means=[]
    for mode in dict.fromkeys(r['mode'] for r in seedmeans):
        rr=[r for r in seedmeans if r['mode']==mode]
        values=dict(mode=mode,seeds=len(rr))
        values.update({m:float(np.mean([r[m] for r in rr if r[m] is not None])) if any(r[m] is not None for r in rr) else None for m in metrics})
        means.append(values)
    grouped=defaultdict(list);meta={}
    for r in gtrows:
        mode,seed,draw=unpack(r['arm']);aid=int(r['annotation_id'])
        grouped[(mode,aid)].append(float(r['hit75']=='True'))
        meta[aid]=(int(r['image_id']),float(r['ici']),int(r['category_id']),r['area_bin'])
    vals={key:float(np.mean(v)) for key,v in grouped.items()}
    ids=sorted({v[0] for v in meta.values()});iindex={i:k for k,i in enumerate(ids)}
    aids=sorted(meta);image=np.array([iindex[meta[a][0]] for a in aids]);density=np.array([meta[a][1] for a in aids]);high=density>.5+1e-10
    draws=np.random.default_rng(20260912).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    contrasts=[]
    def statistic(delta,mask):
        counts=np.bincount(image[mask],minlength=len(ids));sums=np.bincount(image[mask],weights=delta[mask],minlength=len(ids))
        den=draws@counts;num=draws@sums;valid=den>0
        result=np.full(2000,np.nan);result[valid]=num[valid]/den[valid]
        return result
    for left,right in [('linear','original'),('own','original'),('mean','original'),('ordered','original'),
                       ('shuffled','original'),('ordered','own'),('ordered','mean'),('ordered','shuffled')]:
        delta=np.array([vals[(left,a)]-vals[(right,a)] for a in aids])
        for label,mask in [('all',np.ones(len(aids),bool)),('low',density<=1e-10),('middle',(density>1e-10)&~high),('high',high),('nonhigh',~high)]:
            if not mask.any():continue
            boot=statistic(delta,mask);lo,hi=np.nanquantile(boot,[.025,.975])
            contrasts.append(dict(comparison=f'{left}-{right}',metric='r75',group=label,n=int(mask.sum()),
                mean_pp=float(delta[mask].mean()*100),ci95_pp=[float(lo*100),float(hi*100)]))
        # Positive high_gain-nonhigh_gain is gap narrowing; require high gain>0
        # and nonhigh not harmed separately, because degradation can shrink a gap.
        boot=statistic(delta,high)-statistic(delta,~high);lo,hi=np.nanquantile(boot,[.025,.975])
        contrasts.append(dict(comparison=f'{left}-{right}',metric='gap_narrowing',group='high-vs-nonhigh',
             mean_pp=float((delta[high].mean()-delta[~high].mean())*100),ci95_pp=[float(lo*100),float(hi*100)]))
    write_json(task/'RESEARCH_ANALYSIS.json',dict(source_hashes={p.name:sha(p) for p in [task/'task_summary.csv',task/'gt_recovery.csv']},
         modes=means,seeds=seedmeans,contrasts=contrasts,
         scope='Exploratory300 heldout train2017 images,3trainingseeds,full COCOeval before density strata. '
         'Pointwise imagebootstrap after seed/draw averaging, no full training-randomness confidence or multiplicity claim. AP point estimates only.'))
    labels={'original':'原模型','linear':'同h线性','own':'同h非线性','mean':'区域平均','ordered':'正确空间排列','shuffled':'打乱空间排列'}
    lines=['# 本地扩大读出对照结果','',
      '状态：COMPLETE。1,200张拟合、300张留出COCO train2017图片，排除先前96张；三种子、15轮，固定最终轮。原模型和原型/框/类别/分数冻结，仅训练读出。',
      '', '## 官方整图任务结果（×100）','',
      '| 方法 | Mask AP | AP75 | 低ICI R75 | 中ICI R75 | 高ICI R75 | 非高−高差距 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in means:
        f=lambda key:'—' if r[key] is None else f'{r[key]*100:.3f}'
        lines.append('| '+labels.get(r['mode'],r['mode'])+' | '+' | '.join(f(k) for k in ['mask_ap','mask_ap75','r75_low_threeway','r75_middle_threeway','r75_high_threeway','nonhigh_minus_high_gap'])+' |')
    lines.extend(['','低组ICI≤1e−10，中组(1e−10,.5+1e−10]，高组>.5+1e−10；ICI是同类GT框交叠面积和/自身框面积，不是mask遮挡率。先对完整GT/预测做官方匹配，再按实例划分。','',
      '## 高组和差距的配对区间','', '| 比较 | 统计 | 差值百分点 | 95%图像区间 |','|---|---|---:|---|'])
    for r in contrasts:
        if r['group'] not in ['high','high-vs-nonhigh']:continue
        lines.append(f'| {r["comparison"]} | {r["metric"]} | {r["mean_pp"]:+.3f} | [{r["ci95_pp"][0]:+.3f},{r["ci95_pp"][1]:+.3f}] |')
    lines.extend(['','差距缩小必须同时检查高组提高且非高组没有退化，不能仅凭gap_narrowing正值宣布成功。',
       '', '## 范围与实现','',
       '- 全部普通GT保留；未框匹配目标也进入任务分母。固定目标IoU是另一口径，见evaluation/SUMMARY.json。',
       '- 图像未用于本次小头拟合，但属于官方预训练见过的train2017；不是COCO val2017标准成绩，也不是全新论文确认集。',
       '- PureBCE监督与先前32图探针相同，没有只给空间臂增加新损失；未证明超过历史BCE+Dice强对照。',
       '- 区间为2000次图像簇重采样，条件于当前数据和3种子，点态探索性；AP没有计算区间。',
       '- 本地RTX5060Ti，torch2.9.1+cu128，隔离加载远端ultralytics8.4.143和filelock3.18；所有臂共享本地缓存，不与远端数值直接合并。',
       '- 两个已知限制：3×3采样槽位的部分信息也可由内容推断；同参数量不保证同有效函数复杂度。阴性只限制该实现和预算。',
       '',f'原始结果目录：{args.run.as_posix()}。本报告不自动宣告新颖性或根因。'])
    args.out.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(means),flush=True)


if __name__=='__main__':main()
