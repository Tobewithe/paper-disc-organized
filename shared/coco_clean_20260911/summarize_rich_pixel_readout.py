"""All arms, seed means and paired image intervals; no post-hoc selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def unpack(s):
    mode,rest=s.rsplit('_s',1);seed,draw=rest.split('_d');return mode,int(seed),int(draw)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    run=a.run;complete=json.loads((run/'COMPLETE.json').read_text())
    if complete['status']!='COMPLETE':raise RuntimeError('Incomplete run')
    task=read(run/'task_summary.csv');gt=read(run/'gt_recovery.csv');spatial=read(run/'spatial.csv')
    cfg=json.loads((run/'protocol.json').read_text());cache=Path(cfg['cache'])
    ids=json.loads((cache/'selection.json').read_text())['transfer'];index={iid:i for i,iid in enumerate(ids)}
    modes=['original']+cfg['modes'];fields=['mask_ap','mask_ap50','mask_ap75','r75_all','r75_high','r75_low','pair75_high','gap']
    seed_rows=[]
    for mode in modes:
        seeds=sorted({unpack(r['arm'])[1] for r in task if unpack(r['arm'])[0]==mode})
        for seed in seeds:
            rr=[r for r in task if unpack(r['arm'])[:2]==(mode,seed)]
            if len(rr)!=(3 if mode=='rich_shuffled' else 1):raise RuntimeError('Wrong number of draws')
            seed_rows.append(dict(mode=mode,seed=seed,**{f:float(np.mean([float(r[f]) for r in rr])) for f in fields}))
    means=[]
    for mode in modes:
        rr=[r for r in seed_rows if r['mode']==mode]
        means.append(dict(mode=mode,seeds=len(rr),**{f:float(np.mean([r[f] for r in rr])) for f in fields},
            mask_ap_seed_sd=float(np.std([r['mask_ap'] for r in rr],ddof=1)) if len(rr)>1 else 0.))
    # Enforce original whole-task parity with the already published local source table.
    base_old=read(cache.parent/'task/task_summary.csv')
    old=next(r for r in base_old if r['arm']=='original_s-1_d0');new=next(r for r in task if r['arm']=='original_s-1_d0')
    for f in ['mask_ap','mask_ap50','mask_ap75','r75_all','r75_high','r75_low']:
        if abs(float(old[f])-float(new[f]))>1e-10:raise RuntimeError('Original task replay changed: '+f)
    draws=np.random.default_rng(20260912).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    contrasts=[];groupmeans=[]
    pairs=[('global','original'),('scalar','original'),('rich','original'),('rich_neighbor','original'),
           ('rich','global'),('rich','scalar'),('rich_neighbor','global'),('rich_neighbor','scalar'),
           ('rich_neighbor','rich'),('rich_neighbor','rich_shuffled')]
    for domain,rows,metrics in [('task',gt,['hit75']),('spatial',spatial,['iou','coverage','neighbor','background'])]:
        buckets=defaultdict(list);meta={};sets=defaultdict(set)
        for r in rows:
            mode,seed,draw=unpack(r['arm']);aid=int(r['annotation_id']);sets[r['arm']].add(aid)
            buckets[mode,aid].append([float(r[f]=='True') if f=='hit75' else float(r[f]) if r[f] else np.nan for f in metrics])
            meta[aid]=(int(r['image_id']),float(r['ici']))
        expected=next(iter(sets.values()))
        if any(s!=expected for s in sets.values()):raise RuntimeError(domain+' denominator changed')
        aids=sorted(meta);image=np.array([index[meta[i][0]] for i in aids]);dens=np.array([meta[i][1] for i in aids])
        groups={'all':np.ones(len(aids),bool),'low':dens<=1e-10,'middle':(dens>1e-10)&(dens<=.5+1e-10),'high':dens>.5+1e-10,'nonhigh':dens<=.5+1e-10}
        arrays={m:np.array([np.mean(buckets[m,i],axis=0) for i in aids]) for m in modes}
        def boot(values,mask):
            mask=mask&np.isfinite(values)
            den=np.bincount(image[mask],minlength=len(ids)).astype(float)
            num=np.bincount(image[mask],weights=values[mask],minlength=len(ids))
            ds=np.einsum('bi,i->b',draws,den,optimize=False);ns=np.einsum('bi,i->b',draws,num,optimize=False)
            return np.divide(ns,ds,out=np.full_like(ns,np.nan),where=ds>0),int(mask.sum())
        for mode in modes:
            for group,mask in groups.items():
                groupmeans.append(dict(domain=domain,mode=mode,group=group,n=int(mask.sum()),
                    **{m:float(np.nanmean(arrays[mode][mask,k])) for k,m in enumerate(metrics)}))
        for left,right in pairs:
            delta=arrays[left]-arrays[right]
            for k,metric in enumerate(metrics):
                for group,mask in groups.items():
                    b,n=boot(delta[:,k],mask)
                    contrasts.append(dict(domain=domain,comparison=f'{left}-{right}',metric=metric,group=group,n=n,
                        delta_pp=float(np.nanmean(delta[mask,k])*100),ci95_pp=(np.nanquantile(b,[.025,.975])*100).tolist()))
            if domain=='task':
                high,_=boot(delta[:,0],groups['high']);other,_=boot(delta[:,0],groups['nonhigh'])
                contrasts.append(dict(domain=domain,comparison=f'{left}-{right}',metric='gap_narrowing',group='high-vs-nonhigh',
                    delta_pp=float((delta[groups['high'],0].mean()-delta[groups['nonhigh'],0].mean())*100),
                    ci95_pp=(np.nanquantile(high-other,[.025,.975])*100).tolist()))
    result=dict(means=means,seed_means=seed_rows,group_means=groupmeans,contrasts=contrasts,
       original_task_parity='PASS',source_hashes={p.name:sha(p) for p in [run/'task_summary.csv',run/'gt_recovery.csv',run/'spatial.csv']},
       scope='300 already-explored heldouttrain2017 images,3fittedreadoutseeds. Pointwise exploratory2000imagecluster intervals, '
             'seed/draw metrics averaged first; no AP uncertainty, multiplicity or full-training random-effect claim. '
             'AllGT task matching beforeICI, spatial fixedbbox attribution differs. No matched-Precision result.')
    write_json(run/'RESEARCH_ANALYSIS.json',result)
    labels={'original':'原模型','global':'全局系数强对照','scalar':'标量像素对照','rich':'P(x)+h像素读出','rich_neighbor':'P(x)+h+预测邻居','rich_shuffled':'局部P(x)打乱'}
    lines=['# S026：完整局部原型与候选特征的逐像素读出','',
       '状态：COMPLETE。复用1,200拟合+300留出COCO train2017图，7,811拟合目标；三种子、15轮、225个checkpoint。原模型/原型/框/类别/分数冻结；推理不读GT。',
       '', '## 官方任务成绩（乘100）','',
       '| 方法 | Mask AP | 高ICI R75 | 非高ICI R75 | 非高−高差距 |','|---|---:|---:|---:|---:|']
    for r in means:lines.append('| '+labels[r['mode']]+' | '+' | '.join(f'{r[k]*100:.3f}' for k in ['mask_ap','r75_high','r75_low','gap'])+' |')
    lines+=['','## 主比较的高组与差距区间','', '| 比较 | 指标 | 差值百分点 | 95%图像区间 |','|---|---|---:|---|']
    for r in contrasts:
        if r['domain']!='task' or r['group'] not in ['high','high-vs-nonhigh']:continue
        lines.append(f'| {r["comparison"]} | {r["metric"]} | {r["delta_pp"]:+.3f} | [{r["ci95_pp"][0]:+.3f},{r["ci95_pp"][1]:+.3f}] |')
    lines+=['','## 高组固定归属像素指标（乘100）','', '| 方法 | IoU | 自身覆盖 | 同类邻居错误/自身GT | 背景错误/自身GT |','|---|---:|---:|---:|---:|']
    for r in groupmeans:
        if r['domain']=='spatial' and r['group']=='high':lines.append('| '+labels[r['mode']]+' | '+' | '.join(f'{r[k]*100:.3f}' for k in ['iou','coverage','neighbor','background'])+' |')
    lines+=['','## 范围与限制','',
       '- 300图已经用于前轮探索，且属于官方预训练见过的train2017；不是COCO val最终成绩或未接触确认集。',
       '- 训练使用既有原生640 GT框支持中均匀采样的前512点，所有组共同BCE+Dice、样本和更新预算。固定bbox50训练归属不等于复现历史TAL训练。',
       '- 所有像素臂同64宽架构；scalar将P/h输入置零、rich将邻居槽置零。参数数目相同而有效函数容量不同，不能声称严格等容量信息论辨识。全局头采用更宽128隐层作为实用强对照。',
       '- rich_shuffled训练逐目标逐轮打乱P向量与位置，仍保留原像素logit及标量邻居输入；评估每种子3次固定布局。只测试P空间对齐，不单独识别真实邻居的因果价值，也不构成严格的实例错配控制。',
       '- 原型输入先按官方双线性扩展到640，再做非线性像素读出；最后仍裁切原预测框。零残差使用原解码顺序。原始官方任务成绩与S024精确一致。',
       '- 高R75应提高且非高/整体AP不退化；差距通过降低非高组缩小不算成功。仍需匹配Precision、组成控制及未调参验证才能升级方法结论。',
       '- 区间为三种子/打乱指标平均后的2000次图像簇点态区间，无AP区间、多重比较校正或完整端到端随机性推断。',
       '',f'原始结果：{run.as_posix()}，protocol.json、history.json、各组checkpoints、task_summary.csv、gt_recovery.csv、spatial.csv、RESEARCH_ANALYSIS.json。']
    a.report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(means),flush=True)


if __name__=='__main__':main()
