"""Report S054 from frozen candidate response measurements."""
import os,sys
from pathlib import Path
if os.name=='nt':
    os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
import ast,json,shutil
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_subtype_neighbor import dump

BASE=Path(__file__).resolve().parent; ROOT=BASE.parent.parent
OUT=BASE/'diagnostics/candidate_ownership_20260913'
REPORT=ROOT/'refine-logs/coco-structure/CANDIDATE_OWNERSHIP_RESULTS_20260913.md'
D=pd.read_csv(OUT/'metrics.csv'); A=D[D.region=='leak'].copy()

def boot(x,seed=20260913):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    b=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),positive=int((x>0).sum()),negative=int((x<0).sum()))
def fmt(x): return f'{x["mean"]*100:+.3f} [{x["ci95"][0]*100:.3f}, {x["ci95"][1]*100:.3f}]'
def fmt_logit(x): return f'{x["mean"]:+.3f} [{x["ci95"][0]:+.3f}, {x["ci95"][1]:+.3f}]'
def link(p,label=None): return f'[{label or p.name}]({p.as_posix()})'

def main():
    A['margin_mean']=A.margin.map(lambda s:ast.literal_eval(s)['mean'])
    summary=dict(margin=boot(A.margin_mean),target_win=boot(A.target_win),delta_iou=boot(A.competition_iou-A.target_mask_iou),delta_neighbor_error=boot(A.competition_neighbor_error-A.target_mask_neighbor_error))
    summary.update(n_positive_iou=int((A.competition_iou>A.target_mask_iou).sum()),n_negative_iou=int((A.competition_iou<A.target_mask_iou).sum()),n_zero_iou=int((A.competition_iou==A.target_mask_iou).sum()))
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    vals=[]; labels=['Target GT','Leak','Neighbor GT','Background']
    for region in ['target_gt','leak','neighbor_gt','background']:
        vals.append(D[D.region==region].margin.map(lambda s:ast.literal_eval(s)['mean']).to_numpy())
    axs[0].boxplot(vals,labels=labels,showfliers=False);axs[0].axhline(0,color='#555',lw=.8);axs[0].set_ylabel('Target logit − neighbor logit');axs[0].tick_params(axis='x',rotation=20);axs[0].grid(axis='y',alpha=.2)
    x=A.competition_iou.to_numpy()-A.target_mask_iou.to_numpy(); order=np.argsort(x)
    axs[1].barh(np.arange(len(x)),x[order]*100,color=np.where(x[order]>=0,'#147d92','#a7442c'));axs[1].axvline(0,color='#555',lw=.8);axs[1].set_yticks(np.arange(len(x)),A.image_id.to_numpy()[order]);axs[1].set_xlabel('IoU change after candidate competition (pp)');axs[1].grid(axis='x',alpha=.2)
    fig.suptitle('Candidate ownership response on 15 fixed same-class neighbor failures',fontsize=12);fig.tight_layout();fig.savefig(OUT/'RESPONSE_COMPETITION.png',dpi=180);fig.savefig(OUT/'RESPONSE_COMPETITION.svg');plt.close(fig)
    lines=['# S054：候选归属响应与两候选竞争诊断','',
    '**这轮把主线从“候选是否存在”推进到“候选之间的像素归属排序是否出错”。** 在固定的15个同类邻居误报目标上，原图中全部15个目标都能匹配到对应的真实邻居候选；目标候选在错误占用的邻居像素上，平均仍有约六成响应更强。用真实邻居候选作诊断性竞争后，多数实例的邻居误报下降、Mask IoU提高。','',
    '真实邻居候选由COCO GT匹配得到，只用于诊断和定义对照，不能在测试时直接获得，也不是提交方法或COCO AP结果。','',
    '## 1. 实验设置','',
    f'2026-09-13。本地RTX 5060 Ti，15个固定同类邻居误报目标，15/15个邻居候选匹配成功。原图一次正常推理；不改输入、不训练、不求解新系数。目标候选和邻居候选使用各自原始NMS候选的系数，与同一个原型矩阵相乘，得到640×640 Mask logit。','',
    'GT只定义目标、邻居、错误占用邻居像素和背景四种诊断区域。主指标是目标logit−邻居logit及目标响应更大的像素比例。两候选竞争保留目标候选原来的正响应支持，只在两者logit比较后去除邻居响应更强的像素；不改变框、类别分数或候选集合。','',
    '## 2. 响应排序结果','',
    '- 邻居候选匹配：**15/15**。',f'- leak区域目标−邻居logit：**{fmt_logit(summary["margin"])} logit**。',f'- leak区域目标候选响应更强比例：**{fmt(summary["target_win"])}%**。','- 目标GT区域目标候选响应更强比例约95%，邻居GT区域约21%。','',
    f'![响应分布与竞争效果]({(OUT/"RESPONSE_COMPETITION.png").as_posix()})','',
    '目标候选在自己的GT区域明显更强，但在其错误占用的邻居区域并没有稳定输给邻居候选。这个结果说明，许多错误像素上两个候选的区分信息已经存在，问题可能出在候选间归属决策，而不是完全没有邻居候选或原型没有表达能力。它还不能说明该排序错误只由拥挤造成。','',
    '## 3. 诊断性竞争结果','',
    '| 量 | 相对原目标候选掩码 |','| --- | --- |',f'| Mask IoU | {fmt(summary["delta_iou"])} |',f'| 邻居误报（自身面积归一化） | {fmt(summary["delta_neighbor_error"])} |',f'| IoU改善实例 | {summary["n_positive_iou"]}/15 |',f'| IoU下降实例 | {summary["n_negative_iou"]}/15 |','',
    '平均IoU变化为+4.76点，12/15例提高，2例下降，1例不变；邻居误报在15例中均不增加。这个结果比局部特征替换更直接地指向实例归属竞争环节，但仍是固定候选、固定原型的诊断上限，不是自动方法收益。','',
    '## 4. 主线含义与下一步','',
    '**目前最值得继续验证的主线是：密集同类实例的框和候选通常已经存在，但候选在相邻像素上的归属竞争没有被充分利用，导致目标掩码吞并邻居。** 系数仍可能是形成相似响应的原因，但本轮没有单独改变系数，不能把责任归给系数头。','',
    '下一步应从这个诊断上限退回可学习、可自动执行的模块：为每个候选生成邻域竞争特征，例如候选间logit差、局部重叠和相对空间关系，训练实例归属/掩码消解头。对照包括原始mask、简单分数排序、普通重叠消解和不使用邻居特征的同容量头。评价应同时报告全COCO AP、按真实mask边界拥挤分组的Mask R75/Recall、共同Precision下Recall、邻居错误和背景错误。方法尚未实现。','',
    '## 5. 限制与完整性','',
    '候选由原图NMS及COCO bbox50匹配固定，逐像素logit不是校准概率；竞争仍保留目标框裁切。真实邻居候选由GT选出，因此不能把竞争后的4.76点写成无GT方法提升。目标和邻居候选还可能共享类别分数及原型，不能据此判定上游特征、系数或候选生成的自然起源。','',
    '所有15例保存目标/邻居logit、margin、两候选掩码、竞争掩码和GT几何；未修改权重、依赖或定时任务。','',
    '- '+link(OUT/'protocol.json','冻结协议'),'- '+link(OUT/'metrics.csv','75条区域记录'),'- '+link(OUT/'COMPLETE.json','推理回执'),'- '+link(OUT/'RESPONSE_COMPETITION.png','诊断图')]
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf8');shutil.copy2(REPORT,OUT/REPORT.name);shutil.copy2(__file__,OUT/Path(__file__).name);dump(OUT/'SUMMARY.json',summary)
    print('REPORT',REPORT)
if __name__=='__main__':main()
