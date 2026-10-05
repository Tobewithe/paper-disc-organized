"""Render S053 completed results, including uncertain contrasts and counterexample."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
import json,shutil
import numpy as np,cv2
from pycocotools import mask as mu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_subtype_neighbor import dump,sha

BASE=Path(__file__).resolve().parent;ROOT=BASE.parent.parent;OUT=BASE/'diagnostics/feature_cell_partition_20260912';SRC=BASE/'diagnostics/subtype_neighbor_20260912'
A=json.loads((OUT/'ANALYSIS.json').read_text())
def get(fill,region,direction='insert',metric='mask_iou'):
    return next(r['metrics'][metric] for r in A['absolute'] if r['fill']==fill and r['region']==region and r['direction']==direction and r['mode']==('background_control' if region=='equal_background' else 'neighbor'))
def contrast(fill,name,direction='insert',metric='mask_iou'):
    return next(r['metrics'][metric] for r in A['comparisons'] if r['fill']==fill and r['contrast']==name and r['direction']==direction)
def fmt(x):return f'{x["mean"]*100:+.3f} [{x["ci95"][0]*100:.3f}, {x["ci95"][1]*100:.3f}]'
def link(p,label=None):return f'[{label or p.name}]({p.as_posix()})'
def dec(x):return mu.decode(x).astype(bool)

def figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
    for ax,direction in zip(axs,['insert','restore']):
        for offset,fill,color in [(-.1,'texture','#147d92'),(.1,'local_color','#a7442c')]:
            stats=[get(fill,r,direction) for r in ['equal_mixed','equal_exclusive','equal_background']];y=np.array([s['mean'] for s in stats])*100;l=np.array([s['ci95'][0] for s in stats])*100;h=np.array([s['ci95'][1] for s in stats])*100
            ax.errorbar(np.arange(3)+offset,y,yerr=[y-l,h-y],fmt='o',capsize=4,label=fill.replace('_',' '),color=color)
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(3),['Mixed cells','Target-disjoint\nneighbor cells','Background cells']);ax.set_title('Insertion minus original' if direction=='insert' else 'Restoration minus edited');ax.grid(axis='y',alpha=.18)
    axs[0].set_ylabel('Mask IoU change (percentage points)');axs[1].legend(frameon=False,loc='lower right')
    fig.suptitle('P3 cell partition: equal cell count within each of 15 fixed targets',fontsize=12)
    fig.text(.5,.01,'Three geometry subsets averaged per image; one fill seed per type. Pointwise 95% bootstrap CIs; exploratory.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(OUT/'CELL_EFFECTS.png',dpi=180);fig.savefig(OUT/'CELL_EFFECTS.svg');plt.close(fig)
    records=[];fig,axs=plt.subplots(4,4,figsize=(11,11.3))
    for caseidx,iid in enumerate([572408,561465]):
        pack=json.loads((OUT/'pairs'/str(iid)/'masks.json').read_text());own=np.load(SRC/'pairs'/str(iid)/'geometry.npz')['own'].astype(bool)
        im=cv2.cvtColor(cv2.imread(str(BASE/'local_readout_runtime_20260912/data/images/val2017'/f'{iid:012}.jpg')),cv2.COLOR_BGR2RGB);ys,xs=np.nonzero(own);pad=max(15,int(max(np.ptp(xs),np.ptp(ys))*.35));y0=max(0,ys.min()-pad);y1=min(im.shape[0],ys.max()+pad+1);x0=max(0,xs.min()-pad);x1=min(im.shape[1],xs.max()+pad+1)
        for rowoff,direction in enumerate(['insert','restore']):
            row=caseidx*2+rowoff;masks=[pack['original'] if direction=='insert' else next(v['mask'] for v in pack['variants'] if v['mode']=='neighbor' and v['fill']=='texture' and v['region']=='endpoint')]
            for r in ['mixed_all','exclusive_all','all_edit']:masks.append(next(v['mask'] for v in pack['variants'] if v['mode']=='neighbor' and v['fill']=='texture' and v['region']==r and v['direction']==direction))
            titles=['Original' if direction=='insert' else 'Edited prototype','Mixed cells only','Target-disjoint cells only','Union of both'];nums=[]
            for col,(encoded,title) in enumerate(zip(masks,titles)):
                m=dec(encoded);iou=(m&own).sum()/max((m|own).sum(),1);nums.append(float(iou));v=im.astype(float)/255
                for mask,color in [(m&own,[.15,.85,.3]),(m&~own,[1,.17,.2]),(own&~m,[1,.85,.03])]:v[mask]=v[mask]*.45+np.array(color)*.55
                axs[row,col].imshow(v[y0:y1,x0:x1]);axs[row,col].set_title(f'{title}\nIoU {iou*100:.2f}%',fontsize=9);axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
            axs[row,0].set_ylabel(f'COCO {iid}\n{direction}',fontsize=10);records.append(dict(image_id=iid,direction=direction,reference=nums[0],mixed=nums[1],exclusive=nums[2],union=nums[3]))
    fig.suptitle('Fixed positive and adverse examples: local effects need not add',fontsize=12)
    fig.text(.5,.012,'Green: true positive; yellow: missed target; red: false positive. Original images used only as display backgrounds.\nAll cells in each region, texture seed 0, fixed c0 and b0. This figure does not represent the equal-count group means.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.97]);fig.savefig(OUT/'CASE_PANEL.png',dpi=170);fig.savefig(OUT/'CASE_PANEL.svg');plt.close(fig);dump(OUT/'ILLUSTRATIVE_CASES.json',records);return records

def main():
    cases=figures();lines=['# S053：P3混合格与邻居独占格拆分','',
    '**邻居独占格也承载了可撤回的影响；当前不能将S052的改善全部归为直接改到了目标格子。** 但等数量混合格与独占格的直接效果差区间均跨零，没有证据判其中一类更重要。独占格插入相对原图的完整IoU区间仍跨零，超过背景对照与单独稳定修复是不同结论。','',
    f'2026-09-12。全部15个同类邻居误报目标、{A["rows"]}条记录、{A["seconds"]:.3f}秒本地推理。复用S052队列，全部可做等数量对照；没有排除、替补、训练或GT系数求解。两种填充仍仅各seed0。','',
    '## 1. 本轮改变及主比较','',
    '固定原框b0、原系数c0和Proto26的P3输入层。以保存的80×80网格定义M=目标投影∩邻居编辑位置，E=邻居编辑位置减目标投影；M/E互斥，并集精确等于S052操作区域。背景B取原背景编辑格减目标格。先比较全部M、全部E及并集，再按每图k=min(|M|,|E|,|B|)比较相同格数。','',
    '每图k为1—4个，混合格原有1—5个、独占格2—11个。三种区域分别无放回选k格，几何seed0/1/2在推理前固定；两填充、两方向共享这些位置。先对三个子集取图片内均值，再按15张图配对bootstrap2000次。三个子集不是三个独立训练或填充重复，全选时位置也会相同。','',
    '插入从原图特征出发，仅带入对应位置编辑图特征，差值相对原图；恢复从编辑图特征出发，放回对应位置原图特征，差值相对该编辑图c0P1端点。负的恢复差表示撤回编辑端点的收益，不能当作原图直接提高了相同数值。','',
    '## 2. 相同格子数量：实际IoU变化','',
    '以下差值为百分点，区间为探索性点态95%CI；多比较未校正。','',
    '| 操作区域 | 纹理插入 | 纹理恢复 | 颜色插入 | 颜色恢复 |','| --- | --- | --- | --- | --- |']
    for region,label in [('equal_mixed','混合格M'),('equal_exclusive','邻居独占格E'),('equal_background','背景格B')]:lines.append('| '+label+' | '+' | '.join(fmt(get(f,region,d)) for f in ['texture','local_color'] for d in ['insert','restore'])+' |')
    lines+=['',f'![等数量格子对照]({(OUT/"CELL_EFFECTS.png").as_posix()})','',
    '| 图片内配对差 | 纹理插入 | 纹理恢复 | 颜色插入 | 颜色恢复 |','| --- | --- | --- | --- | --- |']
    for name,label in [('equal_mixed_minus_exclusive','M−E（哪种区域更强）'),('equal_mixed_minus_background','M−B'),('equal_exclusive_minus_background','E−B')]:lines.append('| '+label+' | '+' | '.join(fmt(contrast(f,name,d)) for f in ['texture','local_color'] for d in ['insert','restore'])+' |')
    lines+=['','E插入超过背景控制约0.575/0.813点，但其相对原图仅+0.474/+0.715点且两区间跨零；背景均有约−0.10点的小损伤。不可将前者写成“原图稳定修复”。E恢复的实际IoU−1.345/−1.056点及配对背景差均排零，说明即使不替换目标投影格，也可以撤回一部分编辑响应。','',
    'M−E四个直接比较区间均跨零；不能因为某项显著、另一项不显著，就宣布两者有显著差异。按两填充正反向实际/配对变化均超过1点的描述规则，M和E分别只有2/15例，不是普遍修复。','',
    '## 3. 像素变化与等格数的边界','',
    '以下是E插入相对原图；面积指标均按自身有效GT面积归一化。','',
    '| 指标变化 | 纹理 | 颜色 |','| --- | --- | --- |']
    for metric,label in [('coverage','自身覆盖'),('neighbor_error','选定邻居误报'),('background_error','背景误报'),('neighbor_untouched_fp','未被输入直接编辑的邻居误报')]:lines.append(f'| {label} | {fmt(get("texture","equal_exclusive",metric=metric))} | {fmt(get("local_color","equal_exclusive",metric=metric))} |')
    lines+=['','E恢复增加邻居误报：纹理+2.900 [1.061,5.100]、颜色+1.756 [0.274,3.404]自身面积点，两者均排零。它符合邻居错误重新出现的方向，但不能外推全部高拥挤失败。未编辑邻居区域的改善在插入时仅颜色排零，尚未跨填充确认远端错误普遍减少。','',
    '等格数没有强制等特征变化范数。均值L2范数纹理M/E/B为9.96/10.71/11.87，颜色8.95/10.56/11.15；空间排列、目标距离和卷积感受野也不同。E只是网格支撑不与目标重合，不能称与目标完全隔离或纯语义邻居。','',
    '## 4. 完整区域拆分与不可加性','',
    '| 全区域 | 纹理插入 | 纹理恢复 | 颜色插入 | 颜色恢复 |','| --- | --- | --- | --- | --- |']
    for region,label in [('mixed_all','全部M'),('exclusive_all','全部E'),('all_edit','M∪E，重放S052')]:lines.append('| '+label+' | '+' | '.join(fmt(get(f,region,d)) for f in ['texture','local_color'] for d in ['insert','restore'])+' |')
    lines+=['','区域数量不同，这张表只能描述完整区域作用，不能作为每格效果的公平排名。并集−M−E的IoU效应是非加性描述，不是自然因果份额：','',
    '| 填充 | 插入非加性 | 恢复非加性 |','| --- | --- | --- |']
    for f,lab in [('texture','纹理'),('local_color','颜色')]:lines.append(f'| {lab} | {fmt(contrast(f,"partition_interaction"))} | {fmt(contrast(f,"partition_interaction","restore"))} |')
    lines+=['','总体非加性区间跨零。个例572408中，原IoU68.12%，单独M/E插入仅69.44%/68.75%，两者一起74.91%；这个例子不能把整块收益按格子简单分摊，也不能据此认定全组存在普遍协同。561465的整块和两个分区均使原掩码变差，继续保留为反例。','',f'![固定正反例]({(OUT/"CASE_PANEL.png").as_posix()})','',
    '## 5. 主线如何收窄','',
    '目前最合理的机制表述是：**在部分框已较好的同类相邻误报实例中，原型分支的局部邻域表征能改变目标对邻居像素的响应；这一影响经过混合格与邻居独占格，固定原系数也会改变错误掩码。** 它支持继续研究像素的实例归属及局部读出，尚不能判原型容量、某个卷积层起源或系数相似为根因。','',
    '下一步应停止扩大格子/层的盲扫，改问一个可连接方法的问题：在原图被目标误占的邻居像素上，真实邻居候选是否已给出正确且更强的归属响应？若已有可用区分信息，检验候选间归属竞争能否利用它；若两者连像素响应排序也分不开，再考虑训练实例条件的空间表征。必须纳入自身像素损伤、背景对照与普通重叠消解基线，训练可用GT，正常推理不用测试GT。此项尚未执行，不是已确认方法创新。','',
    '## 6. 复现记录','',
    f'全部15例通过旧c/P/原IoU、四编辑输入P和目标输入像素保护；{A["verification"]["selfswaps"]}次局部自替换精确通过，{A["verification"]["S052_mask_replays"]}个S052整块插入/恢复掩码逐像素重放。{A["verification"]["original_hashes"]}个推理产物哈希核对通过。无依赖更改或新执行失败。','',
    '本轮bootstrap统一按image_id排序；有限随机重采样的排序改变会使与S052的CI端点略有差异，重放的原始掩码和均值完全相同。所有15目标进入统计，保存分区掩码、子集、输出掩码和P3局部256通道值。','',
    '- '+link(OUT/'protocol.json','冻结协议'),'- '+link(OUT/'manifest.json','15目标及格数'),'- '+link(OUT/'metrics.csv','780条指标'),'- '+link(OUT/'ANALYSIS.json','完整统计'),'- '+link(OUT/'COMPLETE.json','推理回执'),'- '+link(OUT/'ANALYSIS_RECEIPT.json','后处理及报告回执')]
    report=ROOT/'refine-logs/coco-structure/CELL_PARTITION_RESULTS_20260912.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8');shutil.copy2(report,OUT/report.name);shutil.copy2(__file__,OUT/Path(__file__).name);print('REPORT',report)

if __name__=='__main__':main()
