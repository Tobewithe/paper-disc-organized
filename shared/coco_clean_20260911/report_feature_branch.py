"""S052 report and fixed case figures from completed measurements only."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
import json,shutil
import numpy as np,pandas as pd,cv2
from pycocotools import mask as mu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_subtype_neighbor import sha,dump

BASE=Path(__file__).resolve().parent;ROOT=BASE.parent.parent
OUT=BASE/'diagnostics/proto_feature_swap_20260912_v2';SRC=BASE/'diagnostics/subtype_neighbor_20260912'
REPORT=ROOT/'refine-logs/coco-structure/FEATURE_SWAP_RESULTS_20260912.md'
A=json.loads((OUT/'ANALYSIS.json').read_text());D=pd.read_csv(OUT/'metrics.csv')
META={r['image_id']:r for r in json.loads((SRC/'manifest.json').read_text())['pairs']}

def get(group,fill,stage='input0',region='edit_location',direction='insert',contrast='neighbor',metric='mask_iou'):
    return next(x['metrics'][metric] for x in A['summary'] if x['grouping']=='error_type' and x['group']==group and x['fill']==fill and x['stage']==stage and x['region']==region and x['direction']==direction and x['contrast']==contrast)
def fmt(x):return f"{x['mean']*100:+.3f} [{x['ci95'][0]*100:.3f}, {x['ci95'][1]*100:.3f}]"
def link(p,label=None):return f'[{label or p.name}]({p.as_posix()})'
def readmask(x):return mu.decode(x).astype(bool)

def figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,direction in zip(axs,['insert','restore']):
        for offset,fill,color in [(-.12,'texture','#147d92'),(.12,'local_color','#a7442c')]:
            s=[get('same_neighbor',fill,stage, direction=direction) for stage in ['input0','input1','input2','fused','decoder']]
            means=np.array([z['mean'] for z in s])*100;lo=np.array([z['ci95'][0] for z in s])*100;hi=np.array([z['ci95'][1] for z in s])*100
            ax.errorbar(np.arange(5)+offset,means,yerr=[means-lo,hi-means],fmt='o',capsize=3,color=color,label=fill.replace('_',' '))
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(5),['P3','P4','P5','Fused','Decoder']);ax.set_title('Insertion minus original' if direction=='insert' else 'Restoration minus edited');ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Mask IoU change (percentage points)');axs[1].legend(frameon=False)
    fig.suptitle('Same-class neighbor false positives: 15 fixed targets, edit-location feature patch',fontsize=11)
    fig.text(.5,.01,'Original coefficient and box fixed. Pointwise 95% image-bootstrap CIs; exploratory comparisons.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.95]);fig.savefig(OUT/'LAYER_EFFECTS.png',dpi=180);fig.savefig(OUT/'LAYER_EFFECTS.svg');plt.close(fig)
    cases=[(572408,'input0','edit_location','Neighbor FP / P3 edit-location'),(82812,'fused','target','Target FN / fused target region'),(561465,'input0','edit_location','Adverse case / P3 edit-location')]
    fig,axs=plt.subplots(3,4,figsize=(12,10.2));case_data=[]
    for row,(iid,stage,region,label) in enumerate(cases):
        pack=json.loads((OUT/'pairs'/str(iid)/'masks.json').read_text());geo=np.load(SRC/'pairs'/str(iid)/'geometry.npz');own=geo['own'].astype(bool)
        im=cv2.cvtColor(cv2.imread(str(BASE/'local_readout_runtime_20260912/data/images/val2017'/f'{iid:012}.jpg')),cv2.COLOR_BGR2RGB)
        chosen=[pack['original']]
        for s,r,direction in [('endpoint','full','edited'),(stage,region,'insert'),(stage,region,'restore')]:chosen.append(next(v['mask'] for v in pack['variants'] if v['mode']=='neighbor' and v['fill']=='texture' and v['stage']==s and v['region']==r and v['direction']==direction))
        ys,xs=np.nonzero(own);pad=max(16,int(max(np.ptp(xs),np.ptp(ys))*.38));x0=max(0,xs.min()-pad);x1=min(im.shape[1],xs.max()+pad+1);y0=max(0,ys.min()-pad);y1=min(im.shape[0],ys.max()+pad+1)
        vals=[]
        for col,(encoded,title) in enumerate(zip(chosen,['Original c0 P0','Edited prototype c0 P1','Local feature insertion','Local feature restoration'])):
            m=readmask(encoded);v=float((m&own).sum()/max((m|own).sum(),1));vals.append(v);view=im.astype(float)/255
            for z,color in [(m&own,np.array([.15,.85,.3])),(m&~own,np.array([1,.17,.2])),(own&~m,np.array([1,.85,.03]))]:view[z]=.45*view[z]+.55*color
            axs[row,col].imshow(view[y0:y1,x0:x1]);axs[row,col].set_title(f'{title}\nMask IoU {v*100:.2f}%',fontsize=10);axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
        axs[row,0].set_ylabel(f'{label}\nCOCO {iid}; box IoU {META[iid]["original_box_iou"]*100:.2f}%',fontsize=10)
        case_data.append(dict(image_id=iid,stage=stage,region=region,original=vals[0],edited_prototype=vals[1],insert=vals[2],restore=vals[3],box_iou=META[iid]['original_box_iou']))
    fig.suptitle('Fixed illustrative cases: reversible recovery and retained counterexample',fontsize=13)
    fig.text(.5,.025,'Green: true positive; yellow: target missed; red: false positive. All panels use the original image as display background.\nTexture fill, seed 0. Fixed original coefficient/box. Metrics use the full mask; panels are cropped for visibility.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.96]);fig.savefig(OUT/'CASE_PANEL.png',dpi=170);fig.savefig(OUT/'CASE_PANEL.svg');plt.close(fig);dump(OUT/'ILLUSTRATIVE_CASES.json',case_data)
    return case_data

def main():
    cases=figures();lines=['# S052：原型分支局部特征的可逆影响路径','',
    '**结果推进到了一条具体的传播路径：在同类邻居误报组，P3编辑位置的特征替换可提高原掩码IoU，反向恢复可撤回改善；两种填充的点态区间都排零。** 但这些位置包含目标/邻居混合格子，当前定位的是影响所经过的区域，不是已证明的错误起源，也不是新方法的COCO AP。','',
    f'2026-09-12。本地RTX 5060 Ti，{A["images"]}图/目标，{A["rows"]:,}条干预/端点记录，推理{A["seconds"]:.3f}秒。保持S051全部44例：同类误报15、自身遗漏15、背景14，前两型各6小/6中/3大，背景6/6/2。本轮仅复用纹理/颜色各seed0，不是S051的三填充seed平均，更不是三训练种子。没有训练、GT系数拟合或新方法评价。','',
    '## 1. 实验如何区分路径','',
    '实际Proto26输入为P3 256×80×80、P4 512×40×40、P5 512×20×20。P4/P5经各自卷积并最近邻上采样后与P3相加，再经feat_fuse；另记录上采样后的cv2输出256×160×160。按实际640 letterbox投影GT/编辑区域，再最大池化到各网格。','',
    '每层分别替换目标区域、编辑位置区域、全图的全部通道。插入：以原图特征为底，将编辑图局部特征带入；恢复：以编辑图特征为底，把该处原图特征放回。所有掩码始终使用原系数c0和原预测框b0；因此端点是c0P1，不是编辑图正常输出c1P1，也没有改变检测分数或重新挑选候选。GT用于已固定选样、输入保护、特征区域和评价。','',
    '插入差值=插入后−原图；恢复差值=恢复后−编辑端点。恢复为负表示撤回编辑带来的收益，两者参照不同。背景位置执行相同操作，并报告图片内邻居−背景差。','',
    '## 2. 同类邻居误报：P3编辑位置','',
    '以下单位均为IoU百分点，括号为2,000次图片配对bootstrap的点态95%区间；一图一个目标。多层、多区域比较未作多重校正，是探索性定位。','',
    '| 填充 | 比较 | 插入变化 | 逆向恢复变化 |','| --- | --- | --- | --- |']
    for fill,label in [('texture','纹理'),('local_color','颜色')]:
        for contrast,lab in [('neighbor','邻居操作绝对变化'),('background','背景操作绝对变化'),('neighbor_minus_background','邻居−背景')]:lines.append(f'| {label} | {lab} | {fmt(get("same_neighbor",fill,contrast=contrast))} | {fmt(get("same_neighbor",fill,direction="restore",contrast=contrast))} |')
    lines+=['','效果既超过原图，也超过背景控制，所以不是只靠损伤背景控制得到的相对优势。按观察后定义的严格描述规则（两填充插入与配对插入都>1点、恢复与配对恢复都<−1点），P3编辑位置5/15例通过；这不是独立确认成功率。','',
    '该路径更接近减少邻居误报，而非整体缩小目标。下面均为插入相对原图，错误/覆盖按自身有效GT面积归一化；邻居错误不是除以邻居面积，也不等同完整IoU。','',
    '| 像素指标变化 | 纹理 | 颜色 |','| --- | --- | --- |']
    for metric,label in [('coverage','自身覆盖'),('neighbor_error','选定邻居错误'),('background_error','背景错误'),('own_near_tp','临邻自身正确像素'),('neighbor_untouched_fp','未编辑邻居错误')]:lines.append(f'| {label} | {fmt(get("same_neighbor","texture",metric=metric))} | {fmt(get("same_neighbor","local_color",metric=metric))} |')
    lines+=['','未编辑邻居误报的区间仍跨零；不能把全部改善都解释成远离编辑区域的实例分离。自身覆盖区间跨零也不构成已完成非劣效证明。','',
    '## 3. 层级对照及其他失败类型','',
    '| 编辑位置所在层 | 纹理插入 | 纹理恢复 | 颜色插入 | 颜色恢复 |','| --- | --- | --- | --- | --- |']
    for stage,label in [('input0','P3 80×80'),('input1','P4 40×40'),('input2','P5 20×20'),('fused','融合后80×80'),('decoder','解码中160×160')]:lines.append('| '+label+' | '+' | '.join(fmt(get('same_neighbor',f,stage,direction=d)) for f in ['texture','local_color'] for d in ['insert','restore'])+' |')
    lines+=['',f'![各层绝对变化]({(OUT/"LAYER_EFFECTS.png").as_posix()})','',
    '观察后的图片内P3−P4插入差：纹理+2.626 [0.736,4.526]、颜色+3.134 [1.206,5.191]点。但不同尺度的区域粗化、感受野及混合比例不同，跨尺度混合特征也可能离开自然分布，不能据此宣布多尺度相加是根因。','',
    '| 其他组/区域 | 纹理插入 | 纹理恢复 | 颜色插入 | 颜色恢复 |','| --- | --- | --- | --- | --- |']
    for group,stage,region,label in [('target_fn','input0','edit_location','自身遗漏：P3编辑位置'),('target_fn','fused','target','自身遗漏：融合后目标区域'),('background','input0','edit_location','背景误报：P3编辑位置')]:lines.append('| '+label+' | '+' | '.join(fmt(get(group,f,stage,region,d)) for f in ['texture','local_color'] for d in ['insert','restore'])+' |')
    lines+=['','自身遗漏组尚未在两种填充下建立一致的融合后目标区域完整IoU恢复。三种失败类型的类别、大小构成并未完全匹配，不能从分别显著/不显著直接推出组间差异或拥挤特异性。全部类型×大小×层×区域统计保留在ANALYSIS.json。','',
    '## 4. 具体实例与反例','',
    '| 图片/操作（纹理seed0） | 原图 | 只换整份原型c0P1 | 局部插入 | 局部恢复 |','| --- | --- | --- | --- | --- |']
    for x in cases:lines.append(f'| {x["image_id"]} / {x["stage"]} {x["region"]} | '+ ' | '.join(f'{x[k]*100:.3f}%' for k in ['original','edited_prototype','insert','restore'])+' |')
    lines+=['','82812框IoU已达96.61%。固定旧系数和旧框，只把编辑后的融合层目标区域带入，原Mask IoU约72.58%升至79.88%；反向放回原特征降至72.63%。这说明改善可经该空间表征传递，旧系数也能解码改善后的原型，不证明系数无问题，更不证明错误首先发生于融合层。','',
    '561465保留为受损反例：同样P3编辑位置操作降低原IoU，逆向恢复也未回到原图。不能把均值改善写成普遍单向干扰。','',f'![固定正反案例]({(OUT/"CASE_PANEL.png").as_posix()})','',
    '## 5. 最需要继续拆分的限制','',
    '**15/15同类误报例的P3编辑位置格子都与目标投影格子有重叠。** 平均覆盖目标格子的7.595%，占编辑格子的30.445%；P4/P5的后者为52.556%/76.667%。因此原输入中保护了目标像素，不代表后续替换了纯邻居特征。当前没有P3零直接格子重叠子集，无法靠剔除病例解决这一点。160×160解码层同类组也全部有少量重叠。','',
    '下一项有判别力的实验应固定这15例、P3和两种填充，把编辑位置拆成“目标/邻居混合格”“不与目标投影重合的邻居格”，再做相同插入/恢复、背景控制和等单元数量对照。即便不重合，卷积感受野仍会覆盖邻域，不能称完全隔离。这个拆分尚未运行；不继续盲扫更多层。','',
    '若主要收益来自混合格，可优先研究接触边界的实例归属表征/解码；若纯邻居格也能可靠传递收益，则继续追查邻域信息如何传播到目标。两者都仍需训练时利用GT、测试时自动预测的实现，以及正常COCO整体/拥挤分组评价。当前不恢复CCL或端到端训练。','',
    '## 6. 完整性、执行修正与证据','',
    '99个原始产物哈希核对通过，44例原始c/P/IoU与保存输入端点重放，220次同层自替换精确通过，176个图片×填充×位置组合的全图融合/解码替换与整份原型端点完全一致。后者是模块顺序带来的端点检查，不是独立机制证据。44例全部进入统计，没有根据收益排除。','',
    '首次v1调用失败于GPU输入/CPU权重：AutoBackend复制模型，直接子模块重放误用了原始加载对象。未产生干预结果；保留目录/日志。v2仅将引用改为实际推理后端的GPU模块，此后普通重放及自替换通过。没有更改模型权重、依赖版本或采用重复OpenMP绕过。运行使用本地固定vendor Ultralytics8.4.143 / Torch2.9.1+cu128。','',
    '原始记录与后处理分开：COMPLETE.json覆盖推理产物；ANALYSIS_RECEIPT.json补充统计、44份网格mask、图表、来源依赖与本报告哈希。', '',
    '- '+link(OUT/'protocol.json','干预协议'),'- '+link(OUT/'metrics.csv','全部5632条结果'),'- '+link(OUT/'ANALYSIS.json','全部分组统计'),'- '+link(OUT/'LOCALITY_ANALYSIS.json','尺度/空间补充对照'),'- '+link(OUT/'PATCH_GEOMETRY.csv','实际网格重叠'),'- '+link(OUT/'COMPLETE.json','推理回执'),'- '+link(OUT/'ANALYSIS_RECEIPT.json','后处理回执')]
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf8');shutil.copy2(REPORT,OUT/REPORT.name);shutil.copy2(__file__,OUT/Path(__file__).name)
    print('REPORT',REPORT);print('FIGURES',OUT/'CASE_PANEL.png')

if __name__=='__main__':main()
