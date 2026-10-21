"""Create charts and a plain-language report from measured result JSON."""
from pathlib import Path
import html
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import read_json,write_json


def main(source,out):
    out.mkdir(parents=True,exist_ok=True);d=read_json(source);write_json(out/'results.json',d)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False,'savefig.facecolor':'white'})
    colors={'motion':'#8094a5','botsort':'#7187a4','deepocsort':'#9da6bf','all_appearance':'#daae70','margin_rule':'#b2bbc6','net_benefit':'#4d91c5','correctness':'#987ac7','trajectory':'#087f8c','extension':'#285fca','extension_trajectory':'#789bc9'}
    order=['motion','botsort','deepocsort','all_appearance','margin_rule','net_benefit','correctness','trajectory','extension','extension_trajectory']
    fig,axes=plt.subplots(1,3,figsize=(17,6.8),layout='constrained')
    for ax,split,title in zip(axes,['test','transfer','protected_confirm'],['原5视频：已见复核','斑马1/2：CGS-T独立确认','斑马3/4：PEA独立确认']):
        arms=[a for a in order if a in d['combined'][split]];vals=[d['combined'][split][a]['IDF1'] for a in arms];ys=np.arange(len(arms))
        ax.barh(ys,vals,color=[colors[a] for a in arms],height=.62);ax.set_yticks(ys,[d['labels'][a] for a in arms],fontsize=10);ax.invert_yaxis();ax.set_xlim(0,101)
        for y,v in zip(ys,vals):ax.text(v+1,y,f'{v:.2f}',va='center',fontsize=10)
        ax.set_title(title,fontweight='bold',pad=18);ax.set_xlabel('IDF1 (%) · 越高越好');ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
    fig.suptitle('同一检测与身份特征下的完整对照',fontsize=20,fontweight='bold');fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained');top=d['training_topology'];ys=np.arange(2);left=np.zeros(2)
    for key,c,label in [('positive','#087f8c','后续有益'),('neutral','#d9e1e9','无差异'),('negative','#c56258','后续有害')]:
        vals=[top[k][key] for k in ['new_connection','replacement']];ax.barh(ys,vals,left=left,color=c,label=label,height=.52)
        for y,l,v in zip(ys,left,vals):
            if v>=10:ax.text(l+v/2,y,str(v),ha='center',va='center',color='white' if key!='neutral' else '#354b60',fontweight='bold')
        left+=vals
    ax.text(94,0,'5组无差异 / 1组有害',va='center',color='#526b80',fontsize=11)
    ax.set_yticks(ys,['补接未配对轨迹（88组）','改写已有对应（507组）']);ax.invert_yaxis();ax.set_xlabel('训练干预组数 · 后续12帧正确身份观测变化');ax.set_title('为什么把“补接”和“改写”分开研究',fontsize=18,fontweight='bold');ax.legend(loc='upper right',frameon=False,ncol=3);fig.savefig(out/'topology.png',dpi=170);plt.close(fig)
    names=['pig_1','pig_2','horse_1','horse_2','horse_3','horse_7','zebra_1','zebra_2','zebra_3','zebra_4'];fig,ax=plt.subplots(figsize=(13,4.8),layout='constrained')
    for j,arm in enumerate(['trajectory','extension']):
        pts=[];vals=[]
        for i,n in enumerate(names):
            r=next((r for r in d['rows'] if r['sequence']==n and r['arm']==arm),None);b=next(r for r in d['rows'] if r['sequence']==n and r['arm']=='motion')
            if r:pts.append(i+(j-.5)*.34);vals.append(r['IDF1']-b['IDF1'])
        ax.bar(pts,vals,width=.32,color=colors[arm],label=d['labels'][arm])
    for x in [8,9]:ax.text(x,.28,'T未测',ha='center',fontsize=9,color='#617589')
    ax.axhline(0,color='#5b7083',linewidth=1);ax.set_xticks(range(len(names)),names,rotation=25);ax.set_ylabel('相对 ByteTrack 的 IDF1 差值（百分点）');ax.set_title('改善与损伤同时保留：逐视频结果',fontsize=18,fontweight='bold');ax.legend(frameon=False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True);fig.savefig(out/'per_sequence.png',dpi=170);plt.close(fig)
    boxes=[(35,100,230,100,'视频帧','当前与过去'),(315,100,250,100,'冻结 YOLO26-seg','框、掩码、外观特征'),(615,100,250,100,'身份投影','共享64维嵌入'),(315,295,250,100,'运动关联 B','保留当前高分对应'),(615,295,250,100,'外观候选 A','IoU与相似度限制'),(915,295,325,100,'竞争关联组','整体选择，保持一对一'),(915,505,325,100,'个体 ID 与轨迹','可视化与完整评价')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="660" viewBox="0 0 1280 660"><rect width="1280" height="660" fill="#f5f8fc"/><defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#647b91"/></marker></defs><g font-family="Microsoft YaHei, sans-serif">', '<text x="35" y="48" fill="#17354f" font-size="29" font-weight="bold">方法结构与本轮实际验证范围</text>']
    for x,y,w,h,t,sub in boxes:svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="white" stroke="#bed0e2"/><text x="{x+18}" y="{y+40}" font-size="23" fill="#183b54">{t}</text><text x="{x+18}" y="{y+73}" font-size="18" fill="#667f95">{sub}</text>')
    for path in ['M265 150 H315','M565 150 H615','M740 200 V295','M440 200 V295','M565 345 H615','M865 345 H915','M1077 395 V505']:svg.append(f'<path d="{path}" fill="none" stroke="#647b91" stroke-width="3" marker-end="url(#a)"/>')
    svg+=['<rect x="35" y="455" width="800" height="150" rx="14" fill="#e5eff8"/><text x="55" y="491" font-size="21" fill="#175071">比较三种决策：单帧收益 / 12帧轨迹收益 / 保护式补接</text><text x="55" y="529" font-size="19" fill="#36556c">12帧GT仅用于离线训练标签；部署推理不读取未来帧或GT。</text><text x="55" y="565" font-size="19" fill="#36556c">保护式补接不新增参数；保护当前高分对应不等于整段轨迹无损。</text></g></svg>']
    (out/'architecture.svg').write_text(''.join(svg),encoding='utf-8')
    lines=['# 密集视频身份保持：第二轮实验与真实结论','', '执行设备：离线笔记本 28358lan（RTX 4060 Laptop）。完成日期：2026-10-19。', '',
    '本轮已完成原因对照、现场稀疏补标注、三种新候选及已有方法比较。得到可复核的局部改善，但尚未建立稳定优于已有方法、或者新收益监督具有独立增量的结论。所有阴性与反例保留。','',
    '| 维度 | 本次完整实验的研究链路 |','|---|---|',
    '| 问题 | 首轮完整候选低于 ByteTrack。需要区分外观使用阶段、当前关联收益与后续轨迹影响，并核实“遮挡事件”的标签是否真的对应物理遮挡。 |',
    '| 机制假说 | 外观改写已有对应会传播错接；单帧收益标签遗漏持续影响。竞争解释是相似个体外观不具备稳定可区分性，当前特征与少量训练视频不足以辨别可靠补接。 |',
    '| 针对性方法 | 同 Deep OC-SORT 删除外观阶段；固定检测/身份投影，比较全部外观、规则、单帧收益CGS、同容量正确性、12帧收益CGS-T、保护式补接PEA、PEA加学习门；补充同输入BoT-SORT。三个学习头各803参数、各400步，训练/校准按视频拆分，推理无GT。 |',
    f"| 整体结果 | CGS-T在预先指定的斑马1/2上IDF1由 {d['combined']['transfer']['motion']['IDF1']:.2f} 提高至 {d['combined']['transfer']['trajectory']['IDF1']:.2f}（+{d['gates']['CGS_T_independent_zebra12']['delta_IDF1_native']:.2f}），较全部外观+{d['gates']['CGS_T_independent_zebra12']['delta_IDF1_all_appearance']:.2f}；但历史五视频只提高{d['gates']['CGS_T_development']['delta_IDF1_native']:.3f}。PEA在斑马3/4确认低于基线{abs(d['gates']['PEA_independent_zebra34']['delta_IDF1_native']):.2f}，未过Gate。 |",
    '| 研究决策 | 保留斑马1/2的真实改善作为有限原型成果；关闭当前CGS/CGS-T/PEA配置的追加拟合。主张下一步聚焦“错误补接的时序验证与回退”，并将身份恢复与运动状态更新分离。依据是保护当前高分边仍留下跨帧错接，而再加收益头没有稳定增量。唯一下一步是先冻结现有运动轨迹，在已补标的误接/正确补接事件上验证不污染运动状态的身份恢复；尚未启动，不扫描本批确认集规则。 |','',
    '## 为什么原方案会落后','',
    '1. 同一 Deep OC-SORT 开启外观，在开发集较关闭外观高2.138 IDF1，在历史五视频却低4.073。直接关闭重接阶段外观未过预设开发Gate，不能归因成一个开关错误。',
    '2. 595个训练冲突组中，74组（12.4%）的12帧收益符号与单帧不同。补接88组中有益82、有害1；改写已有对应507组中有益66、有害167。该规律支持保护方向，但不能直接推出跨场景无损。',
    '3. 实际部署式PEA输出的GT事后核对显示，horse_3有25次可判定的跨GT身份补接、22次同GT身份补接，另75次身份依据不足；zebra_4为2次跨身份、2次同身份、12次未知。这解释了保护规则仍会误接，未知事件没有强行算正确。',
    '4. pig_1的ByteTrack漏输2043/2623个GT观测；进一步核对全部conf≥.1的冻结检测，仅742/2623（28.3%）个GT能得到IoU≥.5的一对一检测对应。这个检测覆盖上限支持优先处理该视频的检测/可观测性问题，不能只靠身份关联修复。以上都是当前配置范围的证据，不是对全部外观方法的否定。','',
    '## 完整成绩','',
    'IDF1为百分制，差值为百分点。汇总由官方TrackEval按序列合并，不是逐视频分数简单平均。BoT-SORT/Deep OC-SORT是Ultralytics 8.4.100同检测、同身份特征、无GMC的受控版本，不是原论文榜单配置。','']
    for split in ['test','confirm','transfer','protected_confirm']:
        lines += ['### '+d['split_labels'][split], '', '| 方法 | IDF1 | HOTA | IDSW | FN |','|---|---:|---:|---:|---:|']
        for arm in order:
            if arm in d['combined'][split]:
                r=d['combined'][split][arm];lines.append(f"| {d['labels'][arm]} | {r['IDF1']:.3f} | {r['HOTA']:.3f} | {r['IDSW']} | {r['FN']} |")
        lines.append('')
    t=d['combined']['transfer'];lines += [f"CGS-T相对BoT-SORT +{t['trajectory']['IDF1']-t['botsort']['IDF1']:.3f}，相对Deep OC-SORT仅 +{t['trajectory']['IDF1']-t['deepocsort']['IDF1']:.3f}，低于同容量正确性对照 {t['correctness']['IDF1']-t['trajectory']['IDF1']:.3f}。因此不能宣称轨迹收益监督优于一般正确性学习。PEA在zebra_4较ByteTrack下降3.115，单视频损伤完整披露。",'',
    '## 标注与遮挡边界','',
    '现场补标为2个视频、16个关键帧、18个参考个体，144条记录中143条可用、1条未知。标签来自一次AI视觉复核，未借用跟踪器ID，不是人工密集金标准。ByteTrack、CGS-T和PEA在125个可评相邻关键帧配对上均保持原ID；这些短片没有给出方法额外改善的证据，也不能评价完整遮挡恢复。',
    '对公开GT间断按事前固定顺序抽查8个事件，其中4个在无GT时仍能看到目标，其余不能确认完全消失。未验证任何完整物理遮挡恢复事件；不把36个GT间断改称36次遮挡。补标文件和原图证据可继续人工复核。','',
    '## 创新定位与证据边界','',
    '可检验的新设计是“相对保留运动方案的收益监督”“冲突关联组整体选择”“已有高分对应保护”。匈牙利匹配、ReID、EMA、两阶段关联和一般外观融合属于既有技术。当前实测支持局部效用；机制独立增量与稳定泛化未建立，暂不将其包装成已成熟创新。',
    '原训练四视频中两条拟合、两条校准；共享身份前端曾用过四条，因此校准并非全链路未见。CGS-T因12帧完整窗口删去末尾不足窗口的17组，未额外训练完全同样本的单帧头；该差异也限制时域因果归因。每次新确认只运行冻结配置；horse_7、zebra_1/2被后续方案复用时均改称已见复核。斑马3/4没有拿来重新选阈值。所有成绩是少量完整序列的点估计，不提供虚假的独立帧显著性。','',
    '## 交付与复现','',
    '- 笔记本完整工作区：`D:\\thesis-video-v2-20261018`；旧实验保留在`D:\\thesis-video-20261018`。',
    '- [完整分数与来源](results.json)、[逐视频CSV](scores.csv)、[模型结构](architecture.svg)、[对照图](comparison.png)、[逐视频改善与损伤](per_sequence.png)。',
    '- 代码：仓库`shared/dense_video_identity/`；冻结协议与真实Run收据位于本Study。训练、评价、补标和失败运行均登记，特征大文件保留执行端。',
    '- 新展示包含4个完整配对视频：斑马1/2的改善、horse_3与zebra_4反例。画面来自真实输出，未人工修改轨迹。','',
    '参考：[AnimalTrack数据](https://hengfan2010.github.io/projects/AnimalTrack/download.html)、[ByteTrack](https://arxiv.org/abs/2110.06864)、[BoT-SORT](https://arxiv.org/abs/2206.14651)。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(str(out))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.source,a.out)
