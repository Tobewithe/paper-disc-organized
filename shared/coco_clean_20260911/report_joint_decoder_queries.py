"""Record S070-S073 raw numbers, decision consequences and scope limits."""
from pathlib import Path
import json
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
RUN=ROOT/"diagnostics/joint_failure_decoder_20260913_v2"
QUERY=ROOT/"diagnostics/source_location_coefficient_20260913"
RANK=ROOT/"diagnostics/source_query_ranking_20260913"
ASSIGN=ROOT/"diagnostics/source_query_assignment_20260913"
REPORT=PROJECT/"refine-logs/coco-structure/JOINT_DECODER_QUERY_RESULTS_20260913.md"


def main():
    for r in [RUN,QUERY,RANK,ASSIGN]:assert json.loads((r/"COMPLETE.json").read_text())["status"]=="COMPLETE"
    task=json.loads((RUN/"task_summary.json").read_text())
    summary=pd.read_csv(RUN/"diagnostic_summary.csv")
    qsummary=pd.read_csv(QUERY/"summary.csv")
    d=pd.read_csv(RUN/"instances_analyzed.csv")
    r=pd.read_csv(RUN/"prior_residual_transitions.csv")
    ranks=pd.read_csv(RANK/"contrasts.csv")
    selected=json.loads((QUERY/"selection_summary.json").read_text())
    lines=["# S070—S073：联合失败的真实解码支持与源点读出诊断", "",
        "2026-09-13。本轮取得新的可判别事实，尚未找到已验证的论文方法。不启动端到端训练或继续扫加权损失。", "",
        "## 1. 本轮问题为什么不重复旧实验", "",
        "S067把完整COCO的拥挤召回差距主要定位到同一槽位的框差、掩码差状态；S069只在最终二值掩码上剪除真实矩形外像素，无法恢复被旧裁切剪掉的正确响应。S070实际改变解码裁切；S071以GT空间位置选系数，不用GT掩码IoU挑最优；S072拆出阈值/大小校准与像素排序；S073核对替代源点当前监督分配。它们分别决定是否追裁切、源点位置、校准或分配身份，不以oracle上涨本身作为方法成功。", "",
        "## 2. S070：保持原型与系数不变，只换实际解码裁切框", "",
        "官方yolo26m-seg同一权重，运行one-to-many分支，imgsz640、rect=False、FP32、TF32关闭、conf0.001、NMS0.7、max_det300。现有Windows conda pytorch与独立vendored Ultralytics8.4.143复用，没有安装或升级库。", "",
        "重做2,665张相关图片的前向，记录6,584个联合失败和1,106个框差掩码好的对照。7,690个原始RLE、分数类别及源系数逐项一致；原掩码矩阵乘使用原完整候选批次，避免单独算候选改变浮点路径。全量任务评价保留5,000张图、36,335个普通GT和446,097候选，仅改6,584个失败槽位掩码。", "",
        "干预使用自身原COCO栅格mask的紧整数外接框，映射到原模型输入；从原始未裁切cP响应重新二值化、裁切并缩放到原图。不是S069最终掩码交矩形。全正响应通过相同裁切与缩放给出精确输出支持。GT只在这个明确标注的诊断中使用，不能把成绩当作部署效果。", "",
        "| 完整COCO任务 | 原模型 | 真实解码GT框 | 变化 |", "|---|---:|---:|---:|",
        f"| Mask AP | 43.698 | {100*task['mask_ap']:.3f} | {task['delta_ap_points']:+.3f} |",
        f"| 高E4 Mask R75 | 52.132% | {100*task['hit75_high']:.3f}% | +6.717点 |",
        f"| 低E4 Mask R75 | 60.740% | {100*task['hit75_low']:.3f}% | +4.993点 |",
        f"| 低−高E4差距 | 8.609 | {100*task['gap_low_high']:.3f} | {task['delta_gap_points']:+.3f} |", "",
        "高E4 R75差的图片簇CI为[+6.132,+7.290]点，差距变化CI为[−2.351,−1.074]点；AP没有做显著性检验。修改框只用于mask裁切，候选原Box AP不变。", "",
        "| 联合失败的固定槽位 | 低E4（3,446） | 高E4（2,292） |", "|---|---:|---:|"]
    low=summary[(summary.scope=="joint_failure")&(summary.density=="low")].iloc[0]
    hi=summary[(summary.scope=="joint_failure")&(summary.density=="high")].iloc[0]
    for label,col,percent in [("原Mask IoU","original_iou",True),("GT裁切Mask IoU","gt_crop_iou",True),
        ("原自身覆盖","original_coverage",True),("无裁切自身覆盖","raw_coverage",True),
        ("GT裁切自身覆盖","gt_crop_coverage",True),("GT裁切后Mask75数量","new_mask75",False),
        ("精确原裁切支持不足75%数量","exact_support_below75",False),
        ("未裁切正确响应仍不足75%数量","raw_coverage_below75",False)]:
        x=f"{100*low[col]:.3f}%" if percent else f"{int(low[col]):,}"
        y=f"{100*hi[col]:.3f}%" if percent else f"{int(hi[col]):,}"
        lines.append(f"| {label} | {x} | {y} |")
    lines += ["", "精确输出层恒等分解：原遗漏=未裁切已为正但被裁切丢弃的自身像素+未裁切时仍没恢复的自身像素。高组按实例平均，两部分分别占自身GT的6.697%与12.736%；低组7.077%与10.454%。这说明裁切确有影响，但没有显示裁切遗漏本身在高组更大。后项不能直接叫系数错误，它仍混合c/P、插值和阈值等因素。", "",
        "高组GT裁切后830/2292（36.21%）达到固定Mask75，仍1,462个失败；381个未裁切自身正确像素本就不足75%。仅对框差掩码好者做独立诊断（不纳入AP改动）时，高组有20/310在GT裁切后降到Mask75以下，所以‘真实框必然逐实例改善’也不成立。GT框经过模型栅格后高组29个有效支持覆盖不足95%，不应把这个框当作完美支持上界。", "",
        "## 3. 源点位置的相关性与配方检验", "",
        "高E4框差掩码差者有34.69%的实际候选源点中心落在自身可见mask之外，框差掩码好对照为10.97%。类别×COCO大小×FPN stride共同支持标准化后，为33.29%与17.51%（保留1,268个失败/193个对照）；低E4也有类似差异，不能称密集特有机制。源点中心或stride格子不是完整感受野，也不是训练分配身份。", "",
        f"S071从源点不在自身内部、无任何保留同类Mask75的FN/FP残余病例中初筛{selected['initial_candidates']}例，{selected['eligible']}例存在半径2格以内的自身内部源点及等距离镜像非自身源点，二者均在同一GT框内。进一步配对高低E4的类别、大小、stride及Box IoU差≤0.05，得到48对/96图：FN16对、FP32对。每图只用一个目标，所有选择在干预前固定，不以任何替代mask IoU选候选；大量无共同支持目标排除，不能当全体失败的随机样本。", "",
        "所有对照保持原型、原目标框及输出候选身份不变。自身内部源点与镜像源点同尺度且距原源点相同；另有3×3系数均值和只用原预测掩码内部距离选源点。再将所有系数放到同一GT框下复算以检查裁切交互。", "",
        "| 系数来源 | 原框：高E4 IoU | 原框：低E4 IoU | 共同GT框：高E4 IoU | 共同GT框：低E4 IoU |", "|---|---:|---:|---:|---:|"]
    labels={"original":"原源点","own_query":"自身内部源点（GT选）","reflected_nonown":"等距离镜像非自身点","local_mean":"3×3均值","predicted_interior":"预测掩码内部点（预测选）"}
    for arm,label in labels.items():
        vals=[]
        for crop in ["original_predicted_box","same_gt_box"]:
            for group in ["high","low"]:
                z=qsummary[(qsummary.arm==arm)&(qsummary.crop==crop)&(qsummary.density==group)].iloc[0]
                vals.append(f"{100*z.iou:.3f}")
        lines.append("| "+label+" | "+" | ".join(vals)+" |")
    lines += ["", "原框高组自身内部源点−原源点IoU −2.751点，48个匹配对bootstrap CI[−6.470,+0.463]；共同GT框下−4.218[−8.555,−0.641]点。它比镜像非自身点平均好，但高组区间仍含零，且两者都可能比原源点差。预测内部点−原源点−3.125[−6.200,−0.536]，3×3均值−6.579[−10.373,−3.253]。这些负结果拒绝本次直接搬用/平均配方，不是空间特征无用的证明。", "",
        "## 4. S072：是否仅因为掩码大小/阈值变了", "",
        "在相同原输入640裁切支持内排除crowd，使用保存的c/P计算排序；不再做网络前向。GT仅用于排序标签和等自身覆盖率阈值。保留阈值所有并列值，最大覆盖差0.016个百分点，没有靠拆分并列值优化负类。输入域量与原图COCO IoU不同。", "",
        "| 高E4排序/覆盖诊断 | 原源点 | 自身内部源点 | 差值95%CI |", "|---|---:|---:|---|" ]
    for metric,label in [("auc_all","自身对全部负类AUC"),("auc_same","自身对同类邻居AUC"),
                         ("auc_background","自身对背景AUC"),("precision","匹配原自身覆盖率后的Precision")]:
        x=ranks[(ranks.density=="high")&(ranks.arm=="own_query")&(ranks.metric==metric)].iloc[0]
        lines.append(f"| {label}（n={x.n}） | {x.original:.4f} | {x['new']:.4f} | {x.delta:+.4f} [{x.ci_low:+.4f},{x.ci_high:+.4f}] |")
    lines += ["", "没有发现内部源点提供更好的像素排序。背景AUC下降的点态区间排零，但多指标未校正，且源点监督身份仍是混杂，不能以该项定为机制结论。", "",
        "## 5. S073：内部网格点是否真的给自身实例监督", "",
        "同96图重放官方TaskAlignedAssigner的one-to-many参数（topk10、alpha0.5、beta6、stride8/16/32），所有普通GT一起参与，使用当前冻结推理分数、框和执行模型的实际anchors。没有训练、反向或BN更新；只判断当前检查点的无增强分配，不反推预训练历史。S071各指定源系数逐项复现。", "",
        "| 高E4源点，n=48 | 自身正样本 | 其他实例正样本 | 非正样本 |", "|---|---:|---:|---:|",
        "| 原源点 | 45 | 0 | 3 |", "| GT选自身内部点 | 31 | 4 | 13 |", "| 预测内部点 | 29 | 5 | 14 |", "| 镜像非自身点 | 24 | 11 | 13 |", "",
        "这解释了为什么不能把几何内部点直接当更合适的系数来源。仅限原点与内部点当前都分给自身的30个高组实例，内部点−原点IoU为+0.867[−0.481,+2.407]点；共同GT框下+0.145[−1.346,+1.653]。该事后子组不再保持高低一一配对，也没有稳定优势，不能据此启动新增查询模块。", "",
        "## 6. 后续决定", "",
        "1. 裁切范围是有明确因果干预收益的输出环节，当前GT框修复AP+2.025，但仍有大量响应问题；这不证明检测框误差是网络根因。保留为强诊断控制。", 
        "2. 停止‘源点移进自身’、‘局部系数平均’、‘按当前mask内部选点’这三个直接配方。源点不在自身的相关性不足以支撑方法，当前分配检查进一步揭示监督身份混杂。", 
        "3. 下一次学习实验前，先针对GT裁切后仍失败且无好候选的目标，检查响应错误是否对原型插值/空间分辨率特别敏感，或仅是当前实例读出边界不佳。该检查必须加入相同原型、相同支持和同预算系数/阈值控制，并使用固定新开发样本；不能再把S019/S040的普通GT系数求解重新包装成新发现。", 
        "4. 尚未提出通过验证的新方法，不恢复CCL，不声称空间信息方向全部否定。训练阶段当然可以用GT；最终方法需要在新图上按预测信息输出改善。", "",
        "## 7. 产物与失败记录", "",
        "S070运行310.42秒（含全val评价）；S071主查询19.31秒，S0738.70秒，另有统计时间。S070初次读取YOLO外层对象保留的旧anchor缓存6174点而失败，未生成首图完成记录；实际Predictor会deepcopy模型。v2从实际执行的backend读8400anchors并逐项核验源索引；失败目录保留FAILED.json和源快照。模型权重、标签和环境未因此改变。", "",
        "- [S070结果与逐实例](../../experiments/coco_clean_20260911/diagnostics/joint_failure_decoder_20260913_v2/diagnostic_summary.csv)",
        "- [S070全COCO任务AP](../../experiments/coco_clean_20260911/diagnostics/joint_failure_decoder_20260913_v2/task_summary.json)",
        "- [S071源点对照与区间](../../experiments/coco_clean_20260911/diagnostics/source_location_coefficient_20260913/paired_contrasts.csv)",
        "- [S072排序控制](../../experiments/coco_clean_20260911/diagnostics/source_query_ranking_20260913/contrasts.csv)",
        "- [S073监督分配](../../experiments/coco_clean_20260911/diagnostics/source_query_assignment_20260913/summary.csv)", ""]
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    shutil.copy2(__file__,RUN/"source"/Path(__file__).name)
    facts=PROJECT/"RESEARCH_FACTS.md"
    content=facts.read_text(encoding="utf-8")
    snapshot=RUN/"RESEARCH_FACTS_before_S070_S073.md";assert not snapshot.exists();shutil.copy2(facts,snapshot)
    link="[S070—S073](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/JOINT_DECODER_QUERY_RESULTS_20260913.md)"
    additions=[
      f"| F36 | S070实际解码裁切干预：7,690槽位重放，完整val仅改6,584联合失败mask，原c/P/分数/框输出不变。MaskAP43.698→45.723，高E4 R7552.132→58.849，差距8.609→6.884。高组830/2292固定Mask75恢复；381个未裁切正确响应本就不足75%。 | GT紧框作用在实际解码640域，不是S069最终mask交矩形。原遗漏拆分高组裁切6.697%、未裁切响应12.736%自身面积；后项不能归为系数根因。GT裁切也损伤20/310掩码成功控制。 | {link} |",
      f"| F37 | S070高E4框差mask差源点中心在自身mask外34.69%，框差mask好10.97%；类别×大小×stride共同支持后33.29/17.51%。S071配对48高/48低失败，在相同P和原框借用GT内部源系数：高组IoU−2.751[−6.470,+0.463]点；预测内部点−3.125、局部均值−6.579。 | 同尺度等距离镜像作控制，不用候选maskIoU选替代。相关性未转换成直接配方收益；样本是空间资格及原源点在外的条件子集，非总体AP。停止直接搬用/平均，不否定所有空间建模。 | {link} |",
      f"| F38 | S072同96目标排序控制：高组内部系数的自身对负类AUC0.8638→0.8486、背景AUC0.8529→0.8312；匹配原自身覆盖后的Precision0.5826→0.5727。 | 没有显示更好的排序，不能用面积/阈值变化掩盖负结果。GT设阈值、640输入域、条件队列，不能作方法AP；需结合S073监督身份混杂。 | {link} |",
      f"| F39 | S073当前官方one-to-many分配：高组48原源点45给自身/3非正；内部替代点31给自身/4给他人/13非正。两者都给自身的30例，内部系数−原系数IoU+0.867[−0.481,+2.407]，共同GT框+0.145[−1.346,+1.653]。 | 内部几何位置不等于自身监督身份；事后子组无稳定优势。是冻结检查点无增强当前分配，非历史训练、反向梯度或缺监督根因。 | {link} |"
    ]
    at=content.index("\n",content.index("| F35 |"));content=content[:at]+"\n"+"\n".join(additions)+content[at:]
    content=content.replace("S000—S069","S000—S073",1)
    current="\n**S070—S073新增结论**：实际解码GT框恢复高E4联合失败830/2292，仍不能只追框；直接内部源点替换/均值没有优势，当前分配身份说明内部点不一定给自身监督。继续沿裁切后残余响应查空间表示与读出，禁止据源点位置相关性直接加模块。详见"+link+"。\n"
    content=content.replace("## 1. 后续探索先记住这些事实",current+"\n## 1. 后续探索先记住这些事实",1)
    content=content.replace("## 5. 尚未回答的问题与已计划但未执行的工作", "## 5. 尚未回答的问题与已计划但未执行的工作\n\nS070精确裁切/响应分解与S071—S073源点、排序、当前分配对照已完成，见F36—F39。下述S067—S069‘下一步’现为历史启动记录，不重复运行。",1)
    content += "\n2026-09-13 S070—S073：实际解码、候选源点查询、等覆盖排序与当前官方分配完成。记录源点直接替换不成立的具体配方及监督身份混杂；原型/系数深层根因未定，方法主线仍未验证。\n"
    facts.write_text(content,encoding="utf-8")
    tracker=PROJECT/"refine-logs/coco-structure/EXPERIMENT_TRACKER.md"
    with tracker.open("a",encoding="utf-8") as f:
        f.write("\n\n# S070—S073 COMPLETE（2026-09-13）\n\n"
            "S070实际640解码GT框，原始RLE与源c逐项重放，7690槽位/2665图；全val AP+2.025，E4差距−1.725，高联合失败830/2292达到Mask75。剩余381原始uncropped自身响应不足75%，说明不能仅修框。初次stale anchors失败保留，v2使用deepcopied执行模型head，310.42秒完成。\n\n"
            "S071固定48高/48低条件匹配目标，GT空间位置选内部源系数、等距离镜像、局部均值和预测内部点；高组原框内部−原IoU−2.751，均值/预测内点亦退化。S072等覆盖/排序未显示内部更好。S073当前TAL发现内部 donor31/48给自身而original45/48；二者都own30例仅+0.867点CI跨零。停止这些直接源点配方，不把关联当方法，也不否定所有空间建模。报告JOINT_DECODER_QUERY_RESULTS_20260913.md。无网络训练/环境改动/定时任务更改。\n")
    print("S070-S073 report, source, facts F36-F39 and tracker updated.")


if __name__=="__main__":main()
