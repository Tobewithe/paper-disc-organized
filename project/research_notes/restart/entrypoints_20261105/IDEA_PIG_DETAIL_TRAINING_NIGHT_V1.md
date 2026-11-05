# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md](../../../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_DETAIL_TRAINING_NIGHT_V1
kind: idea
title: 局部训练、细节增量与原能力保持：夜间四臂V1
summary: 四臂完整评价后全部未过采用线；P较同条件R提高2.515091 AP、较D提高4.105891，仍低于项目保留N且损伤超限，保持相对作用与净效用分开。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_OBSERVATION, OUTLINE_PIG_DETAIL_EVIDENCE, OUTLINE_PIG_TRAINING, OUTLINE_PIG_TRANSFER_GAP, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_PIG_ADAPTIVE_DETAIL_20261103]
exploration_status: closed
assessment: mixed
author: AI依据用户2026-11-03要求夜间较长训练序列、冒烟后排队的新增授权
curation_status: source_based_draft
---

## 问题、依据及待验证想法

既有原生训练出现过适配损伤，而近期128TRAIN上框较准且MaskIoU0.50–0.75的失败仅15个，原32例局部重看入口按规则停止。这不证明细节无用；也不能为凑数扩大那个Study。夜间新授权下，另立同初值、同真实源顺序、同更新预算的四臂比较，问正常开发集能力能否由局部训练及保护改变。

R是普通完整图续训，L和D具有相同局部几何、分别低通及保留细节，P只在D之上增加固定教师原Mask／类别响应保持。D−R检验裁剪训练复合效应，D−L检验细节相关增量，P−D检验保护的收益和损伤交换。不把三者混作一个已成立机制，不预设猪体语义或新颖性。

## 有限验证与当前阶段

[协议V1](../../../../experiments/pig_detail_training_night_20261103/PROTOCOL_V1.md)固定1272真实TRAIN、单seed、完整432dev一次末点评价及test426封存。工程真实scratch通过后，仅依据速度冻结共同正式更新数，再提交四臂独立同初值的串行Slurm及依赖评价／终态记录。总帽8GPUh／60CPU-coreh，失败计入；本序列至多1GPU、项目≤6GPU。

继续线、对照和护栏在协议中明确。未过则结束本版，不自动加seed、扩量或调参。Slurm1162冒烟已通过四臂各12次真实scratch更新，共48次／106前向；全部输入、L/D标签几何、D/P输入、教师／BN冻结、保持梯度及原生重载通过。按预定速度公式冻结各16,500步、共66,000正式更新。

正式作业1165–1171现已全部COMPLETED／0:0；四臂各16500步、共66000正式更新及完整432dev评价结束。N/R/L/D/P/V23 Mask AP分别为57.343500／52.734108／54.284877／51.143308／55.249199／60.753278。D−R=−1.590800、D−L=−3.141569，未支持本配置的裁剪或细节净收益；P−D=+4.105891、损伤从11.2888%降至4.4685%，保留保持项的相对作用，但P仍低于N 2.094301点、超原1%损伤护栏。

按原冻结规则结束全部当前配置，不自动调参、延训或加seed。具体Idea状态为已结束／混合：保持项有相对正效应，所有可采用性继续线失败。它不是整体方法有效，也不能被概括为所有细节无用。报告及来源见[完整结案](../../../../experiments/pig_detail_training_night_20261103/REPORT_V1.md)，原[交接](../../../../experiments/pig_detail_training_night_20261103/HANDOFF.md)保留启动时快照。

## 2026-11-04综合反馈：保持项相对有效，当前模型仍不采用

[复盘](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md#3-夜间四臂保持项缓解退化细节训练未取得净效用)补清：N是多阶段训练后的项目保留YOLO权重，R才是本轮同N起点、同主要预算的普通续训。P−R=+2.515091、L−R=+1.550769，不能把所有方法说成低于普通续训；P−N=−2.094301及4.4685%损伤仍不足以替换N，原停止决定保持。

D−L仅约束本版同裁剪下的细节／低通输入，裁剪并没有取得独立高分辨率新观察；16500步按工程速度确定也缺少学习曲线依据。广义细节价值、阶段效应与普通续训退化原因仍未知，不由这次复盘批准延训、换保持loss或恢复病例不足的小试。

本Idea继续`closed/mixed`；上述区分回到[广义细节Idea](../../../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)、[细节缺口](../../../outlines/OUTLINE_PIG_DETAIL_EVIDENCE.md)、[适配缺口](../../../outlines/OUTLINE_PIG_TRANSFER_GAP.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。

2026-11-04只读核到全部Run／accounting终态并复核311项产物SHA，实际17803GPU秒／107076CPU-core秒，无新作业或test读取。实现、数据、权重、逐例结果和冻结源全部在服务器，本地仅保留总结与来源指针。
