---
schema_version: 1
id: REVIEW_PIG_APPEARANCE_STABILITY_V1_20261103
kind: review
title: 成像外观敏感性首试：本范围未发现大幅实例成功数损失
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_TRANSFER_GAP, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_DATA, OUTLINE_PIG_TASK, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_EVIDENCE]
idea_ids: [IDEA_PIG_APPEARANCE_STABILITY_V1, IDEA_PIG_SYNTHETIC_COUNTERFACTUALS, IDEA_PIG_COVERAGE_TRANSFER]
coverage: declared
verdict: frozen_condition_effect_below_prespecified_size_no_method_training
author: AI依据真实工程／正式运行、原标签与完整预测独立复算及8张固定预览审阅
scope:
  target: 固定模型和几何时的实例成功数对成像条件变化的敏感性
  metric: 预定160格点IoU50最大一对一匹配成功数的真实损失与合成差异
  dataset: 哈希预选128真实TRAIN图／1460实例及30组合成A/B／280owner
  protocol: experiments/pig_appearance_stability_20261103/PROTOCOL_V1.md
  method_utility_tested: false
  independent_generalization_tested: false
evidence:
  - path: experiments/pig_appearance_stability_20261103/runs/RUN_PIG_APPEARANCE_FORMAL_20261103_b34157c23fcd/SUMMARY.json
    revision: ee0caf7ee68738376b8c94d43b78f50ccf5a5ec19b1f7b1202e42519dafd00d2
    locator: 冻结主读数、全部对象覆盖、区间及原停止出口
    relation: references
  - path: experiments/pig_appearance_stability_20261103/runs/RUN_PIG_APPEARANCE_FORMAL_20261103_b34157c23fcd/run.json
    revision: f7018cb2c98e5c008f6fc2a4cacc7af9d1157b6340fbb323b8c6b787f25f161b
    locator: 实际阶段、命令、工作目录、源码／环境／输入身份、开始与终态
    relation: references
limitations:
  - 真实TRAIN为模型已见来源；IoU50和低输出阈值下原图成功数触顶，不证明细边界、AP或未见域稳定。
  - 真实数字色调与合成物理照明是不同干预，分别解释，不能合并分母。
  - 合成只在原owner_valid内诊断，未改变辅助标签权限或unknown语义。
  - 简单外观范围未达效应线不能排除更强成像变化、场景几何或其他跨条件问题。
---

完成了首个真正模型干预实验：固定原生N权重与解码，处理128张预选真实TRAIN和全部30组成对照明渲染，0优化。**在预先冻结的“可同时成功的实例数”读数上，真实损失2／1460＝0.136986%，合成配对差异0／278；均低于选择方向用的效应线。当前不启动针对这种粗粒度损失的保持方法训练。**

| 维度 | 本次实验的整体研究链路 |
|---|---|
| **问题** | 剩余缺口中，跨条件可靠性有长期价值；现成A/B渲染有几何不变照明对照，真实TRAIN也可做固定色调干预，能立即测模型反应。历史观察缺可核序列、额外细节缺独立原图来源、关系标注缺人工成本；前一片段缓存入口的支持不足保留。本次选择依据见[DESIGN_AND_SELECTION.md](DESIGN_AND_SELECTION.md)。 |
| **机制假说** | 若当前实例区分强依赖成像外观，固定几何、标签和模型时应出现足够大的实例成功数损失。竞争解释是本范围外观已足够稳健，或敏感性仅来自有限合成域。真实主量预定至少5%且簇区间下界>2%；合成对照至少10%且下界>2%。这只是首试投入门槛，不是旧SPM效用Gate。 |
| **针对性方法** | 固定原生N与无GT正常O2O前向；128 TRAIN按源ID哈希预选，分别原图／gamma0.65／gamma1/0.65，padding不变。全部30组A/B核同body、mesh、相机、成像映射、完整标签SHA及数组。真实用原160独立Mask，合成仅在原owner有效域诊断，不造正常分割GT。预测先全部完成并保存640 RLE，再解码GT计算固定160采样IoU50最大匹配；两侧分别整簇bootstrap10000次。 |
| **整体结果** | 真实原图／bright／dark成功数1460／1458／1460，全部1460 GT；逐图预定最坏损失2／1460＝0.136986%，13个声明group簇区间[0,0.395849%]，低于5%。合成A/B各278成功，全部280 owner均有有效支持，30组逐对成功数差异0／278，经验bootstrap区间[0,0]；这不代表总体风险为0。两侧基线资格均满足，同RGB的12组正式控制逐值一致，模型状态SHA不变。无新增方法，故“相对原生方法收益／已有方法增量”不适用；读数不是COCO AP或官方召回。 |
| **研究决策** | **结束“本组外观变化造成大幅粗粒度实例成功数损失”的具体验证入口，降低仅靠这组变化开展保持训练的优先度。** 广义跨条件能力与受控合成工具保持未知，不能由本试验否定。原图在IoU50低阈值候选读数上触顶，真实来源已见；细边界、置信度排序、新相机／农场、强成像和场景几何尚未检验。需要这些对象的独立失效证据才能决定后续，不为保住主张修改门槛或扩量。 |

## 冻结主量与实际读数

真实每图原成功数为n，bright／dark为b、d，`L_real=sum max(0,n-min(b,d))/sum n`；合成为`S_syn=sum abs(a-b)/sum max(a,b)`。全部图和组均保留，未换输入或缩小分析集合。

| 对象 | 原／A成功数 | bright／B成功数 | dark成功数 | 主量与区间 |
|---|---:|---:|---:|---|
| 128真实TRAIN、1460 GT | 1460 | 1458 | 1460 | 损失2／1460＝0.136986%；95%簇区间[0,0.395849%] |
| 30个几何配对、280有效owner | 278 | 278 | 不适用 | 配对成功数差异0／278；经验簇区间[0,0] |

真实损失来自1493和771两图，各1例，均属声明group `1020s1120s2003-2`；不是跨13个来源均观察到失效。补充核对表明合成逐owner成功／失败状态也没有变化，不只是正负抵消后的总数相等；这是对原计数的核查，不替换冻结主量。

## 为什么不能把计数稳定写成一切都稳定

这次使用score>0.001的正常最终输出，诊断的是已有输出中可同时匹配的Mask能力；既不惩罚低质量额外候选，也不检验分数排序。真实原图1460／1460已触顶，IoU50也较粗，因此这组结果不能证明AP、细轮廓或置信度可靠。TRAIN的已见来源更不能充当独立泛化。

8张固定预览已逐张打开，图像确有色调或照明变化。下排为便于查看只显示score>=0.25，颜色按每图预测顺序，不表示跨图稳定ID；看到颜色或显示Mask变化不能直接解释成身份错归。见[视觉回执](VISUAL_REVIEW.json)。

为解释这种显示与主量的区别，仅复用已保存输出做了**事后、非预注册**说明：在预览阈值0.25下，真实原图／bright／dark成功数为1454／1446／1453，逐图最坏损失13／1454；合成A／B为239／243，逐对绝对差异4／243。见[EXPLORATORY_DISPLAY_THRESHOLD.json](EXPLORATORY_DISPLAY_THRESHOLD.json)。没有扫描阈值或新跑前向，也不据此重开5%／10% Gate；它混合了Mask和分数变化，不能证明校准机制或方法效用。

## 执行、独立核验与资源

- 工程1158／`RUN_PIG_APPEARANCE_ENGINEERING_20261103_5c3781be30ed`：8次前向，六项竞争／增广路／空集等构造检查、真实输入、恒等与成对身份资格通过；只用于正式准入，不作科学阳性。
- 正式1159／`RUN_PIG_APPEARANCE_FORMAL_20261103_b34157c23fcd`：444次正常图像前向＋24次重复控制前向；12组控制scores／boxes误差0，Mask及raw身份逐值一致。两阶段合计476前向、0优化、0新渲染、0 dev／test读取。
- 两作业均COMPLETED／0:0，分别11／45分配秒，合计56 GPU秒＝0.015556GPUh，224 CPU-core秒＝0.062222h，峰值1GPU，低于本新Study帽；不计入旧Study。
- 54项原Run产物和158个原标签NPZ全部SHA回传核验。完整10685条正常检测、444个IoU矩阵由原标签和完整640 RLE在本地用独立位运算重算，最大误差0；再用残量网络BFS最大流复算匹配，主量和两侧bootstrap区间均一致，见[RESULT_VERIFIED.json](RESULT_VERIFIED.json)。本次确实重算原始数值，不只核文件哈希。

## 留存位置更新（用户2026-11-03要求）

本地当前仅保留本报告、设计／协议说明及精简Study／Run／核验回执。完整预测、IoU矩阵、逐例数据、原标签、预览、日志、执行代码和完整冻结源均保存在服务器，见[REMOTE_STORAGE.json](REMOTE_STORAGE.json)。清理前已逐SHA核验三个相关原始Run的70项服务器产物，并将本地待移除文件另行归档核验；该处置不改变指标、Gate或原始Run。上述本地独立复核是清理前已经完成的历史事实，数据副本现已移除，后续核验在服务器进行。

[冻结清单](FROZEN.json)、[工程回传摘要](ENGINEERING_RETURN.json)和[正式回传摘要](FORMAL_RETURN.json)保留为精简追溯资料；其中引用的完整文件在服务器原Run和本地副本归档内，不表示本地仍有相应数据。此前回传曾因Windows命令长度限制在传输准备阶段失败，改为标准输入传脚本后完成；原模型计算未重跑，科学代码和已完成Run未修改。

结果回到[具体Idea](../../project/ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md)、[条件可靠性主线](../../paper/manuscripts/pig_condition_reliability/README.md)、[迁移缺口](../../project/outlines/OUTLINE_PIG_TRANSFER_GAP.md)及[中心纲要](../../project/outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。
