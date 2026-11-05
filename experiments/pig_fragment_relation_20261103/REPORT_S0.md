---
schema_version: 1
id: REVIEW_PIG_FRAGMENT_RELATION_S0_20261103
kind: review
title: TRAIN片段关系适用性实验：当前原标注支持不足
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_ORGANIZATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TASK, OUTLINE_PIG_MODEL, OUTLINE_PIG_REAL_DATA]
idea_ids: [IDEA_PIG_FRAGMENT_RELATION_PILOT, IDEA_PIG_INSTANCE_INFORMATION_USE, IDEA_PIG_RELATION_ANNOTATION]
coverage: declared
verdict: current_cache_support_insufficient_stop_this_entry_method_not_tested
author: AI依据真实Slurm运行、全量回传记录复核及12张固定预览审阅
scope:
  target: 当前缓存是否适合承载显式片段关系研究
  metric: 原断连、唯一监督支持数及冻结最低工程支持量
  dataset: 固定1272真实TRAIN图像／14642原160网格独立Mask
  protocol: experiments/pig_fragment_relation_20261103/PROTOCOL_S0.md
  study_id: STUDY_PIG_FRAGMENT_RELATION_S0_20261103
  run_id: RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230
  object: 既有真实TRAIN原160网格独立Mask
  new_model_or_mechanism_validation: false
  raw_mask_algorithm_independently_reexecuted: false
evidence:
  - path: experiments/pig_fragment_relation_20261103/PROTOCOL_S0.md
    revision: c96b6d9179ef3d5f42dcd0e799377363eda7625467cc574842196d468aa853c0
    locator: 真实数组解码前冻结的对象、算法、分母与停止条件
    relation: references
  - path: experiments/pig_fragment_relation_20261103/runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/SUMMARY.json
    revision: 4284d2de6a6873f14503502867bb98075a099291827a40b9224c21d23c02d47d
    locator: 全1272图／14642实例的测量与固定样例
    relation: references
  - path: experiments/pig_fragment_relation_20261103/runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/run.json
    revision: 407ebffaac8daa602d959d395913c7ca2cb823584a8a8888c1b6d5c3ddb49cff
    locator: 实际执行身份、六项构造检查、环境、源码与Idea冻结版本
    relation: references
limitations:
  - 缓存连通性不能替代物理可见性，也没有测量正常模型错误。
  - group是原manifest字段，不等于独立农场、独立个体或新来源。
  - 本次没有模型训练、人工成本测量、dev或test读取。
---

## 为什么做、怎样检验

[片段关系Idea](../../project/ideas/IDEA_PIG_FRAGMENT_RELATION_PILOT.md)希望利用同体断片及其关系，但目标数据是否能承载该问题尚未知。先按[冻结协议](PROTOCOL_S0.md)分析所有1272张真实TRAIN、14642个原独立Mask，避免在对象支持不足时就训练。

原Mask先用8邻域计非平凡分量；面积至少16格点且至少实例面积2%。至少两个原分量各有16格点不与其他原Mask重叠时，计为可靠原断连支持。另报剔除重叠后的断连，它仅代表唯一监督区域，不能替换原断片或称物理可见GT。

## 实际结果

| 读数 | 实际结果 | 能说明什么 |
|---|---|---|
| 全量输入 | 1272图、14642实例，13个声明group | 未缩小或替换集合；所有标签原SHA、形状、数值和实例数校验通过 |
| 原Mask非平凡断连 | 15实例、15图 | 只反映原160网格标签几何 |
| 可靠原断连支持 | 14实例、14图、3个声明group；占全部实例0.095615% | 低于预定128实例／32图的最低工程支持量，group条件达到 |
| 唯一监督区域断连 | 164实例 | 其中150个原Mask不满足原断连定义，断连由剔除重叠诱发；不能悄悄改作主对象 |
| 原独立Mask累计像素中的重叠 | 49278／2831949 | 是按实例累计的重叠像素，不是物理遮挡比例 |

可靠原断连分布为6／7／1，来自三个声明group；图像或实例出现数不等于独立动物。这个全量描述也不是正常分割错误分母。

固定8张原断连、4张无原断连对照已逐张打开审阅。样例能看到身体／设施相互遮挡，部分原Mask跨设施保持连通，例如对照469与原断连642。原预览现存服务器`runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/previews/`，定位见[REMOTE_STORAGE.json](REMOTE_STORAGE.json)。上述有限审阅不能给全部GT重新定性，未逐一给全部14个支持作物理可见性裁决。

## 研究决策及其边界

**结束“直接用当前原160缓存中的显式断片训练关系方法”这一具体入口，当前不启动其方法训练。** 首试Idea记为已结束／支持不足，主线暂停于输入支持不足；广义实例信息利用Idea仍判断未知，关系标注Idea仍为候选。

这次降低的是当前数据入口的可行性，不是否定同体归属问题、关系监督或文献方法。图像中的遮挡可以很多，而原标注仍连通；不能据14个原标签断片推出真实视觉难例稀少，更不能把150个诱发断片冒充已确认可见样本来跨过门槛。

后续若有高分辨率原标注、明确可见标签、可核的序列或真实归属错误证据，可以重新审查问题与输入。需要说明新增证据改变了什么；本次不为保住方案追加人工切片、降阈值、扩量或另加loss。当前没有已具备条件的方法训练后继。

## 执行与交付

Slurm **1157**、`RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230`，真实进程开始于UTC 13:01:19，正常完成。六项构造检查通过；Slurm为COMPLETED／0:0，分配15秒×2CPU＝30 CPU-core秒、0 GPU秒，Study帽2 CPU-core小时内。优化步数0，未加载模型或权重。

## 留存位置更新（用户2026-11-03要求）

本地当前仅保留本报告、协议说明与精简Study／Run／核验回执。逐实例／逐图记录、原预览、日志、执行代码及完整冻结文件均在服务器；原70项相关Run产物及本地副本归档已逐SHA核验后，清理了本地数据副本。服务器位置见[REMOTE_STORAGE.json](REMOTE_STORAGE.json)，清理不改变原Run、Gate或科学结论。

[冻结源清单](FROZEN.json)及[原回传摘要](runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/transfer.json)保留追溯信息；其中数据文件现按远端位置读取，不表示本地仍存在。清理前已复核逐实例14642行、逐图1272行及12张固定预览选择，见[RESULT_VERIFIED.json](RESULT_VERIFIED.json)；该历史复核依据回传充分统计，没有第二次解码原始Mask冒充独立算法复现。后续数据核验在服务器执行。

结果回到[实例组织问题](../../project/outlines/OUTLINE_PIG_ORGANIZATION.md)、[片段缺口](../../project/outlines/OUTLINE_PIG_MODEL.md)和[中心纲要](../../project/outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。
