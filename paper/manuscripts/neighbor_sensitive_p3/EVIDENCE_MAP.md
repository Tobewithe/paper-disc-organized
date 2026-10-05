# 论文主张、实现与结果对应表

写稿日期：2026-09-15。本文件是正文的复核附件，不增加实验规范或停止门槛。中文/英文稿使用相同原始表格。这里只记录已经存在的结果，撰写与图表生成没有新增模型实验。

## 1. 正文结构与原始证据

以下路径以项目根目录 `C:/Dpan/codexproject/paper-disc-organized` 为起点；链接直接指向对应文件。

| 论文内容 | 可核实事实 | 直接来源 |
| --- | --- | --- |
| 表 1：全量候选阶段 | 36,335 普通 GT；3,699 未匹配；原始几何 1,459、其中 small 1,429 | [summary.csv](../../../experiments/no_final_slot_lineage_20260914/summary.csv)、[逐 GT](../../../experiments/no_final_slot_lineage_20260914/per_gt.csv)、[实现](../../../diagnostics/no_final_slot_lineage_20260914/no_slot_lineage.py) |
| 失败/成功选样 | 384＋384，不同图像、类别相同、面积近邻匹配；先固定选样再测干预 | [selection.csv](../../../experiments/small_raw_geometry_origin_20260914/selection.csv)、[实现](../../../diagnostics/small_raw_geometry_origin_20260914/small_geometry_origin.py) |
| 表型与局部位置 | 得分最高局部 P3 位置的 IoU、中心误差、类别得分；与 oracle IoU 分开 | [summary.csv](../../../experiments/small_raw_geometry_origin_20260914/summary.csv)、[paired_effects.csv](../../../experiments/small_raw_geometry_origin_20260914/paired_effects.csv) |
| 初始联合编辑 | 对比度、近邻修补、联合编辑；联合 +10.219 IoU 点，146 个 raw Box50 恢复 | [结果](../../../experiments/small_object_causal_interventions_20260914/summary.csv)、[图像编辑实现](../../../diagnostics/small_object_causal_interventions_20260914/causal_image_probe.py) |
| 表 2：单邻居控制 | 302 个失败，邻居减等形背景 +13.864；blur +10.438；299 个远端控制 +13.493 | [paired_contrasts.csv](../../../experiments/neighbor_specific_recovery_20260915/paired_contrasts.csv)、[实现](../../../diagnostics/neighbor_specific_recovery_20260915/neighbor_specific_recovery.py) |
| 恢复定义及成功控制 | 105/302 满足操作性严格定义；214 对的交互 +14.604；成功对象也有 Box50 损失 | [实例判定](../../../experiments/neighbor_specific_recovery_20260915/neighbor_specific_instances.csv)、[阈值转移](../../../experiments/neighbor_specific_recovery_20260915/threshold_transitions.csv)、[交互](../../../experiments/neighbor_specific_recovery_20260915/matched_failure_control_interactions.csv) |
| 表 3：P3/P4/P5 替换 | 联合编辑供体；局部 P3 +10.049；完整 P4 +1.969；完整 P5 +0.445 | [feature_localization_summary.csv](../../../experiments/small_object_causal_interventions_20260914/feature_localization_summary.csv)、[实现](../../../diagnostics/small_object_causal_interventions_20260914/feature_causal_localization.py) |
| 方法公式 | 选样、RGB 编辑、教师位置、位置级 gate、GT SmoothL1＋类别 BCE、batch 缩放 | [实际训练实现](../../../experiments/counterfactual_p3_distillation_20260914/scripts/train_counterfactual_p3_selective.py)、[远端实现副本](../../../experiments/counterfactual_p3_distillation_20260914/remote/train_counterfactual_p3_selective.py) |
| 表 4：三种子实例效果 | 239 failure、199 control、135 完整对；按实例平均三种子再 bootstrap | [聚合 CSV](../../../experiments/counterfactual_p3_distillation_20260914/rich_across_seed_target_effects.csv)、[评价实现](../../../experiments/counterfactual_p3_distillation_20260914/scripts/eval_targeted_rich.py)、[聚合实现](../../../experiments/counterfactual_p3_distillation_20260914/scripts/aggregate_target_effects_across_seeds.py) |
| 表 5：官方任务指标 | best.pt 保存预测；1,576 图；official COCOeval；Mask AP 均值差 −0.011 | [逐模型官方指标](../../../experiments/counterfactual_p3_distillation_20260914/rich_eval_saved/official_metrics.csv)、[核对报告](../../../experiments/counterfactual_p3_distillation_20260914/RICH_EVALUATION_RESULTS.md)、[重评实现](../../../experiments/counterfactual_p3_distillation_20260914/scripts/evaluate_saved_predictions.py) |
| 固定最后轮与配方探索 | 固定第 3 轮是 native 验证；broad 与 selective 同时改变 λ | [三种子报告](../../../experiments/counterfactual_p3_distillation_20260914/THREE_SEED_RESULTS.md)、[一轮配方报告](../../../experiments/counterfactual_p3_distillation_20260914/REPORT.md) |
| 续训与完整数据 | S101 seed 1 中断恢复；S105 全量转换审计；完整数据 seed 0 一轮已配对完成，定位误差改善、整体 AP 持平 | [事实记录 S101/S105](../../../project/facts/RESEARCH_FACTS.md)、[全量完成结果与评分修正](../../../experiments/counterfactual_p3_fullcoco_20260914/RESULTS_20260915.md) |

2026-09-15 补充：全量 saved predictions 的原 COCO 评分因类别 ID 与评估图片范围错误而失效，已在独立 Run 重新评分。完整结果、输入版本、限制与修正过程集中保存在上述全量完成报告。此前 pilot 的尺寸分层曾将 bbox 面积用于 segm 检测面积，投稿前应独立重评分；当前不把其 Mask AP_S/AP_M/AP_L 当作最终已核实结果。正文原有五表仍为 pilot，未被全量数字覆盖。

## 2. 写稿时必须区分的口径

| 项目 | 本文采用的准确表述 | 不能替换成 |
| --- | --- | --- |
| 3,699 的分母 | 指定一对多流程中 Box50 最终未匹配 GT | 全部 Mask75 失败，或全部 COCO 错误 |
| 原始最佳框 | 不限类别、用 GT 选 IoU 最大值的几何可用性 | 实际无 GT 检出率、AP 上限或独立候选数量 |
| 最终掩码 IoU | GT 辅助选择同类最佳框，再评价它关联的正常裁切掩码 | 官方一对一匹配的掩码召回率 |
| 训练后 center_error | 全部层级最佳原始框的中心误差 | 已证明是 P3 框的中心误差；单邻居实验另有 P3 专用指标 |
| 邻居实验保持目标不变 | 原始分辨率目标像素不变 | 缩放后所有目标输入或目标特征完全不变 |
| 等形背景 | 普通非 crowd 实例并集以外；形状面积匹配 | 到目标距离匹配，或确认纯背景、无 crowd |
| P3 移植 | 联合编辑响应可经 P3 头部输入传递 | 找到上游唯一根因，或已经证明是融合错误 |
| 单邻居与联合编辑 | 不同编辑协议的互补证据 | 同一个完整因子实验，或单邻居响应已完成全层定位 |
| 训练 gate | 所选位置的 student<.5、teacher≥.5、差≥.1 | 原图所有候选都失败；同一教师因编辑而跨阈值 |
| 教师作用 | 选择位置；GT 是框监督目标 | 用教师框替代真实 GT |
| nomination 的 0.02 得分 | 有界辅助贡献，可改变 IoU 差小于 .02 的选择 | 只有 IoU 完全相同才参与的严格 tie-breaker |
| 训练视图 | 近邻填色＋RGB 目标增强；可能选择无邻居目标 | 只删除单个邻居，或只对105个严格恢复类训练 |
| 三种子区间 | 先实例内平均种子差，再对实例/配对重采样 | 对所有可能训练随机性的显著性证明 |
| Mask AP 持平 | 目前三个短程模型的均值近似中性 | 已证明完全无代价或统计非劣效 |
| 全数据一轮 | 完整图像数量的一轮筛查 | 已充分收敛、完整训练预算确认 |

`rich_eval_saved/COMPLETE.json` 的历史说明曾使用 “final epoch predictions”，与后续训练日志核对及 `RICH_EVALUATION_RESULTS.md` 中的 `best.pt` 来源说明不同。本稿采用已核对的保存预测口径，并分别报告固定 epoch 的 native 结果；不把二者混成同一个 AP 比较。

S101 记录 seed 1 在方法第 2 轮中断后恢复。当前 `_setup_train()` 复制当时的学生创建教师，教师没有独立持久化恢复。因此该次续训具有教师重建差异。正文披露这一实现限制；没有修改现存运行，也没有把三个 pilot 种子宣称为最终无偏确认。

## 3. 文献核验与本文的区别

本轮以作者 arXiv 记录、会议原文或出版元数据核对题名、作者/年份和相关论点，并读取与本文相关的摘要/原文内容；不是穷尽式查新，也不代表已经逐段复核所有附录。不能据此写“首次”或“没有已有方法”。

| 文献 | 一手来源 | 本文使用范围 |
| --- | --- | --- |
| COCO | [作者稿](https://arxiv.org/abs/1405.0312) | 数据集、实例评价背景 |
| FPN | [作者稿](https://arxiv.org/abs/1612.03144) | 多尺度特征背景，不把“小目标依赖 P3”本身当新发现 |
| YOLACT | [作者稿](https://arxiv.org/abs/1904.02689) | 共享原型＋实例系数的既有范式 |
| TIDE | [作者稿](https://arxiv.org/abs/2008.08115) | 输出错误/AP 分析与本文候选生命周期分析的区别 |
| Shetty context | [作者稿](https://arxiv.org/abs/1812.06707)、[作者海报](https://datasets.d2.mpi-inf.mpg.de/rakshith/notusingcar/poster.pdf) | 对象移除诊断上下文、编辑用于学习早已有之 |
| Repulsion Loss | [作者稿](https://arxiv.org/abs/1711.07752) | 拥挤定位的目标吸引与邻居排斥 |
| LAD | [作者稿](https://arxiv.org/abs/2108.10520)、[WACV 原文](https://openaccess.thecvf.com/content/WACV2022/papers/Nguyen_Improving_Object_Detection_by_Label_Assignment_Distillation_WACV_2022_paper.pdf) | 教师引导位置/分配并非新概念；需要原图教师控制 |
| LD | [作者稿](https://arxiv.org/abs/2102.12252) | 定位知识迁移先例；当前 GT 回归不应混称教师分布蒸馏 |
| MetaOD | [会议原文](https://conferences.computer.org/ase/pdfs/ASE2020-4sGQAuWfliLhpf5VK4ZM4u/676800b053/676800b053.pdf)、[作者机构元数据](https://researchportal.hkust.edu.hk/en/publications/metamorphic-object-insertion-for-testing-object-detection-systems/) | 对象编辑揭露检测失败及用于改进的既有路线 |

## 4. 后续结果写入位置

不把未运行对照填成负面或正面结果。现稿已形成完整论述，以下是增强投稿证据的具体增量，不是新的强制规范。

| 已完成待接入 / 尚缺的结果 | 回答的问题 | 正文接入位置 |
| --- | --- | --- |
| 全量配对：已完成 | 全数据一轮的中心/尺度误差改善；正确官方 Mask AP +0.008、Box AP +0.052 点；小目标 Mask AP/AR 点估计下降 | 5.3 新增独立 full-data 表，不覆盖 pilot；详见完成报告 |
| 原图教师，同权重/同预算 | 编辑视图是否比普通教师选择更有效 | 5.4 方法组件表 |
| 无 gate，同 λ；常规 GT 难例监督 | gate 和特殊位置选择是否有额外价值 | 5.4 方法组件表 |
| 新的未参与研发队列及充分预算 | 可迁移性与稳定性 | 5.1 列新队列、5.2/5.3 分别列结果 |
| 空间距离匹配的局部编辑控制 | 近邻语义还是局部纹理/边界机制 | 3.4 扩充而非改写现有对照定义 |

## 5. 重建图表

在 PowerShell 中执行：

```powershell
$env:PYTHONUTF8='1'
conda run -n pytorch --no-capture-output python 'C:/Dpan/codexproject/paper-disc-organized/paper/manuscripts/neighbor_sensitive_p3/build_paper_assets.py'
```

脚本只读取既有 CSV，将英文和中文正文中的五个生成区块更新为同源表格，并输出三幅 PNG/PDF/SVG。表格独立副本及来源清单位于 `tables/`。不需要 GPU，不训练或改写模型，不重跑统计。Windows 上使用 `conda run` 激活环境的 DLL 搜索路径；本机直接调用环境内 Python 虽能 import matplotlib，但绘制 patch 时会出现 DLL 异常，激活环境后正常。
