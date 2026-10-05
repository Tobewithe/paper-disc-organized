# paper-disc：分割失败机制研究与实验进展

当前研究从 YOLO26m-seg 的真实失败实例出发，追查检测候选、原型、系数读出、监督与掩码形成的关系，寻找能在正常预测中兑现的改进。**已经建立若干可恢复空间和条件可达性证据；尚未建立主线上的稳定实用方法收益或唯一失败根因。**

本仓库为私有科研文档快照，更新至 **2026-10-04**。本轮只整理和发布已有证据，没有重新训练。7月的猪跛足识别、DST-VideoMAE 与开题材料保留为历史阶段，不作为当前分割研究的执行计划。

## 阅读入口

1. [当前研究状态](CURRENT_STATUS.md)：主要成果、近期结果、限制和待决问题。
2. [全部实验索引](EXPERIMENT_INDEX.md)：{{STUDIES}} 个已登记 Study；数量包含历史导入、诊断、方法、失败尝试与冒烟测试，不等于独立结论数。
3. [机器可读目录](catalog.json)：稳定 Study ID、登记状态、报告和 Run。
4. [证据来源清单](SOURCE_MANIFEST.json)：源报告相对路径与版本；[快照信息](PUBLICATION.json)记录导出时间和覆盖数量。

`experiments/` 保存逐项概述，`reports/` 保存报告及小型统计快照。大规模逐实例产物、数据、权重、连接信息和训练日志保留在研究主项目。

## 最新重点

- [候选关系读取](experiments/candidate_relation_readout_20261004.md)：四组训练及评价完成，短预算下没有明显新增收益；明确保留训练充分性与规模外推的未知项。
- [位置×目标互补判别](experiments/position_objective_complement_20261004.md)：官方TAL位置上的修正为负，几何匹配位置上的修正明显为正，但跨位置迁移接近零；位置因素得到支持，尚未形成GT-free部署方法。
- [候选响应共识](experiments/candidate_response_consensus_20261004.md)：零训练组合没有超过原模型或普通分数加权，停止当前配置。
- [监督—评价域审计](experiments/supervision_output_domain_audit_20261003.md)：标签冲突不能解释全部退化。
- [证据通路使用审计](experiments/evidence_path_usage_20261004.md)：已完成；额外路径有微小非零贡献，但当前检查点的空间修正未达到预设实用量级。

## 怎样使用这些记录

可让阅读者先读取 `CURRENT_STATUS.md` 和 `EXPERIMENT_INDEX.md`，再根据问题进入具体报告。提出新实验前先查历史；不要把 raw 候选、官方 TAL 正样本、几何匹配候选和最终推理输出混为一谈。候选 IoU、Mask75 修复与 COCO AP 是不同指标。

GT 可用于训练和诊断，但测试时依靠 GT 求出的 oracle 不能算部署收益。短程阴性结果限定于实现、对象和预算；正信号也需要独立确认。历史报告中的措辞保留原来源范围，当前解释及修正记录优先。

本次更新对象是 **Tobewithe/paper-disc**；原 `paper-disc-summary` 为另一份较早快照。科研文件与语义映射以主项目和工作台为准，GitHub 是阅读与分享副本。后续发布应使用本仓库对应配置，保留历史结果，不上传原始数据或凭据。
