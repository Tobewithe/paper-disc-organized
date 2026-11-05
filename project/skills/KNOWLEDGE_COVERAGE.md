# 本地知识覆盖记录

本轮按用户要求检查既有知识的遗漏，将查出的理论、方法、经验与反例补入Skill。研究Idea、纲要、目标、候选方案、预算及下一步安排保持研究组织身份；`project/ideas/`和`project/outlines/`不纳入迁移。

## 范围和结果

盘点包含525份本地作者材料或来源台账，初始合计5,971,367字节，范围为研究笔记、相关阅读记录、部分主线方法/理论材料及实验根目录的方法综合。原PDF、原始全文、检索响应、Run大产物与维护归档不作知识正文搬运。逐文件范围、初始指纹、分类理由与去向见[覆盖清单](KNOWLEDGE_COVERAGE.json)。

本轮建立59份明确来源身份，分别保留本地阅读/综合、数学审查、原实证报告和来源台账的角色。来源在10个新增或扩充的Skill包内由`CURATION_SOURCES.json`记录；原已归档的辅助学习和TIDE正文直接复用。当前共13个Skill，知识入口见[主题导航](../research_notes/README.md)。

| 原遗漏范围 | 知识维护位置 |
|---|---|
| 科研固着、反思、反馈及真实失败链反例 | [科研推理](model-method-selection/references/research-reasoning.md) |
| 原型分配、局部对应和不变性 | [表示适应](feature-adaptation-methods/references/correspondence-and-prototypes.md) |
| 共同坐标、条件融合与功能依赖 | [融合知识](feature-fusion-methods/references/coordinates-and-dependence.md) |
| unknown、owner、监督支持及风险代理 | [标签与损失](segmentation-loss-design/references/label-support-and-risk.md) |
| BN、EMA、梯度路径和函数保持经验 | [训练状态](pretrained-transfer-methods/references/training-state-and-transfer.md) |
| 对象表示、能量责任、基底数学和读出经验 | [数学](yoloseg-method-design/references/object-representation-math.md)、[实证与架构](yoloseg-method-design/references/readout-and-project-lessons.md) |
| 评价单位、有限oracle、条件成本和干预反例 | [评价与决策](vision-evaluation-methods/references/evaluation-and-decision.md) |
| 遮挡、amodal、因果记忆与视频身份 | [遮挡Skill](occlusion-video-identity/SKILL.md) |
| 合成分布、成像频谱、几何标签、表面与接触 | [合成Skill](synthetic-scene-methods/SKILL.md) |
| RGB定位、部位owner、观察与模型证据 | [视觉定位Skill](visual-grounding-and-evidence/SKILL.md) |

## 怎样读覆盖状态

`curated_knowledge_sections`表示已有实质提炼及明确正文目的地，具体只读到哪些部分以包内来源范围为准。`covered_by_existing_skill`表示之前已保存、此次复用。来源台账只支持出处/阅读身份，不自己成为新的科学结论。

其它条目按研究组织、历史实证、来源获取或写作派生材料保留。`related_knowledge_entry`仅提供主题路由，明确不代表该文件全部内容已抽取。旧方案、历史“下一步”和论文组织不会因分类变成知识或执行授权。原报告和阅读快照保持原时点，后续知识演进统一由Skill维护。

这次没有对全部历史逐例/逐Run材料和各版稿件做逐行语义复审，因此525不是全文阅读数，也不构成全仓库零遗漏声明。已知未读候选继续保留为检索线索；此前仅核摘要、部分章节或API的来源不升级为全文掌握。经典多任务共享归纳偏置、任务相关性与泛化条件仍需系统补读。

本轮没有新增外部检索、原论文全文审读、渲染、模型前向或优化。所有文献自报、数学例与项目结果保留原证据条件。文件哈希、链接、版本和副本一致性用于核对留存，不评价知识是否已掌握或方法是否有效。
