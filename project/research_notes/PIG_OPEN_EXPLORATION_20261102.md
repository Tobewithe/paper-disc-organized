---
schema_version: 1
id: REVIEW_PIG_OPEN_EXPLORATION_20261102
kind: review
title: 从中心命题打开不同Idea：对主线与论文框定的纠正
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_REAL_DATA, OUTLINE_PIG_DATA, OUTLINE_PIG_MODEL, OUTLINE_PIG_TRAINING, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_EVIDENCE, OUTLINE_VIDEO_OCCLUSION]
idea_ids: [IDEA_PIG_KNOWLEDGE_FOR_INSTANCES, IDEA_PIG_ASSIGNMENT_PART_MASK, IDEA_PIG_INSTANCE_INFORMATION_USE, IDEA_PIG_OBSERVATION_INFORMATION, IDEA_PIG_COVERAGE_TRANSFER, IDEA_PIG_SYNTHETIC_COUNTERFACTUALS, IDEA_PIG_RELATION_ANNOTATION, IDEA_PIG_VISIBILITY_AND_AMBIGUITY, IDEA_PIG_ADAPTIVE_DETAIL]
supersedes: REVIEW_PIG_PAPER_NARRATIVE_20261102
coverage: declared
verdict: open_exploration_candidates_not_method_validation
author: AI依据用户本次明确纠正与已有资料推导；候选潜力未经实验验证
scope:
  action: restore_broad_central_question_and_explore_distinct_ideas
  superseded_scope: prior_task_framing_and_candidate_priority_only
  workbench_documentation_clarified: true
  new_research_line_or_study_or_run: false
  remote_access_training_evaluation_generation: false
  original_gate_budget_pause_or_test_changed: false
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 既有文献指定核读段落；提供线索，不代表新想法已获新颖性或效用确认
    relation: references
  - path: project/research_notes/sources/concept_synthesis_release_72908863d895ceae.txt
    revision: 72908863d895ceaea2354b15632013bebe473d7f3e1124a634e559190ea2a94f
    locator: 有限可控能力和标签资格，非新受控对照已执行
    relation: references
  - path: project/research_notes/sources/assignment_bridge_pause_7e86dc12ee5f9b0e.txt
    revision: 7e86dc12ee5f9b0e34995982c582e925bed004c8880982c4824879780ab065c7
    locator: 原实验暂停和未决保持，只约束所属执行
    relation: references
---

## 用户明确纠正的是什么

用户说明：“中心纲要代表一个较大的命题，主线只是其中一个切入方向，不能把主线等同于整个命题，也不要让它限制后续探索。”并要求“重点是打开思路、发现更好的想法，而不是完善论文框架、机械拆分提纲或过早收敛”，允许更改妨碍表达的工作台规范。

因此，本次取代上一轮任务理解、组织方式与总体候选排序；保留旧论文草案作为局部历史材料，实验事实、阴性、Goal、Gate和暂停不被取代。

## 暂定的大命题理解

根据[已有中心标题与资料](../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)，暂按真实猪场怎样获得准确、稳定、可信且可迁移的逐头实例分割来理解。具体应用、输入与成本权衡仍开放；不把单帧、YOLO、合成、部位标签或一项AP目标升为整个命题的固定前提。

现有SPM主线从猪体／对象知识切入，提供教训和能力；它不拥有全部问题空间。未挂线或跨线Idea可以先形成，不必完成旧主线、进入论文目录或立刻冻结协议。

## 打开的不同方向与联系

[开放探索图](../outlines/OUTLINE_PIG_HYPOTHESES.md)展开七种不同改变：实例由什么证据组成、是否获取历史观察、合成怎样帮助认识、应补充什么标注、怎样表达未知、怎样迁移与保持、哪里值得获取更多细节。

三个优先继续推敲的想法是：

- **关系型标注与片段组装：**共同设计任务需要的关系和最低成本监督，有直接猪领域归组与弱监督线索；关系信息是否冗余、人工是否便宜仍未知。
- **少量真实可见观察与可信归属：**改变可用证据，可能突破单帧歧义，并能提供训练对应；序列可用性和可靠锚点尚未核。
- **受控合成作为认识工具：**复用有限生成能力，为多种表示和迁移问题产生可区分的想法；合成发现的真实适用范围未知。

跨场景可靠性有长期价值，其余想法继续保留。这里筛选的是值得多想和寻找直接依据的潜力，不是训练准入、永久排序、统一模块方案或新运行计划。

## 规范与源文件的实际调整

根[AGENTS](../../AGENTS.md)和通用规范入口（原路径：`project/科研项目通用规范/README.md`；[服务器归档回执](maintenance/nonrun_file_compliance_20261104T093520Z/ARCHIVE_RECEIPT.json)）明确中心纲要的外延与具体Study执行边界。工作台`workbench/research/KNOWLEDGE.md`与`README.md`澄清主线是切入方向，论文是可选成果，`paper/manuscripts/`是兼容路径；未改API、数据结构或研究线登记。

总纲和开放问题页移除聚焦论文作为默认组织，数据、模型、训练、评价与证据页改为连接不同Idea。原有Idea保持稳定ID，信息利用、观察与覆盖方向解除旧局部方案限制；新增受控合成、关系标注、可见性歧义和按需细节四个候选，均未挂正式线、研究判断未知。演变轨迹和旧叙事记录显式说明本次纠正。

## 证据边界与执行状态

本轮复用已有文献与本地记录，未新增文献搜索或完成专题新颖性核验；没有新增图像审阅、人工标注、数据生成、模型训练／评价或远端访问。主要错误占比、原视频、原图细节、人工成本与新来源仍是未知，不能把推导写成已证机制。

原六臂暂停、两臂终态待核、dev未评价、test426封存均保持。具体恢复只沿[原总控](../ACTIVE_PLAN.md)和所属Study，开放Idea不继承任何旧执行许可。

修改前备份、来源与结构核验记录在[_maintenance/pig_open_ideas_20261102](maintenance/pig_open_ideas_20261102/README.md)。
