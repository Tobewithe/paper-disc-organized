# paper-disc-organized

本项目保存纲要、Idea、实验设计与报告、精简Study／Run记录，以及文献、写作和维护资料。当前任务与执行范围从工作台指南规定的入口读取。

## 读取入口

先读取[AGENTS.md](AGENTS.md)，了解科研角色、算力资源和两份指南的读取要求。接续时先看[当前安排](project/ACTIVE_PLAN.md)，确认目标与执行范围；首次开展工作或规范变化时完整读取[工作台指南](project/WORKBENCH_GUIDE.md)及任务相关的[通用规范](project/科研项目通用规范/通用规范.md)，同一任务复用已读且未变化的内容。

寻找知识从[知识入口](project/research_notes/README.md)进入，全部知识统一保存在[项目Skill主份](project/skills/README.md)，用户目录保留加载副本。研究Idea和纲要保留研究组织身份，不纳入知识迁移；项目笔记保存阅读过程、应用与证据。当前猪实例分割的问题空间见[中心纲要](project/outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。

## 资料位置

| 位置 | 内容 |
|---|---|
| `project/outlines/` | 问题纲要、已有认识与开放问题 |
| `project/ideas/` | 具体主张、依据、探索进度与判断 |
| `project/research_lines.json` | 主线身份、入口和Idea／Study关联 |
| `project/ACTIVE_PLAN.md` | 当前目标、状态、暂停范围和下一动作 |
| `experiments/<study>/` | Study登记、协议、报告、交接及远端证据定位 |
| `paper/manuscripts/<line>/` | 已登记主线的说明和研究材料 |
| `project/skills/` | 全部知识的Skill主份、来源、配套文件、覆盖与恢复清单 |
| `project/research_notes/` | [知识入口](project/research_notes/README.md)、项目应用、跨实验笔记与Idea演变 |
| `project/research_notes/literature/` | 按主题或阅读批次保存的文献来源、阅读范围与当时分析 |
| `project/research_notes/sources/` | 当前文档引用的固定版本来源与SHA清单 |
| `project/research_notes/restart/` | 有必要保留的历史安排与接续快照，按对应时点解释 |
| `project/research_notes/maintenance/` | 目录整理、迁移、清理和维护回执 |
| `paper/manuscripts/writing/` | 按documents、drafts、presentations保存的写作与展示材料 |

完整实验数据与执行产物按服务器留存约定管理，本地记录其位置、版本和核验结果。Study／Run使用稳定身份，资料移动只更新位置与引用，不重新登记实验或改变测量结果。

## 本地文件检查

运行`python project/check_active_plan.py`，只读核对AGENTS.md到两份指南的索引、工作台指南到ACTIVE_PLAN.md的入口和当前文件链接。Skill文件与加载副本的检查见[维护说明](project/skills/README.md)。检查不判断实验是否完成，不修改工作台索引，也不启动远端任务。

2026-11-04目录整理的位置映射、删除范围及核验结果见[目录清理回执](project/research_notes/maintenance/directory_cleanup_20261104/SUMMARY.md)；当时实验目录以外文件的保存位置整理和服务器归档见[非实验文件规范整理回执](project/research_notes/maintenance/nonrun_file_compliance_20261104T093520Z/SUMMARY.md)。后续知识与入口修订不由这些历史回执覆盖。
