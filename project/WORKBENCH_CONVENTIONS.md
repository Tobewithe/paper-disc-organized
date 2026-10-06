# 工作台文件约定

本文件只规定文件归属、身份和引用；科学问题、实验协议和结论留在实验报告/论文记录中。工作台从磁盘自动发现文件，`project/structure.json` 是生成索引，不手工维护。

## 结构与身份

- 新实验（含诊断、方法、消融、复现）放 `experiments/<稳定目录名>/`，目录内有 `study.json`。
- 每次训练、评估、seed 或重试放所属实验的 `runs/<Run ID>/run.json`；Run 属于实验。
- 共用数据/权重放 `assets/`，共用代码放 `shared/`，论文放 `paper/manuscripts/<论文>/`，跨实验记录放 `project/research_notes/`。
- 实验独立于论文，可以被零篇、一篇或多篇论文引用。`project/research_lines.json` 的 `study_ids` 只是引用清单，不决定所有权。
- 已有目录、Study ID、Run ID 和历史路径保持兼容；改名不重建身份。缺失、不可访问和未验证状态保持原样。

## 写入规则

普通文件直接落盘，不逐文件登记。新 Study/Run 才需要最小 JSON；已有记录先读取再合并，避免覆盖别的任务。普通代码编辑不额外写日志。

需要科学记录时，引用原文、Run、产物版本、评价范围和限制；不要复制结论。支持/反对关系必须有明确来源，不能由目录名、机器或文件存在推断。

外部机器回传到对应实验的 `runs/` 后再进入本机扫描；保留执行状态、产物状态、回传状态和文件可用性四个维度。工作台索引可重建，不能替代源文件和运行记录。

## 研究线

研究线由 `project/research_lines.json` 和 `paper/manuscripts/<论文>/` 的实际文件发现；线程关联若已登记在 `owner_thread_id` 中只作来源信息，不决定默认研究线。未选择研究线时按项目范围工作。论文按需引用实验，实验不为论文复制。迁移盘点保存在 `_maintenance/workbench_alignment/`，仅用于迁移协调。

详细外部接入和运行格式见工作台 `workbench/research/EXTERNAL_CONTRACT.md` 与 `workbench/research/README.md`。本文件不要求 AI 启动同步、检查 UI 或填写审批表。
