# 科研任务最小约定

本目录是工作台映射的科研文件根目录：`C:\Dpan\codexproject\paper-disc-organized`。用户与 AI 在 Codex 中推进研究；AI 负责把文件写入正确目录，工作台自动扫描。不要为了入库填写表单、手动改索引或检查网页。

## 只记住这六条

1. 普通文件直接写入：实验脚本放实验目录的 `scripts/`，报告放实验目录，执行产物放对应 `runs/<Run ID>/`；共享数据/权重放 `assets/`，共享代码放 `shared/`，论文材料放 `paper/manuscripts/<论文>/`。
2. 新实验在 `experiments/<稳定目录名>/` 新建 `study.json`；诊断、方法、消融和复现都属于实验，用 `kind` 区分。已有目录和 ID 继续沿用。
3. 新执行在所属实验下新建 `runs/<稳定 Run ID>/run.json`。不同 seed、重试和独立评估分别建 Run；Run 不能脱离实验单独挂在项目根目录。
4. 论文关联是可选、多对多的。没有论文关联不算缺失；不要为论文复制实验或创建占位论文。
5. 只写已知事实。缺失、未回传、未验证的信息保持未知；普通代码编辑不额外写研究日志。
6. 用户形成决定、运行前后或解释结果时，补最小记录：问题/选择、运行与产物、观察、范围限制、下一步。结论引用原文和版本，不复制出第二份结论。

最小结构：

```text
experiments/<实验>/
  study.json
  REPORT.md                 # 有报告时才建
  scripts/
  runs/<Run ID>/
    run.json
    ...本次结果与日志
```

`study.json` 最小示例：`{"study_id":"STUDY_<UUID>","title":"实验名称","kind":"diagnostic"}`。
`run.json` 最小示例：`{"run_id":"RUN_<UUID>","started_at":"2026-10-06T12:00:00+08:00"}`。时间未知就省略，不用文件时间代替。

## 何时读取额外规范

- 只做文件整理、报告撰写或普通分析：按上面六条即可。
- 设计训练、消融或正式评价：读取 `project/guidelines/EXPERIMENT_SPEC_STANDARD.md` 的相关章节，再读取 `project/WORKBENCH_CONVENTIONS.md` 的存储规则。
- 需要干预术语或研究计划时，按问题读取 `INTERVENTION_TERMINOLOGY.md` 或 `RESEARCH_PLAYBOOK_20261004.md`；不要默认通读整个 `project/guidelines/`。
- 研究线不在本文件中固定。需要论文上下文时，读取 `project/research_lines.json`、对应 `paper/manuscripts/<论文>/` 文件夹和当前线程已明确的研究目录；未选择研究线时按项目范围工作。论文只是引用实验的视角。

## 执行与工作台边界

实际训练/评估可用工作台 `workbench/research/runner.py`，每个 Run 保留实际命令、环境、输入、状态和产物。工作台数据库、`.research/` 与 `project/structure.json` 自动维护；不要直接编辑。

只有在用户要求排查索引、保存修订或验证记录时，才使用 `researchctl record/source/sync/validate`。外部机器回传 Run 时保留 `transfer.json` 和 `manifest.sha256`；未回传不等于失败。

更完整的接口细节见工作台 [科研记忆说明](C:/Dpan/codexproject/Aggregation%20Workbench/workbench/research/README.md) 与 [外部接入合同](C:/Dpan/codexproject/Aggregation%20Workbench/workbench/research/EXTERNAL_CONTRACT.md)。当前用户指示优先于本文件；历史报告和旧规范只在其原适用范围内有效。
