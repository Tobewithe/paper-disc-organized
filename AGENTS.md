# 科研文件落盘：最小约定

工作根目录：`C:\Dpan\codexproject\paper-disc-organized`。默认 PowerShell 7.6.4；深度学习使用 conda `pytorch` 环境。外部接入合同见 `C:\Dpan\codexproject\Aggregation Workbench\workbench\research\EXTERNAL_CONTRACT.md`。

AI 按以下规则保存文件，工作台服务运行时自动扫描。日常无需调用 `researchctl`、同步或校验索引、检查工作台状态；其他说明中旧的手动维护要求不作为文件入库前提。

1. **普通文件：直接保存。** 实验专用脚本放在该实验的 `scripts/`，报告放在实验目录，每次执行的产物放在该实验的 `runs/<运行>/`。共享数据/权重放 `assets/`，共享代码放 `shared/`，论文文档放 `paper/manuscripts/<论文>/`。无需逐文件登记。
2. **新实验：加一个 `study.json`。** 统一放在 `experiments/<实验>/`，包括诊断、方法、消融和复现。保存稳定 ID、名称和类型，示例见下方；已有实验复用原目录和 ID。论文关联可为空，也可引用到多篇论文。
3. **新 Run：加一个 `run.json`。** 放在所属实验的 `runs/<运行>/`，保存稳定 Run ID；工作台由所在实验目录识别归属，无需重复填写 `study_id`。已有记录含 `study_id` 时保持与所属实验一致。不同 seed、重试和独立评估各用一个 Run，同次执行回传沿用原 ID。
4. **已知信息如实保存。** 需要按开始时间排序时，`run.json` 加 `started_at`，填写带时区的实际时间，如 `2026-09-15T14:30:00+08:00`；未知则省略，不用复制时间代替。状态、报告入口、命令、日志、配置和快照有现成记录就保留，不为入库补造信息、补写完整审查或重跑实验。已有 JSON 字段保留，修改前读取当前内容，避免覆盖其他任务的改动。
5. **落盘即完成。** 简要返回目录/结果入口即可；不检查网页或索引。整理旧文件采用复制，保留原文件；远端回传到对应实验的 `runs/` 后才进入本机扫描，未回传如实说明。原项目 `C:\Dpan\codexproject\paper-disc` 不改动，数据库、`.research/` 和自动生成的 `project/structure.json` 由工作台维护。

Run 的执行状态、产物状态、回传状态和文件可用性分别记录；远端 Run 回传时附 `transfer.json` 与 `manifest.sha256`，不以“尚未回传”代替“执行失败”。

```text
experiments/<实验>/
  study.json
  REPORT.md                 有报告时保存
  scripts/                  实验专用脚本
  runs/<运行>/
    run.json
    ...本次结果、日志等文件
```

新 `study.json` 的简短示例（占位 ID 换成新生成的唯一 ID，类型按实际填写）：

```json
{"study_id":"STUDY_<UUID>","title":"实验名称","kind":"diagnostic"}
```

新 `run.json` 的最小示例（占位 ID 换成新生成的唯一 ID）：

```json
{"run_id":"RUN_<UUID>"}
```

已有实验追加普通文件不重写登记；只有新实验、新 Run 才需要相应的新 JSON。最小记录足以建立文件和实验/Run 的映射，未记录的执行信息与科学证据仍保持未知。科研任务本身需要的分析、论证和验证照常进行。

## 研究思路

任何头脑风暴都要有可追溯依据，区分事实与假设，先查已有正负实验和相关文献，再依据对象选择方法及验证；遵循 [有依据的研究头脑风暴](project/guidelines/EXPERIMENT_SPEC_STANDARD.md#0-有依据的研究头脑风暴)。
