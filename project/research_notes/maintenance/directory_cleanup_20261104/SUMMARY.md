# 目录清理与资料迁移回执

用户指令：第一类和第三类删除，第二类对接工作台指南迁移。

## 已删除

- `project/knowledge/`
- `project/facts/`
- `project/hypotheses/`
- `project/audits/`
- `project/research_status/`
- `artifacts/`
- `assets/`
- `shared/`
- `archive/`

删除前盘点共47,231个普通文件，逻辑大小22,802,594,414字节；另有4个链接。此数值不是磁盘实际回收空间测量。

## 已迁移

| 原位置 | 现位置 |
|---|---|
| `literature_search/` | `project/research_notes/literature/` |
| `project/knowledge_sources/` | `project/research_notes/sources/` |
| `_maintenance/` | `project/research_notes/maintenance/` |
| `paper/presentations/` | `paper/manuscripts/writing/presentations/` |
| `paper/manuscripts/paper_documents/` | `paper/manuscripts/writing/documents/` |
| `paper/manuscripts/paper_drafts/` | `paper/manuscripts/writing/drafts/` |

六个目录的2,836个原文件均存在，文件标识保持；其中21个迁移文件仅按清单修改引用或清单路径，其余原文件大小和修改时间保持。原维护回执按当时内容留存，内部原路径不是现行导航。

## 核验

- 当前入口检查通过：5份文档、445个本地链接。
- 当前纲要与Idea的459个文件链接有效，48处证据路径有效。
- 15份固定版本来源SHA通过；来源快照未改写。
- 11份保留文件SHA未变，473个实验目录保留。
- `.git/`、`.research/`、`.codex/`、`.workbuddy/`及现有实验、主线保留。
- 未访问服务器、刷新工作台索引、提交或推送Git。

清理后的文件位置见[工作台指南](../../../WORKBENCH_GUIDE.md)，逐项记录见[RECEIPT.json](RECEIPT.json)。
