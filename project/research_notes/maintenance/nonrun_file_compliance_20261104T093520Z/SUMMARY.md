# 非实验文件规范整理回执

本次按用户要求只整理实验目录以外的文件。`experiments/`整体排除，未归档、迁移、修改或删除其中任何文件。

## 处理结果

- 将1741份执行代码、依赖、完整备份、原始导出和并行规范在服务器校验归档后移出本地。
- 将348份写作材料、助手记忆及定位／导航记录迁入工作台指南规定的位置。
- 清除44份生成缓存；保留文献原文、研究笔记、固定版本来源、写作成稿与相关图源。
- 通用规范目录只保留`通用规范.md`；研究线专属规则已转为服务器归档，当前入口不再引用它们。
- 基础Agent配置改为读取AGENTS及其两份指南，不再引用不存在的研究简报、Research Wiki或ARIS路径。
- 一份Idea的进度字段由`in_progress`规范为同义的`exploring`；科学判断未改。

## 主要保存位置

| 内容 | 位置 |
|---|---|
| 写作成稿、源文件与图源 | `paper/manuscripts/writing/` |
| 项目导航清单 | `project/research_notes/maintenance/navigation/PLAN_CATALOG.json` |
| 存储与交付定位 | `project/research_notes/maintenance/storage_pointers/` |
| 助手历史记忆 | `project/research_notes/maintenance/assistant_memory/` |
| 维护清单和简明回执 | `project/research_notes/maintenance/<operation>/` |

## 服务器原件

主归档：`/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/storage_handoffs/nonrun_file_compliance_20261104T093520Z`。

补充归档：`/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/storage_handoffs/nonrun_file_compliance_20261104T093520Z/supplement`。

服务器已核验主归档与补充归档的每个SHA对象和清单引用；另4份源码／超长路径备份逐文件复制并核验SHA，共保全2091份原文件。`OBJECTS.tar`保存去重对象，`FILE_MANIFEST.jsonl`用原项目相对路径定位这些对象；逐文件副本由相应回执给出原路径和服务器位置。原位置迁移或本地删除不改变原始字节的SHA。

主归档及补充归档中的原件，可在服务器使用各自目录中的`restore.py`，指定原路径前缀和一个全新的输出目录；逐文件副本按相应回执复制。恢复工具不执行其中代码。归档证明见[主归档回执](ARCHIVE_RECEIPT.json)、[补充归档回执](SUPPLEMENT_ARCHIVE_RECEIPT.json)、[源码快照回执](SOURCE_SNAPSHOT_ARCHIVE_RECEIPT.json)及[超长路径备份回执](LONG_PATH_BACKUPS_ARCHIVE_RECEIPT.json)，逐项迁移和最终核验见[RECEIPT.json](RECEIPT.json)。

历史记录内的原路径只表示取证时位置，不能据此宣称本地原件仍存在。源文件整理与工作台索引刷新分开，本次未执行sync、提交或推送Git。

## 最终核验

- 非实验部分未发现未归属文件、执行代码、依赖、缓存或伪装成文本的代码备份；只读文件检查工具和基础配置除外。
- 44份纲要／Idea的元数据及引用通过，459个当前知识链接有效，15份固定版本来源SHA未变。
- PASS: 5 active docs, 446 local links, current routing。
- 两个展示页面的10处迁移引用已修正；额外源码证据的服务器定位见[源码快照回执](SOURCE_SNAPSHOT_ARCHIVE_RECEIPT.json)。
- AGENTS、通用规范和工作台生成索引字节未变；未刷新工作台索引，未提交或推送Git。

最终清理回执已逐字节回传服务器：`/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/storage_handoffs/nonrun_file_compliance_20261104T093520Z/CLEANUP_FINAL_RECEIPT.json`，SHA-256：`908b429aa7929dcfe4fb258b2f2a3db4b547023fc5a07c14f2e46a105a2f133e`。该回执包含额外4份逐文件副本的原路径映射。Git差异空白检查按仓库现有换行设置通过。
