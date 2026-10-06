# 本研究线运行规范

本文件只适用于 `neighbor_sensitive_p3` 研究线，不是项目全局规范。用户当前线程指定其他研究目录时，以其他目录的规则为准。

- 研究对象：YOLO26m-seg 与 COCO 实例分割；官方 COCO 预训练权重 `yolo26m-seg.pt`。
- 主要运行环境：conda `pytorch`；当前已核对的 Ultralytics 版本为 `8.4.100`。
- 默认输入与评价：原始 COCO 标注、原协议数据范围、640 输入和 FP32；具体图片清单、checkpoint、seed、分支和指标以单个实验 `PROTOCOL.md` 为准。
- `mask_ratio=1`、`overlap_mask=true` 是当前已记录的官方配置事实；若实验改变它们，必须在实验协议中单独说明。
- one-to-many、one-to-one、NMS 和诊断分支不在本文件混为一个默认结论；比较时按 Run/协议分别记录。
- 论文目录保存研究线文件；实验仍独立放在 `experiments/`，不因论文复制。

研究线入口：`README.md`、`DEVELOPMENT_PLAN.md`、`EVIDENCE_MAP.md`。线程关联以项目 `research_lines.json` 的 `owner_thread_id` 为来源信息，不决定科学结论。
