# 本研究线运行规范

本文件只适用于 `mask_boundary_calibration` 研究线，不是项目全局规范。用户当前线程指定其他研究目录时，以其他目录的规则为准。

- 研究对象：冻结 COCO 预训练 YOLO26m-seg/YOLO26s-seg 的选择性掩码读出校准。
- 主要运行环境：conda `pytorch`；当前已核对的 Ultralytics 版本为 `8.4.100`。
- 默认评价范围：COCO `val2017` 的 4,500 图研究范围；m/s 模型、图片清单、校准划分、checkpoint、seed 和指标以单个实验 `PROTOCOL.md` 为准。
- 当前已记录的默认执行口径为 one-to-one、640 输入、`conf=0.001`、`max_det=300`、FP32；改变口径时在对应实验协议中声明。
- 训练、冻结读出、阈值诊断和迁移确认分别按 Run 记录，不把固定诊断收益当作部署方法结论。
- 论文目录保存研究线文件；实验仍独立放在 `experiments/`，不因论文复制。

研究线入口：`README.md`、`EVIDENCE_MAP.md`、`MECHANISM_EXPLORATION_PLAN_20260920.md`。线程关联以项目 `research_lines.json` 的 `owner_thread_id` 为来源信息，不决定科学结论。
