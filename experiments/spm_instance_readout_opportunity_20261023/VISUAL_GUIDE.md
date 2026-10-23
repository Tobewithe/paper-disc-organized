# 实例原型学习质量图集

四组图来自同一个零训练诊断。GT 支持只用于分析，不能作为正常部署结果或所有读出器的数学上界。原始 Run 图保留，展示重排不改数据、对象选择或科学裁决。

| 图 | 阅读要点 |
|---|---|
| [GT 支持的均值读出](presentation/01_GT_supported_readout.png) | 305 个固定失败；训练特征优于随机，但均值 IoU 增量与修复率均未过门槛。重排数值和门槛说明，避免标注靠近画布边缘。 |
| [真实训练实例覆盖](presentation/02_training_coverage.png) | S 1646/3308、M 5388/10284、L 967/1050；只是固定提取 episode 覆盖，不是优化器访问次数。图例已移出柱顶。 |
| [无 GT 全量 AP 与尺寸分层](runs/RUN_SPM_INSTANCE_OPPORTUNITY_4d95693a2383/figures/03_GT_free_AP_and_size_controls.png) | 原生 57.3153，训练读出 25.2765；五条全量预测均通过独立复算。 |
| [同一实例掩码与相似度](runs/RUN_SPM_INSTANCE_OPPORTUNITY_4d95693a2383/figures/04_same_instance_readout_cases.png) | 绿色在一行内始终标记同一固定对象。按预注册规则选样，不是四个最差病例；存在原生成功对象。标题中的 upper-bound 应按本页说明读作有限 GT 支持诊断。 |

数据来源与科学解释见[完整报告](REPORT.md)，展示脚本为[scripts/render_review_figures.py](scripts/render_review_figures.py)，新图哈希见[presentation/MANIFEST.json](presentation/MANIFEST.json)。未来 V5 的对象三维图、跨视角对应与遮挡恢复图尚未产生，不能用本诊断图冒充新模型结果。
