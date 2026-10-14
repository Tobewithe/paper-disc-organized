# 本次执行状态

S0 已完成笔记本原生验证、零残差重放、独立 COCO 账本及两份历史对账。**原生 GT 账本通过；官方发表 AP 复现未通过。** 当前没有继续运行的本次任务，未启动 S1 或训练。结论与阶段表见 [REPORT](REPORT.md)，机器可读裁决见 [DECISION](DECISION.json)。

- 正式基线：`RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170`；5000 图、539128 条检测，Box AP 52.3514301541 / Mask AP 44.0141748477；一位小数 52.4/44.0，未达到 52.5/44.1。两套 COCO 评估器的 24 指标差均为 0.0，精确重放通过。
- 普通 GT 36335：Mask75 成功 22957、失败 13378；T=6229，仍未执行系数 oracle。另有 446 个 ignored GT。
- 历史对账：`RUN_HISTORY_RECONCILIATION_ea22c6c2ae3c49eda6899b1ea98f40f4`；两份原 GT 来源分别核验，官方图片大小/CRC 校验通过。
- 笔记本工程：`RUN_ENGINEERING_LAPTOP_4996fad760f7412a88d8d16f09598fb9`；24 项严格测试与配置检查通过。
- 16 图原 smoke：`RUN_NATIVE_SMOKE_a6dc7e4d82d847579eaf2c4f94416d05`；前向完成，评价记录钩子失败，原状态保留。离线恢复 `RUN_SMOKE_RECEIPT_RECOVERY_f3d8b73fb3d94a2bb36ceb756db6c34f` 通过 G1，没有重复前向。
- 执行端：`YANGLEI`、RTX 4060 Laptop 8GB、Python 3.12.3、Torch 2.5.1/CUDA 12.4；执行根 `D:\coco_wire\baseline_native_20261013`。独立 `_site` 固定 Ultralytics 8.4.100 与 faster-coco-eval 1.8.0，旧环境 8.4.27 保留。
- 原旧任务 PID 22408 按用户明确授权停止，落盘产物保留，见 [停止回执](OLD_TASK_STOP_RECEIPT.json)。没有重启旧审计。
- 轻量回执已回传并逐文件 SHA 校验，大预测、GT 表、P/c0/F 张量仍在笔记本对应 Run。每个回传 Run 保留 transfer.json、manifest.sha256 和本地校验回执。

采用 [通用配置规范](../../project/科研项目通用规范/基线复现与配置一致性.md) 与研究线 [配置合同](../../paper/manuscripts/neighbor_sensitive_p3/BASELINE_CONFIGURATION.md)。唯一下一步是取得并核对官方发布 Run 的实际验证回执和预测来源，定位发表 AP 差距；本轮不再追加前向，也不自动进入 S1。
