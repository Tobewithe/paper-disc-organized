# 第一轮实验与系统交付

用户指定执行设备为离线笔记本 `28358lan`。本次模型推理、300 步身份学习、200 步更新收益学习及六组跟踪对照均在该设备完成。台式机完成下载、Public 中转和结果排版。

## 当前结论

原型系统和实验链条已完成；完整候选没有通过预设效用门槛。科学结论见 [REPORT.md](REPORT.md)，原始门槛见 [PROTOCOL.md](PROTOCOL.md)。本轮关闭当前配置的继续调参，不追加训练。

## 位置与启动

- 笔记本完整目录：`D:\thesis-video-20261018`。
- 在笔记本双击 `START_DEMO.cmd`，打开 `http://127.0.0.1:8871`；无需互联网。
- 当前台式机通过已有 SSH 沿网线转发同一地址，可直接浏览笔记本系统。
- 台式机 Public 证据镜像：`C:\Users\Public\thesis-video-20261018`。
- 仓库可交付材料：`paper/presentations/密集视频个体跟踪_实验与系统_20261018`。
- 原始检测缓存、特征张量、权重、完整 Run 和历史结果保留在笔记本。仓库只收入轻量回执。

## 完整性验收

核验 Run：`RUN_DELIVERY_VERIFICATION_20261018T233104_cbc8108d`，执行主机 YANGLEI。

- 11 个官方 train/dev/test 视频记录在演示方向修复前后完全不变。
- 36 份未受期限修复影响的逐帧对照输出 SHA256 一致。
- 共检查 22,218 组方法-帧、296,983 条输出：确认 ID 帧内唯一，框和分数来源于原始检测，检测行合法。
- 482 条确认事件全部等待不超过 6 帧。
- 从 IDTP/IDFP/IDFN 独立复算 IDF1，五个测试视频 GT 分母均一致。
- 5 个 1920×1080 MP4 逐帧完整解码；合辑 747 帧、15 FPS、49.8 秒。
- 浏览器实际选择并提交新视频，笔记本完成 15 帧分析，页面自动显示结果。导入不训练，无 GT，不进入科学成绩。
- 官方评价器完美轨迹/身份切换小例和 4 项因果/期限测试通过。

## 真实运行索引

- `features`：`RUN_FEATURES_20261018T225606_a471c9e2`
- `demo_features`：`RUN_FEATURES_DEMO_REPAIR_20261018T231428_024a5cf1`
- `identity`：`RUN_IDENTITY_TRAIN_20261018T230440_4db76d96`
- `utility`：`RUN_UTILITY_TRAIN_20261018T230451_0658b85b`
- `evaluation_dev`：`RUN_EVALUATE_DEV_20261018T231913_2554ce6e`
- `evaluation_test`：`RUN_EVALUATE_TEST_20261018T231947_350c57df`
- `evaluation_demo`：`RUN_EVALUATE_DEMO_20261018T232142_36b481c4`
- `delivery_verification`：`RUN_DELIVERY_VERIFICATION_20261018T233104_cbc8108d`

## 工程修复与边界

两个无 GT 演示片段的旋转方向修复，以及候选消失后的等待期限修复均有独立记录；旧结果保留。最终报告引用期限修复后的 Run，未更改训练预算、模型权重、测试集合或阈值。

AnimalTrack 的 GT 间断重现并不等于人工确认的物理遮挡。本次只有猪、马两类，不能据此宣称通用动物泛化或遮挡恢复已经解决。ByteTrack 与 Deep OC-SORT 的差异不只外观，不能将二者差值直接解释成外观的独立因果效应。
