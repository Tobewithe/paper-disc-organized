# 第三轮交付凭据

真实执行设备：用户授权的笔记本28358lan，主机名YANGLEI；环境见[ENVIRONMENT.json](ENVIRONMENT.json)。本轮新增训练0步，固定方法全比较及事后裁决已完成；科学状态为`negative_fixed_rollback_configuration`。

- [完整实验报告](REPORT.md)
- [离线展示](../../paper/presentations/密集视频身份保持_确认与回退实验_20261019/index.html)
- [展示说明](../../paper/presentations/密集视频身份保持_确认与回退实验_20261019/使用说明.md)
- [当前Study状态](study.json)
- [回传验证](TRANSFER_VERIFICATION.json)
- [完整工程验证](runs/RUN_REVERSIBLE_DELIVERY_VERIFICATION_20261019T013919_9f5f7ca2/SUMMARY.json)
- [冻结科学裁决](runs/RUN_REVERSIBLE_ADJUDICATION_20261019T012656_3c285433/DECISION.json)

7,595原视频帧、19段视频×7方法的完整成绩；固定身份层对照共99,259个基线观测，几何/分数/行身份和计数一致。各Run的实际源码与合同哈希通过检查；输入视频、GT和模型哈希通过检查。4段1600×900视频共175输出帧全部解码通过。HTML本地资源链接和133行成绩表通过检查；图表、关键视频帧、结构图已视觉核查。没有宣称已做浏览器端交互渲染验收。

回传包`C:\Users\Public\thesis-video-v3-20261019\delivery_v3.zip`：106,666,297字节，1,556个文件逐项校验通过。SHA256：`5d7b3acf87333b431919bb846c32d039085e34f605e0150c48c06653e6548c6e`。

未达到的目标明确保留：稳定超过ByteTrack及预指定BoT-SORT未成立，回退净效用未成立，可靠完整遮挡恢复标注集仍缺失。11例物理遮挡补充评价来自AI复核官方几何标签，不冒充完整人工重新标注。工程交付完成不等于这些科学目标达成。
