# 交叉审计协议

本实验固定官方 TAL one-to-one 候选、prototype、检测框和图像分片，只改变共享系数头拟合目标。完整官方缓存包含 fit/dev/val 796/197/196 张图、6058/1402/1346 个候选；两个目标都使用完整 fit 集，不做 20k 扩展，也不新增数据。

并列启动两条精确拟合任务：

- `old160`：COCO 原始实例 mask，按旧流程 letterbox 到 640、最近邻缩到 160，在 GT 框 support 上以 `BCE/area + 0.003/2||Δc||²` 拟合。
- `official640`：官方 TAL overlap owner mask，prototype 双线性到 640，在官方 target-box support 上以 segmentation gain `9.83241` 加同一输出正则拟合。

两条任务都使用 FP64 全候选解析目标和解析梯度，停止门槛为 `gradient_inf < 1e-6`。两张 map 完成后，统一在固定官方候选上用同一官方解码器评估 native、old160 map、official640 map，报告候选均值和 image-macro 两种口径，并用 5000 次整图 bootstrap 给出差值区间。

判定：只有两张 map 都达到梯度门槛且评估成功，才用结果解释损失目标与求解路径对旧/官方差异的相对贡献；任一 map 未达到门槛则该交叉审计不作科学结论，保留失败 Run。

队列脚本在笔记本后台运行并按依赖自动启动评估。启动后不需要本会话监控；日志、状态和结果写入本实验远端 `runs/`。
