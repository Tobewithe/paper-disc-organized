# 复制迁移交接与盘点更正

记录于2026-09-15，来源为两个负责任务向协调任务返回的交接消息。以下远端状态由负责任务报告，协调任务本轮没有连接远端读取或控制运行。

## 本地写入切换

两个任务均确认停止对原 paper-disc 的新增本地写入，后续读取 paper-disc-organized/AGENTS.md 并使用整理版绝对路径。原远端运行目录保持原位置。

主线任务（01a09da9-5107-7cb2-81b4-7ef90f80f19f）已在整理版修正 evaluate_full_coco_exports.py 的汇总键名冲突。整理版保留该新版本；旧目录中的版本另存 .research/copy_sources/experiments/counterfactual_p3_fullcoco_20260914/evaluate_full_coco_exports.py，没有覆盖新版本。主线后续结果接收目录为 experiments/counterfactual_p3_fullcoco_20260914/remote_snapshot_20260915；独立重评分运行记录位于 runs/official_rescore_20260915_v2。

笔记本任务（01a09fc8-1ce1-7ec0-b7e6-8cecdbc0605c）列出的 PROTOCOL_DECODER_V2.json、METHOD_DEVELOPMENT.md 及五份配套脚本已纳入增量复制，包括最后写入的 summarize_decoder_controls.py。

## 旧 heldout 样本数更正

笔记本任务报告：实际读取远端 matched_records.csv 得到4944图，旧脚本的半集合为2472图；此前247图来自误读本地500图副本。当前盘点应使用该任务更正的2472图，仍不能解释为符合4500图协议的验证。早期盘点和旧摘要保留原版本以追溯这次更正；后续科学结果记录仍需关联实际输入清单和文件版本。

## 执行状态交接

主线任务报告 fullcoco seed0 一轮训练及后评估已结束，正在原远端使用已保存预测修正 COCO 类别编号和空预测图片范围后的重评分，没有重启训练。

笔记本任务报告新的运行 ID 为 RUN_c0797f4ee9644d459f825cfd76ea90d6，执行位置 D:\coco_wire\runs\mask_boundary_route_20260914\RUN_c0797f4ee9644d459f825cfd76ea90d6，使用500图官方解码顺序与8个固定对照，runner 托管，报告时正在运行。前次 OOM 运行 63ca40e6dc0748e6b95e92f5f272c0d1 已由该任务保留失败记录。此交接信息不替代实际执行端 run.json，未由协调任务合成或确认运行成功。
