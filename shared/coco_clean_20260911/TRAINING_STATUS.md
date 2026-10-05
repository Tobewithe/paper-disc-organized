更新：2026-09-11。用户后续授权找到候选后自行实验，现已完成独立的三种子系数头/像素残差学习试验和全量 COCO 失败分解。原六组正式队列仍暂停，未重启。当前证据见 ../../refine-logs/coco-evaluation/MECHANISM_LEARNING_STATUS.md。下面 12:04 的‘没有更新网络权重’仅描述当时冻结诊断，不能代表后续局部学习实验。

# 干净 COCO 训练状态

当前决定（2026-09-11 12:04，北京时间）：按用户要求暂停后续训练，先定位机制。远端已完成官方预训练模型的 300 图 COCO 固定解码器诊断、闭式系数求解和复核；没有更新网络权重。最后检查无 `train_pair.py`、`run_queue.py` 或隔离训练诊断进程。历史队列仍保留 FAILED，未重启。

新证据见 `../../refine-logs/coco-evaluation/MECHANISM_LOCALIZATION.md`。下一阶段先验证空间归属方向，不继续学习率扫描或训练 CCL。以下是故障发生后的历史快照，不是当前运行进程。

快照：2026-09-11 11:16，北京时间。这是手工核验快照，实时状态以远程 `queue_status.json` 和运行日志为准。

- 远程目录：`/root/autodl-tmp/coco_clean_20260911`。
- 历史 Screen：`5645.coco_clean_3seed_train`；历史队列 PID 5647、训练 PID 5651，均已退出。
- 正式队列状态：`FAILED`。`baseline_s0` 在第 9 轮第 2278/2340 个 batch 因梯度总范数非有限而退出；失败时的张量未保存，不能仅凭该报错区分单个梯度非有限与范数归约溢出。已完成并保存 `epoch0.pt` 到 `epoch7.pt`；后续五组未启动。
- 顺序：`baseline_s0 → ccl01_s0 → baseline_s1 → ccl01_s1 → baseline_s2 → ccl01_s2`。
- 每组 15 轮，固定 batch16 / 640px / AdamW，种子 0/1/2，CCL 权重 0 与 0.1。
- 原协议要求每个完成的正式组保留 15 个逐轮 checkpoint（epoch0.pt 到 epoch14.pt），另外保存 last/best 和最新 FP32 recovery；主比较预先固定 epoch14.pt 的 EMA。当前没有完成的正式组。
- 数据和梯度检查通过；短程只检查执行正确性，不构成方法效果结果。
- 诊断状态：从第 8 轮有限恢复状态重新建立数据加载器后，第 9 轮 2340 个 batch 未重现异常；由于增强流不是原始运行的逐 batch 重放，这不能替代正式重跑决定。
- 正式训练尚未完成，不能把前 8 轮作为最终论文结果。后续重跑前需先确定数值稳定修复，并重新执行受影响的 Baseline/CCL 成对预算。

远程启动为持久队列，终端断开后继续运行；任一任务失败则队列停止并保存失败状态，不自动缩减 batch 或跳过种子。

本地证据：`audits/LAUNCH_GATE.json`、`audits/CCL_GRADIENT_PASS.json`、`audits/SMOKE_PAIR_PASS.json`、`runs/baseline_s0/launch_receipt.json`。本地 `queue_status.json` 仅为下载时快照。

