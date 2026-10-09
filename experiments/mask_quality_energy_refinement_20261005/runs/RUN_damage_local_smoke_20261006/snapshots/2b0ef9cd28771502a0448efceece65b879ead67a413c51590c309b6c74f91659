# 损伤机制诊断，2026-10-06

用户已关闭租用服务器，并明确允许在本地运行。这里只执行已有 checkpoint 的诊断，不训练、不改变 Stage-I 评价结果或 STOP 决定。

固定样本为 `IMAGE_SPLIT.json` 中前 64 个 DEV image ID，使用全部官方 one2one TAL 候选，不按结果挑选。先独立执行前 2 张图片的 smoke Run，再执行 64 张主 Run；两次各有独立 run.json。重新从原始 JPEG、官方权重和 FP32 官方数据管线生成 P、h、c0、框和 overlap GT，逐候选核对既有 DEV 身份与 A/D1/D IoU。Windows、RTX 5060 Ti、Torch 2.9.1+cu128 与历史 Linux/Torch 2.8.0 的差异必须记录，不宣称完整数值等价。

冻结 Quality epoch-3 checkpoint、FIT 导出的 rho、lambda=0.003 和原来的两步更新。证据 e 始终固定在 c0，不把 GT 输入模型。GT 仅用于事后标签及梯度方向诊断。

每个候选记录 gQ(c0)、训练实际定义的 cropped overlap SoftIoU 对 c 的梯度及夹角。沿同一个单位 Q 梯度方向，固定观察位移 rho/4096、rho/128、rho/8、rho/4 和 rho/2 的 Q、训练 SoftIoU、正常原图 HardIoU。rho/4096 用于局部方向检查，rho/128 用于有限小步检查，rho/8 和 rho/4 是 Success 训练半径，rho/2 是实际第一步。该曲线不用于选择步长、checkpoint、半径或模型版本。

复现原来的第一步及第二步，同时记录第二步 gQ、0.003*(c1-c0)、gE=gQ-reg 的范数、夹角和实际位移。额外构造相同第二步长度下的 Q-only 和 penalty-only 第二步，隔离第二步恢复的机制；这些只是假设干预的诊断，不是新的方法评价。所有解码统一使用原来的 official process_mask 和 inverse-letterbox 二值路径。

按 baseline Mask75 success、最终 D 损伤及其余候选给出计数与比例。小样本是描述性机制证据，不能替代全量 FINAL 或支持泛化收益。方向失配、有限步长跨峰和 surrogate/正常解码不一致可以重叠，不把它们当作互斥原因分摊。

纯统计 Run 另外分析既有完整 DEV/FINAL 原始行，保存来源 SHA、脚本和运行状态。失败或部分 Run 保留。不得覆盖既有实验产物，原执行协议及历史报告保持不变。
