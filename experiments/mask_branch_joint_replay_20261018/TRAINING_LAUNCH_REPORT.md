# 恢复开训记录

2026-10-18 11:43:51（Asia/Taipei）核查：恢复前置全部通过，G_budget正式训练已运行1625/15000步，C_native与PC_native仍待依赖；不是三组已完成。当前任务链将继续原预算训练和评价，不需保持SSH或Codex连接。

- 原生/旧G完整5000图：原生Box AP52.347950、Mask AP44.013836；旧G Mask AP44.152393，增益+0.138557。三项迁移护栏通过，官方发布指标parity仍failed，不改写。
- 全5000历史输入张量/几何重放通过；同机539125最终检测。原生成功GT22958，较历史22957为失7/增8，后续按22958精确census，不混旧分母。
- 正式32图smoke通过：三臂各2工程步，原生RLE精确，有限梯度，PC原型与系数梯度均有效；显存峰值reserved4752146432 bytes。工程权重丢弃，正式从原初始化开始。
- 三组保持原20k图、3轮/15000步、有效batch4、AdamW lr1e-4和原门槛。当前G参数167872/17张量，实际优化器核对通过。
- 独立作业108已完成；109 G运行；110 C、111 PC、112评价及2000重采样、113结尾证据核验等待依赖。113只读已存在Run并生成closure/results/REPORT.md、CONFIGURATION_AUDIT.json；不是定时任务，不追加模型运行或训练，也不替代本机独立统计复算。

核心Run：RUN_JOINT_PREPARE_56fa3ba7fe974a9ab6df42599788ef75；RUN_JOINT_NATIVE_299731c6c64d4cc0acf3b0915401e1e6；RUN_JOINT_SMOKE_8c70c8df51504bdabbc3813a7ac7af9d；RUN_JOINT_TRAIN_G_budget_73cca72b692e4097a974426de2a5b226。

438项冻结文件及19项CPU工程测试通过，20项启动/迁移/smoke收据回传SHA通过。原失败Study不覆盖。详见PROTOCOL.md、ENGINEERING_RECOVERY.md、audits/launch/TRANSFER_VERIFICATION.json及SLURM_SUBMISSION.json。
