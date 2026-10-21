# RFC A/Q 训练完成复核（评价前）

2026-10-17真实笔记本状态：A于13:08、Q于13:28（Asia/Taipei）完成，旧有限训练序列已退出。两臂各1500步、3轮、6000图访问；没有追加训练。当前科学效用需由另行授权的冻结评价裁决，本文件不宣称方法有效。

| 项目 | 核验结果 |
|---|---|
| A | RUN_RFC_TRAIN_A_59b094a35f9b4fc3a45f7a86862ec677；last SHA 1383818808b7c52594ba5dfb49ae8f4b858a900b02fb65dccd5f1c55e6bf1963 |
| Q | RUN_RFC_TRAIN_Q_ed1e5e49f60c42908d23822e5804f668；last SHA 877a3b221fb0b81fa20772495cfbade530402252202a8fff86a2cf816c058942 |
| 数据/初始化 | 原2000图、218209输出；IMAGE_ORDERS、TRAINING_JOIN与G字节一致；initial.pt与原B一致，张量SHA ff3107ba… |
| 容量/实际更新 | 均167872参数、17张量；原训练日志17张量均有非零有限梯度；本机独立读取保存的AdamW状态，每个参数step都恰为1500 |
| 优化 | 原soft-IoU + 0.1保护BCE；AdamW lr=.001、betas=.9/.999、eps=1e−8、wd=.0001；FP32、无AMP/调度/早停/增强；只使用最后一轮checkpoint |
| 数值环境 | 笔记本Python3.12.3、torch2.5.1 CUDA12.4、Ultralytics8.4.100、NumPy2.4.3与G实际环境逐项一致；未把命名为deterministic的CLI参数当作真实后端状态 |
| 保留的空监督 | 每臂42次无匹配监督图片访问、1368次空ROI实例（跨3轮）；未从数据或评价分母删掉 |
| 冻结保护 | 原P/c0/基础模型、Box/Class/score/排序与旧G完整保留；全部训练Run产物SHA通过，训练源码与原EXECUTION_SOURCE一致 |

[TRAINING_PAIR_FREEZE.json](TRAINING_PAIR_FREEZE.json)封存双checkpoint；[独立CPU张量/优化器核验](preflight/TRAINING_CHECKPOINT_INDEPENDENT_LOCAL.json)使用本机torch2.9.1仅读取已有文件，没有forward，模型运行环境仍是笔记本冻结版本。[训练回传收据](preflight/TRANSFER_VERIFICATION_05.json)记录355文件/ZIP哈希核验。

[评价前冻结Run](runs/RUN_RFC_EVAL_FREEZE_1cea8787508c4d59a29c2a1dbb75c451/TRAINING_COMPLETION_VERIFICATION.json)在笔记本再次核对训练身份后，先冻结推理与源码，再一次抽取新损伤面板。原失败RUN_RFC_EVAL_FREEZE_dcf258fd926f4634b75574c35d06bb56为审计字段名错误（steps误读为optimizer_steps），发生于抽签和科学预测之前；保留失败，工程修复没有改checkpoint、配置或门槛。旧入口源码保存在preflight/evaluation_freeze_v1/。
