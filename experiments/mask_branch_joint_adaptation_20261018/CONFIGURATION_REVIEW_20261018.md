# 训练前工程中止：配置复核

2026-10-18。只核已有产物；没有追加训练、重新评价或修改冻结实现。

| 对象 | 核验结论与范围 |
|---|---|
| 冻结协议/配置/源码/输入资产 | 本机和服务器各420项全部匹配FROZEN_MANIFEST；manifest两机相同；PREFLIGHT/PREPARE/NATIVE的CONFIG_FROZEN与CONFIG字节相同 |
| Python与依赖 | 服务器独立101核验Python3.12.3、torch2.5.1+cu124、torchvision0.20.1+cu124、Ultralytics8.4.100、numpy2.4.3、pycocotools2.0.11、faster-coco-eval1.8.0、OpenCV4.13.0.92、Pillow11.1.0、PyYAML6.0.2通过 |
| 权重/初始化 | 权重、旧G initial及last均在420项固定资产核验内；没有新增正式训练checkpoint |
| 数据 | train118287、val5000及标注SHA通过；20k固定选择和三轮顺序已生成；5000验证图集合及逐图SHA与历史一致 |
| 实际输入身份 | **失败**：4434图(batch,position)不一致、3610跨批、20图input_hw改变。版本和名义batch16相同不能保证矩形输入相同 |
| 16图集成预检 | 完成首批16图/2349行，zero-delta RLE及三臂初始P/c/logits精确重放；每臂只用前4图反传，零optimizer更新。G的17张量梯度有限、末层2张量非零；C的18及PC的32张量均有限非零。只通过程序内44GiB上限，未记录实测峰值或正式8小时预算预测，不能替代正式smoke |
| 实际参数与数值策略 | 已执行阶段的26项原生参数、8项runtime policy及8份固定源码pin匹配。ENVIRONMENT未单列faster-coco-eval版本，但独立READY记录1.8.0，入口版本检查也覆盖全部9包 |
| 训练配方与预算 | 20k×3轮、有效batch4、AdamW lr1e-4等冻结配置未变；三臂均未启动，不能将计划配置写成已执行15000步 |
| 原生/旧G迁移 | NATIVE仅导出32图4234行后失败；两个JSON数组均未闭合。完整Ns/G_old_s AP、原成功GT census和迁移护栏均未完成 |
| 正式评价 | 没有三臂最终checkpoint、完整预测、AP/损伤、2000次bootstrap结果；原科学Gate全部待评 |
| 配置总体判断 | 静态版本、源码和配置未漂移，但实际验证批次身份漂移，**当前执行不能作为有效同配置方法比较**。不得放宽严格断言 |

冻结协议SHA：`4d7c983e70f6d9c62266f768b2738cd4acf0d0e0249bd5ac746c5dd65a9f9f97`。
配置SHA：`3386fee552835e1fbdde782cb927ac70dfe4f4416433092358132ba671274437`。
manifest SHA：`c79f73d44b281ec38196228b1615a47d360fc97f3c23c19bbd01421bee87958c`。

本轮可复用的是已下载数据、独立环境、冻结原始输入和已通过范围内的工程证据。旧失败Run、原预测片段和原始日志全部保留；准备合同不能直接视为历史验证重放已通过。

证据：[本机配置核验](audits/status_20261018/LOCAL_CONFIGURATION_AUDIT.json)、[服务器420项SHA复核](audits/status_20261018/remote_snapshot/audit/FROZEN_RECHECK.json)、[完整批次比对](audits/status_20261018/BATCH_ORDER_AUDIT.json)、[数据环境就绪](audits/status_20261018/remote_snapshot/study/data_environment_READY.json)、[130项回传校验](audits/status_20261018/TRANSFER_VERIFICATION.json)。

逐项收据审阅见[RECEIPT_REVIEW.json](audits/status_20261018/RECEIPT_REVIEW.json)。
