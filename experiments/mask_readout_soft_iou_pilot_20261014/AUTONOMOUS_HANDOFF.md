# 笔记本独立执行交接

2026-10-14 02:23（Asia/Taipei）：用户要求可断开连接，已将后续执行交给笔记本 YANGLEI 的一次性 Windows 计划任务 `CodexSoftIoUPilot_53afd0bb`。任务由本机 Task Scheduler 启动，控制进程 PID22820；注册 SSH 命令已退出后，控制器与评价继续运行。数据、缓存和 checkpoint 均位于笔记本本地 D 盘，不依赖本台电脑共享路径。

- Smoke：`RUN_SOFT_IOU_SMOKE_2cb8fd414d1540dcb4a16b34b0c0ac37`，通过；已回传、验SHA并独立复核。
- 正式训练：`RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb`，3轮、6000图访问、1500步完成；初始权重、逐轮顺序、训练配对与原B逐字节一致，原合同全部既有字段一致；独立 checkpoint/全日志审计通过。51项证据已回传并核验。
- S2评价：`RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0`，已由控制器自动启动；交接检查时损伤清单已冻结，科学结论尚未判定。

后台控制器 `scripts/continue_independently.ps1` 不属于新的模型或训练配置；仅接续既有训练、调用冻结评价器、独立复核及打包。其SHA和计划任务身份见 `AUTONOMOUS_TASK_RECEIPT.json`。任务不重复训练、不重抽已产生的损伤样本、不自动启动S3；工程错误保留产物并停止。

可关闭本台电脑上的 Codex/SSH 或断开网线。笔记本须保持开机、用户会话不注销、不合盖；控制器临时阻止空闲休眠，结束时释放。建议接电；不修改永久电源策略。

真实状态与完成结果保存在笔记本：

```
D:\coco_wire\baseline_native_20261013\experiments\mask_readout_soft_iou_pilot_20261014\
    AUTONOMOUS_STATUS.json
    AUTONOMOUS_RESULT.json       # 仅全部阶段完成后写入
    autonomous_execution.log
    runs\RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0\
        run.json / CONFIG_AUDIT.json / SUMMARY.json / DECISION.json
        LOCAL_INDEPENDENT_RECHECK.json / transfer.json / manifest.sha256 / evidence.zip
```

重连先读上述执行端终态，按现有Run回传评价包并独立复核，再写正式结果报告和更新总控；不要新建训练或评价Run。根目录本地 `AUTONOMOUS_STATUS.json` 是交接快照，实时状态以笔记本为准。
