# 服务器执行接续

2026-10-20；这是一次性任务的接续记录，不是定时器或自动监控。状态更新以服务器实时 Run 和 Slurm 为准，本文/本机 Study 是时间点快照。

- 登录：`ssh -p 22 a25023104@192.168.71.10`。
- 根目录：`/srv/gpu-lab/users/a25023104/projects/video-occlusion-pretraining-v02`。
- 解释器：`/opt/yolo/ultralytics-8.4.100/.venv/bin/python`，PyTorch2.6.0/CUDA12.4。
- 完成：E0的8家族/1,152帧、自动标签、四臂16工程步、最终零更新核验。`E0_PASS.json` 为真。
- 当前：Job160完整数据生成、检查、底座特征缓存，Run `RUN_VOR_RENDER_PILOT_TRAIN_0000_565c71df87ac`。
- 已提交：Job161依赖160成功后，顺序运行B/P/I/Hpos四臂各2,000步，再评价和固定序号可视化。当前正式训练0步，不将排队当启动。
- 资源：每个阶段单卡A6000/4CPU/24GiB；160最多24小时，161最多12小时。上游失败则后继取消，不自动重试或增加预算。
- 其它账号和其它研究线的作业不可取消或修改；尤其本账号 Job154 属于另一研究线。

先读 `PIPELINE_STATUS.json`、`JOB_CHAIN.json` 和最新 `runs/*/run.json`，再结合 `squeue -u a25023104`、`sacct -j 158,159,160,161` 及 `/srv/gpu-lab/users/a25023104/logs/vor-<jobid>.log`。队列消失不代表成功；失败需看退出码及错误日志。

正式源码已在 `TRAIN_FREEZE.json` 冻结。不要直接修改 `shared/video_occlusion_pretraining/*.py`、配置、数据家族清单或E1协议；训练启动会验证 SHA。若有工程故障，保留旧 Run，先明确修复是否改变科学合同、累计预算和重新执行范围，不能盲删锁、原地覆盖数据或从头多跑2,000步。

自然结束后读取：

1. `trained_B/P/I/Hpos.json`：末步checkpoint和实际Run。
2. `E1_RESULT.json`：评价Run、逐家族对照及预冻结裁决。
3. 评价Run中的 `SUMMARY.json`、`FAMILY_METRICS.json`、`VALIDATION_THRESHOLDS.json`、`REPORT.md`、`synthetic_results.png`。
4. `visualizations/`：固定test_0000–test_0003的真实预测比较，不按成功率挑案例。

核对四臂初始化、数据暴露、实际优化步数和最终checkpoint，回传轻量文件并更新本线 Study/ACTIVE_PLAN。不要重跑已完成步骤，不自动20k、多seed或改阈值。当前E1只验证同渲染器的留出资产、条件对应与实例定位；真实视频和固定YOLO/BoT-SORT效用未测，不能提前宣布改善。

自动 `DECISION.json` 首先裁定 P 主比较。结案还必须依据已经输出的 Hpos−P 和记忆干预区间，逐项执行 E1 的竞争解释裁决；不能把自动主比较状态当成 Hpos 的否决，或仅以记忆扰动退化宣布机制成立。

本线优先科学问题仍是：**P相对等预算B，是否在改善重现恢复的同时控制错误接受，并且确实使用额外历史？** 通过才用同一checkpoint做真实冻结迁移；未通过依据预冻结出口裁决，不再加规则拼接。
