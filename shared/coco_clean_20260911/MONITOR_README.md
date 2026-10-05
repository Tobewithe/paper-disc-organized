# 远端训练监控

直接在远端终端运行，无需安装依赖：

```bash
bash /root/autodl-tmp/coco_clean_20260911/monitor.sh
```

默认每 5 秒刷新。显示六组队列状态、进程存活、当前训练/验证进度、GPU、磁盘、最新验证指标、已登记 checkpoint 数量，以及当前 CCL 任务最近完整训练轮次的配对统计。首轮验证前 AP 显示 `—`；训练完成、验证完成、checkpoint 保存分别计数。

```bash
# 只打印一次
bash /root/autodl-tmp/coco_clean_20260911/monitor.sh --once

# 每 10 秒刷新，隐藏最近日志
bash /root/autodl-tmp/coco_clean_20260911/monitor.sh --interval 10 --log-lines 0

# 输出一次 JSON，供其他脚本读取
bash /root/autodl-tmp/coco_clean_20260911/monitor.sh --json
```

`Ctrl+C` 只退出监控。脚本只读文件和进程状态，不启动、停止、重启或改动训练，也不会发送通知。日志超过 10 分钟未更新会提示检查；这一提示本身不代表失败。队列 JSON 仅在任务切换时更新，因此不将它的修改时间当作训练心跳。

AP 显示为 0–100 分，来自当前训练流程的验证 CSV。不同轮次不能直接作优劣比较，最终论文结果仍使用原始 COCO JSON 和正式实例分层评估。checkpoint 计数只包括索引已登记且实际文件大小相符的逐轮文件，不加载权重、不核算完整文件哈希。

监控默认读取脚本同目录的 `queue_status.json`、`runs/`、`logs/`；移动脚本时可用 `--root /path/to/experiment` 指定目录。进程存活核查需在训练所在的 Linux 实例运行；GPU 读取依赖系统已有的 `nvidia-smi`。
