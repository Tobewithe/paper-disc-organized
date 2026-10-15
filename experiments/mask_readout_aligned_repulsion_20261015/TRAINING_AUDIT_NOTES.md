# 训练审计回执的求和差异

R与REP的笔记本训练审计均通过。回传当前机器后，冻结审计器的全部配置、图序、join、源码/产物哈希、初始化、checkpoint、实际Adam步数、参数白名单与逐条loss组合检查仍通过，但最后与已保存笔记本回执逐字段完全相等的断言失败。原异常为 `prior audit differs`，原脚本与笔记本回执未改动。

JSON规范化后，唯一差异是描述性 `per_epoch_mean_training_loss`：R最大绝对差2.220446049250313e−16，REP最大4.440892098500626e−16。当前机器Python 3.10.19的普通 `sum(losses)/2000` 与笔记本回执的结果末位不同；当前机器对同一份6000行日志使用 `math.fsum(losses)/2000`，两臂三轮均与笔记本回执完全相等。差值也均在标准浮点顺序求和前向误差界内。训练发生在同一笔记本环境，未将当前机器环境用于模型训练或预测导出。

新增的独立补充脚本 `scripts/audit_training_report_supplement.py` 执行原审计器直到上述最后断言，只允许描述性轮均值符合该精确解释；所有其他JSON字段必须完全一致。不对科学指标、Gate、优化器值或哈希增加容差。收据为 `INDEPENDENT_LOCAL_TRAIN_R.json` 和 `INDEPENDENT_LOCAL_TRAIN_REP.json`。据此将配置状态记为“实质检查通过，回执求和差异保留”，不声称原冻结审计脚本在当前机器原样全通过。

早期排查文件 `R_LOCAL_AUDIT_DIFFERENCE.json` 保留其排查时状态。该文件按Python对象直接比较还列出optimizer betas的tuple/list表示差异；原审计器本来就使用JSON比较，补充审计也确认优化器JSON完全相同，这不是配置漂移。

没有重训、重选checkpoint或改写Run。该补充不决定S2科学结论；科学验收仍由完整预测和原冻结门槛给出。
