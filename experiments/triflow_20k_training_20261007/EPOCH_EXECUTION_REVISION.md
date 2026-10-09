# 逐轮评估执行修订

用户于 2026-10-07 要求每个 epoch 都评估完整 5,000 图。原训练四文件和 `PROTOCOL.md` 的字节保持不变；新增透明 observer 在原训练实际 `epoch_boundary` 原子保存之后同步评估，再返回同一训练循环，额外源码及补充协议另行记录。

实际 32 图工程验证通过：带真实边界回调的完整训练与原 32 图参考的 head、AdamW、全部 RNG、样本顺序、梯度及科学计数/损失逐字节一致，最终 head SHA均为 `5f3c4751cd96132a26eeb586be37bca692b6f7c6f72061f8a32f275e1cd4987f`。工程回调只作为 observer 执行验证，不冒称正式 COCO AP。

原正式 S0 在最新已持久化断点处控制停止，保留原 PID/源码/日志和更新；Windows PowerShell 缺少 Get-FileHash 导致停止后复制登记失败，Python 的 R1 恢复登记校验并复制实际 payload 后通过。observer R1 再从该前缀续跑。随后交叉审查发现两处接口错误：epoch evaluator 的回调 SHA角色键，以及独立核验器的 checkpoint 文件名下划线；只修这两份新增回调源码，旧源归档保留，再使用 R2 来源锁定续跑。以上停止的 failed Run 是受控执行修订记录，不是模型科学失败。

当前 R2 从 **第1轮第15,874个已遍历图像位置、chunk4、36,581次已提交更新**恢复；不可变恢复文件 SHA256 `969d7ff17b503af1b0c2d760e6fc5dab2dd37c86da74f812ca3927b25dbd7615`。`RUN_TRIFLOW_20K_TRAIN_S0_R2` 于 2026-10-07 18:16（北京时间）实际启动，已继续更新；原科学来源 SHA仍为 trainer944d/stream6460/frozenIOcff75/core1b96，额外 observer 为 e64c，新评估器 fae9，新独立核验器 c8fb，各自完整 SHA见对应 SOURCE/OBSERVER_INPUTS。

每轮真实保存 `head_epoch_XX.pt`，对应独立评估/核验 Run；前7轮计算完整集 AP/AP75/APsmall/BoxAP，第8轮增加原配对诊断。上述 R2 启动登记时 epoch1 尚未完成；后续实际评估及核验分别保存在所属 Run，固定第8轮最终结果需对照最终训练审计的真实参数状态。

2026-10-08 用户授权精简诊断后，第二轮回调完整结束，再从真实持久化前缀切换到 `RUN_TRIFLOW_20K_TRAIN_S0_R3`；实际诊断运行时改动、数值一致性验证、断点和来源记录见 `DIAGNOSTICS_REVISION.md` 与 `DIAGNOSTICS_EXECUTION_ADDENDUM.md`。原科学合同及全部历史运行保留。

2026-10-08 北京时间 23:09:55，R3 流程在终轮评价、独立核验、最终训练审计和曲线报告后真实完成（各阶段 exit0）；固定第8轮结果、范围、参考门槛及未知项引用 `REPORT.md`、`RUN_TRIFLOW_20K_EPOCH_CURVE_REPORT_S0/EPOCH_CURVE.json` 和各原 Run，不替换为最优轮次。本地已回传 19 个终态 Run，41,536 个文件通过清单校验，原 `transfer.json`、源清单及当前清单保留，冗余逐图方法缓存仍留笔记本。

依据独立 B 会话用户消息 `01a11c1c-d1b0-72e1-bb20-881cfd457673` 的真实发送授权，A 在原生读取核实后已向“审查员6AU”（`01a11bf8-26b3-7490-b8bf-62b4b308b1ba`）发送终态、验证、结果、原产物与未知项通知；发送工具成功不代表 B 已处理或通过科学审查。原 8 轮预算、历史及实际消耗保持，未追加实验；本配置阶段完成不代表新论文总目标完成。
