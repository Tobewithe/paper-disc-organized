# AMT-v1 启动前配置核对

协议冻结时间：2026-10-15T18:07:18.798469Z（笔记本当地 2026-10-16 02:07）。
协议 SHA256：`5d944b782471f18558d017dad659882b0f0ec1efe172e2640f85349c622d974c`。

笔记本 [PREPARATION](PREPARATION.json) 与本机 [冻结核验](LOCAL_FREEZE_VERIFICATION.json) 通过：34 个源码/协议文件一致，31 项输入身份锁定，完整 2000 张缓存 SHA 核验完成；7 项行为测试两机通过。原生配置只允许既有训练缓存与 S0 的 data/project 路径不同，数值运行环境完全等于原训练。

| 事项 | 生效规则与证据 |
|---|---|
| 改变项 | 仅符合冻结规则的训练失败实例目标：原 GT→有限 oracle 的硬二值 proto 函数；不是系数向量回归 |
| 数据、匹配、分母 | 原 fit2000/IMAGE_ORDERS/TRAINING_JOIN，完整保留；逐图在教师生成和训练时重新核对 |
| 初始化与模型 | 原 B initial.pt，167872 参数，原 B 架构；不接续旧 last.pt |
| Loss与保护 | 原 soft-IoU 函数及全输出 .1 BCE；目标替换不改变实例/图分母；成功 GT 不换标签 |
| Optimizer与预算 | 原 AdamW、microbatch1/累积4、3轮/1500步；最后一轮 checkpoint；无 AMP/调度/增强/早停 |
| 预处理与解码 | 原 8.4.100、权重、P/c0/F、每图 rect/batch 几何；原 native decoder；所有零残差 RLE 必须精确相同 |
| 推理与评价 | 原生面积≥9216才应用新Δc；原 dev512/T655/Box75配对2925；新成功面板1024在checkpoint冻结后一次抽取 |
| 对照与门槛 | 原 soft-IoU checkpoint复用；原S2三项，加同开发T相对原面积门控对照的配对增量区间下界>0 |
| 额外成本 | 教师求解与教师/训练smoke单列；只保证adapter训练预算相同，不称总算力相同 |

一次性 Windows 任务 `Codex-AMT-Training-20261016` 使用与历史成功任务相同的 Interactive 用户上下文，无触发器、无失败重启，仅手动启动一次。由任务计划程序承载，启动 SSH 返回后新 SSH 连接已确认 pipeline PID468 与教师子进程继续运行。SSH 可以断开；依赖笔记本保持开机、登录和可执行状态。任务只到训练结束及独立训练配置审计，不自动启动 S3/S4。

这份文档是启动前核验，不是结案审计或科学通过证据。教师 Gate、真实训练日志/optimizer state、checkpoint 与实际目标使用仍须由各 Run 产物核验；完成后另交配置复核和通俗结果说明。
