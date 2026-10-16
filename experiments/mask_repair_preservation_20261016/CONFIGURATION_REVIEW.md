# 运行后配置复核

2026-10-16。正式 `RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01`，smoke `RUN_REPAIR_KEEP_SMOKE_1473cf76889b4163b962d2e7086995b7`。**身份、冻结配置和统计复核通过；求解收敛门槛未全部通过。** 二者分别记录，不能用配置通过掩盖数值未决。

| 核验 | 实际证据与结果 |
|---|---|
| 原协议和源码 | 21份冻结文件两机字节SHA一致，运行前后未改；protocol SHA `7359df0ba8bc23314c98adcbb96ca188a2e06e97ea0952ec2e43f19b84de8123` |
| 源模型/数据/输出 | 17项源输入、247图和247缓存均与冻结值相同；[POST_RUN_INPUT_AUDIT](POST_RUN_INPUT_AUDIT.json)共532项检查零漂移 |
| 版本和实际运行参数 | Ultralytics8.4.100、Torch2.5.1、原源码SHA与各依赖、CUDA/TF32/cuDNN/确定性环境均与原生源合同逐字段相同，运行后环境相同 |
| 推理几何和掩码语义 | 原batch16/rect的真实cache几何；完整N行FP32 GEMM与原生分块、首次阈值/裁框、原图resize后byte；真实全输出核验数为**49976** |
| 最终身份与重放 | 49976条原生RLE逐像素精确；新臂所有非目标输出的两阶段mask不变；框、类别、score、检测ID及顺序固定 |
| 样本/优化预算 | 全256条/243图完整，768次有限求解；120iter/150closure硬上限、原正则/初值均未变；耗时1300.422秒<7200秒 |
| 梯度和求解状态 | C256通过、E251通过、EK238通过；23条未达≤1e−6。保留所有条目和失败，不补迭代、不把标准改成1e−5、不删除失败样本 |
| 数值代理与等幅 | 所有臂实际FP32评价；最大代理误差9.731662e−6，最大等幅RMS误差3.981442e−7，逐项容差检查全部满足 |
| 网络训练和权重 | 新网络训练步0、基础模型/Adapter forward0；未修改任何checkpoint。有限系数优化768次明确单列 |
| 评价 | 完整原图同检测IoU；原生N、同网格C、原S1和旧门控R_G按同一256条分别列出；未计算或声称新AP |
| 统计 | 243图宏平均，2000次图像配对bootstrap，seed1729，linear百分位；原分母不变。已暴露开发面板，不称独立确认 |
| 独立复算 | 两机冻结verify核验1792原图RLE、1280输入RLE、像素计数、系数转换、统计及原裁决均一致；本机NumPy2.3.5与执行端2.4.3复算在冻结比较精度内一致，原生推理环境始终为执行端原版本 |
| 结束状态 | 执行端已无本Run进程及Study锁；全部80项原始Run产物回传SHA一致，8,574,381字节 |

规范化运行事实：[原生回执](runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01/CONFIG_AUDIT.json)、[数值例外](runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01/NUMERICAL_EXCEPTION_LEDGER.json)、[回传核验](TRANSFER_VERIFICATION.json)。原始CONFIG_AUDIT的`status=passed`指配置一致，其中`all_numeric_proxy_checks_passed=false`这一汇总字段合并了求解收敛检查；逐条数据确认代理/等幅本身都通过，未过的是23次求解收敛。原文件不回写，由本审计说明精确含义。

工程问题保留：冻结前的账本类别字段读取错误见[预检失败](engineering/PREFLIGHT_V1_FAILURE.json)，改为同SHA原标注关联后通过；正式启动器首次硬编码powershell.exe不可用，未创建Run，见[FULL_LAUNCH_FAILURE](FULL_LAUNCH_FAILURE.json)。核实真实shell为PowerShell7后，用实际pwsh.exe路径启动；保留原空日志，科学命令、代码与协议未变，没有第二次正式实验。

Study在正式运行中曾沿用smoke的`scientific_status=engineering_only`与`artifact_status=complete`摘要字段；正式Run自己的`run.json`始终正确记录`unadjudicated/partial`。本次结案已统一Study终态，历史Run未覆盖。这是生命周期摘要问题，不作为科学证据。

数值未决保留，EK-v1不晋级；关于已收敛丢失导致最高保留率88.05%的补充界限见[REPORT](REPORT.md)，不替换原协议或DECISION。全部旧研究Gate与官方发布parity状态保持。
