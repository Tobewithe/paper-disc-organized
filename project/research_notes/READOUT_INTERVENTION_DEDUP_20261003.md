# 原型—系数链路：已做干预与两个未闭合问题

2026-10-03。训练等待期间的记录核对；不修改正在运行的快速筛选，不启动实验。按当前 AGENTS.md 保存。本表以实际报告及当前实现为依据；不同候选域、目标和解码的数字不串成一条方法排名。

| 问题 | 已实际改变什么 | 已得结果及范围限制 | 不应重复什么 |
|---|---|---|---|
| 固定表示下共享末层够不够 | 固定官方TAL的 h/P/标签，求原生64→32仿射末层并补条件最优性证书 | 6058个fit候选上的共享—独立目标差界为[0.885277,0.959143]；同参数val宏IoU下降0.985pp。证明的是固定函数类、固定目标，非整个系数分支或所有Mask75解不可达。[证书](../../experiments/coefficient_official_tal_affine_20260930/FROZEN_SHARED_CERTIFICATE_REPORT.md) | 再训练一个静态线性头，或把优化更多轮当作尚未检验的唯一解释 |
| 加入P或原mask条件是否足够 | 7H加入原生z0空间输入；PCDCR加入166维统计/几何条件与rank8动态读出 | 7H真实logit未优于错配；PCDCR相对静态头−0.0153pp且区间跨零，相对错配也无优势。限定于实际输入和训练设置。[7H](../../experiments/coefficient_raw_mask_condition_20260925/RESULTS.md)、[PCDCR](../../experiments/prototype_conditioned_dynamic_coeff_20261002/REPORT.md) | 再拼P统计、z0或增大低秩头，声称首次让系数依赖原型 |
| 框区域读取与原生分支联合训练 | 7J读取系数分支完整空间图ROI；后续BGCR及S/D同时训练原生cv4与框证据 | 7J未建立方向/IoU优势；正式joint全val S−N宏IoU−0.1660pp，S−A−0.4564pp，S−D区间跨零，但正确框优于错框。不能沿用较早冻结版本“投影优于直接残差”作为普遍结论。[7J](../../experiments/coefficient_spatial_head_information_20260926/RESULTS.md)、[joint正式结果](../../experiments/box_evidence_joint_training_20261003/runs/RUN_575d0d7a163343908c0fd8beccedc0f8/REPORT.md) | 再称首次使用预测框/空间ROI，或只靠解冻系数头、更多轮数解释失败 |
| 空间预测投影回原型是否可兑现 | 9/23间接与端到端投影；OGPS将既有ownership送同一Ridge求解器 | OGPS的GT8转换在框好mask坏组兑现有限oracle增益比93.65%；7O预测经solver优于直接图3.5569pp，却低于原模型1.0920pp。GT可行不证明无GT证据可用，投影也非任意输入必优。[投影训练](../../experiments/response_projected_coeff_20260923/REPORT.md)、[OGPS](../../experiments/ownership_solver_replay_20261002/REPORT.md) | 只把现成空间图接Ridge，或再端到端训练同类投影，作为全新命题 |
| 哪些空间观测更重要 | 稀疏GT提示；OGPS用GT修正8/64个高传播影响格；当前U/Q/P/L/B固定64格比较GT-free选择 | GT替换H8有明显恢复，但公平误差能量匹配仅覆盖42.27%，完整机制判定未通过。当前五臂正在比较均匀、不确定性、互补性及正负读取预算，结果未出。[误差传播](../../experiments/ownership_solver_replay_20261002/runs/RUN_ec575d4cc7a341a0bb98b8b5cff1cf63/ERROR_PROPAGATION_REPORT.md)、[当前协议](../../experiments/prototype_readout_fast_screen_20261003/PROTOCOL.md) | 把GT选点效应当可部署收益；当前未出结果就再开另一套静态选点/正负配额实验 |
| 一实例多套空间系数是否更好 | 四个2×2分区各输出32维残差，与普通系数MLP/空间头比较 | 全COCO点估计分区AP44.2292、普通系数44.2564；空间头AP优势很小，但小目标Mask75召回优于普通系数1.0286pp。单种子、参数不完全匹配；不能把平均AP接近等同其它指标无价值。[分区对照](../../experiments/coco_spatial_readout_controls_20260922/REPORT.md) | 把多区域多系数直接改名为新机制，或以所有指标须同时上涨否定特定失败改善 |

## 尚未被当前五种选点覆盖的机制缺口

**1. 更新后的mask能否改变下一轮读取，而非只提供一次条件？**

当前代码的选择始终由冻结c0决定，一次前向中的当前c只参与单次证据生成；7H/PCDCR也没有“生成c1后，根据Pc1重新读取F，再生成c2”的闭环。检索未找到这类已完成对照。[当前选点实现](../../experiments/prototype_readout_fast_screen_20261003/scripts/screen_models.py)、[点读出](../../experiments/prototype_guided_evidence_selection_20261003/scripts/point_head.py)。未解问题是初始错误响应是否把有效证据长期排除在读取之外。真正有判别力的是相同两次读取、共享同样参数/计算预算，第二次使用更新响应与继续使用原响应的区别，而非两轮比一轮；需看新读位置、原图修复/损伤，不能只看内部loss下降。这里是待检验的机制，不是宣布多迭代必有效；原始漏掉的区域不能因硬mask被永久禁止读取。

**2. 其它候选的实例响应是否提供单实例读出没有的“这是谁”的信息？**

当前N/U/Q/P/L/B及既有BGCR/S，按当前候选自己的query、预测框和P独立形成系数；共享图像特征不等于显式读取其它候选的query/响应。本次检索未找到已完成的跨候选条件交互训练。框真/错对照支持区域对应有用，却未证明是哪种像素混淆造成整体泛化差，因此只把此项列为缺口，不给它既成机制身份。[joint错框结果](../../experiments/box_evidence_joint_training_20261003/runs/RUN_575d0d7a163343908c0fd8beccedc0f8/REPORT.md)、[当前读出](../../experiments/prototype_guided_evidence_selection_20261003/scripts/joint_head.py)。有用的判别是保持自身P/query/框不变，同一交互容量下比较真实相关候选信息与关系错配信息，并核对收益是否发生在确有其它实例混淆的对象。邻居必须来自GT-free预测池，不能取TAL正样本列表；重复候选未必是不同实例，mask也不应被强制全像素互斥。官方实例BCE本来已区分当前GT与非自身像素，不能再把“BCE没有实例归属标签”当动机。

两项文献来由与用户此前已提出内容见[已有文献讨论](PROTOTYPE_TO_COEFFICIENT_LITERATURE_20261003.md)；本记录没有新增文献检索或新颖性断言。它们均改变信息读取的依赖关系，而非仅增加参数。当前先完成U/Q/P/L/B筛选；这里不授权自动新增训练、门控或损失，也不要求多个指标同时改善。
