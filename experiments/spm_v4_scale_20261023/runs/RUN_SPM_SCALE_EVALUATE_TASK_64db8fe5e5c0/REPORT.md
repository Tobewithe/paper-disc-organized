# SPM-YOLO V4.1 全量实例任务原型实验

本实验完成1272真实训练图、432真实开发图、2048几何/4096外观、4096 cutout，B* 120轮、原型20000步及四臂各8000步。尚未达到相对最强保留原生参照的+5点目标；原型特异性效用未共同获得支持。本结果为单seed开发集选优，未使用独立test。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| **问题** | 上轮54图Run的P3门控近零，关原型AP几乎不变；小幅整体收益不能证明原型帮助分割。本实验检验实例监督和直接高分辨率Mask读出能否纠正这一缺口。 |
| **机制假说** | H：fg/bg辅助目标缺乏实例归属监督，间接门控容易被绕开。跨外观实例匹配与直接Mask基底有望把学习原型变为有效输出。竞争解释为原型冗余，或任意额外特征/基底均可达到同效；F−G/R和同checkpoint定向破坏共同区分。 |
| **针对性方法** | 固定同一全量B*的P2+P3与归一化，512→256→128投影按5k真实/cutout、10k混合、5k真实训练；train-only fg16/bg16中心。F增加32原型基底/32零初始系数，保留原32基底及原生O2O/process_mask。B/G/R/F同8k步、batch12=8real+2cut+2syn、相同样本与增强序列，前1k仅Mask路径，后7k最终head，上游/投影/中心冻结。 |
| **整体结果** | F Mask AP 57.3570，相对同预算B +0.0417pp、G +0.1534pp、R +0.1175pp、B* +2.6772pp。Mask AP75相对B +0.1050pp、B* +2.9480pp；Box AP相对B +0.0002pp。 相对max(B*,B) +0.0417pp；原同预算主比较+5门槛未达到。正常减干预：off +0.0565pp；permute_channels -0.0861pp；shuffle_spatial +0.5941pp；random_dictionary +0.3705pp。尚未达到相对最强保留原生参照的+5点目标；原型特异性效用未共同获得支持。破坏实验只说明该已训练输出是否依赖分支，不独立证明因果机制。 |
| **研究决策** | 当前选定方向：`retain_best_actual_checkpoint_task_spm_and_stop_scaling_this_dictionary`。唯一下一步：`use_existing_same_checkpoint_interventions_and_G_R_controls_to_identify_the_supported_bottleneck`，本次未启动后续训练。goal保持未完成，保留全部阴性/损伤；不得将代码完成或小收益当作5点达成。 |

| 模型 | Mask AP | AP50 | AP75 | Box AP | 推理+解码中位毫秒 |
|---|---:|---:|---:|---:|---:|
| B*全真实基线 | 54.6798 | 83.7150 | 66.2551 | 64.4790 | 21.22 |
| B同预算续训 | 57.3153 | 84.9835 | 69.0982 | 66.4962 | 21.80 |
| G普通Adapter | 57.2036 | 84.9833 | 68.9479 | 66.4962 | 23.90 |
| R随机字典 | 57.2395 | 84.7828 | 68.9471 | 66.4962 | 23.80 |
| F任务原型 | 57.3570 | 84.6210 | 69.2032 | 66.4965 | 23.87 |

分母均为同432张真实dev、同640画布；Mask mAP50–95及差值单位为绝对百分点。AP来自全量官方COCO评价，分组与展示IoU不替代AP。检测类别/score路径在同F干预前后逐项一致；原生空Mask过滤可能移除不同输出，已单独记录。

| 原型干预（同一F权重） | Mask AP | 正常−干预pp |
|---|---:|---:|
| off | 57.3005 | +0.0565 |
| permute_channels | 57.4431 | -0.0861 |
| shuffle_spatial | 56.7629 | +0.5941 |
| random_dictionary | 56.9865 | +0.3705 |

| 文件名family | 图数/实例 | F−B pp | F−B* pp |
|---|---:|---:|---:|
| 1020s1220s2003-2 | 48/588 | -0.1716 | +6.2775 |
| 1050s1132a1110s3003-5 | 307/1614 | +0.4382 | +2.2900 |
| 1050s1132a1110s3004-2 | 46/92 | +0.0145 | +2.8233 |
| 1060a1000s1110s4001-2 | 31/569 | -0.4717 | +0.3506 |

文件名family不是已核实物理猪栏；它们是描述性开发分组。F/R有效可训练参数完全一致；G相对F的实际容量差为 9280，G/F并非严格容量匹配，R用于该控制。G/R共享同投影预训练信息，不能把新增P2信息本身全部归给学习中心。

原始数据仅使用原PigLife train的1704图分组切分；原test未参与训练、构库、阈值或模型选择。合成来自同一Quaternius猪资产；姿态、布局、密度、相机和外观随机化不等于真实分布完全覆盖。

运行、复核与证据：

- 冻结方法：[TASK_PROTOCOL_V3.md](../../TASK_PROTOCOL_V3.md)、[评价合同](../../EVALUATION_PROTOCOL.md)、[全量基线](../../BASELINE_PROTOCOL.md)、[最强原生参照目标验收补充](../../GOAL_ACCEPTANCE_ADDENDUM.md)。
- 总评价Run：`RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0`；源Run、检查点SHA与选优step见 [SOURCE_RUNS.json](../../runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/SOURCE_RUNS.json)。
- [全量指标](../../runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/METRICS.json)、[裁决](../../runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/DECISION.json)、[分组损伤](../../runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/FAMILY_RESULTS.json)、[真实贡献](../../runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/CONTRIBUTIONS.json)。
- 原生/方法所有训练、工程失败和先前近零门控阴性均保留；GPU评价步骤0，未追加训练/seed/网格。
- 训练和推理加载见 [USAGE.md](../../USAGE.md)，实际技术图见 [VISUAL_GUIDE.md](../../VISUAL_GUIDE.md)。
