# V23.1真实域微调整体结案

本轮实际完成V23已有候选2000步、原决策器4轮/1284更新、432正常RGB完整验收及本地实物交付。Mask AP60.738777，对原生+3.395277点、对V23-0.014501点；+5联合门槛未通过、护栏全部通过。正常推荐为V23，整个研究goal仍active。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| 问题 | V23真实完整AP60.753278，原生+3.409778、损伤18/2126；唯一经过全部护栏的正收益起点。旧训练real/syn各50%，13真实来源14–218图；V25/26另加质量头反而退化。待回答已有V23继续真实域学习能否产生额外效用。 |
| 机制假说 | H：从实际有益末权重进行有限真实任务适配和来源覆盖，比增大网络更值得优先检验；预测完整正常AP超过V23并保持护栏。H_alt：现有表示/内容可辨识性限制主导，额外曝光只过拟合或改变已有收益。配比、采样、学习率和预算同时变化，不作单因素或独立原型机制归因。 |
| 针对性方法 | 保留同一1406161参数候选和17552参数决策器、64地址/九角色/26动作，新增参数0。实际V23末state初始化；N/C21、框/类别/分数/排序/检测行冻结。6400real/1600syn共8000曝光、全部2552图，real图—来源混合采样；实际1592图/41059查询重建640成本，原风险器再4轮。固定末EMA及末轮，不挑中间权重，test426封存。 |
| 整体结果 | 完整432图/2863GT/12899行Mask AP60.738777，N+3.395277、V23-0.014501点；Box AP66.884503及全部12项统计/检测身份精确。Mask75损伤19/2126，修复73，尺度差[2.206215, 4.109632, 0.904918]；+5联合门槛未通过、护栏全部通过。独立72项误差0、逐GT50/75精确；12fresh、实际便携CLI、CPU重载及23图实际QA完成。 |
| 研究决策 | 本轮没有达到相对V23至少+.1点的替换条件，+5也未达，因此关闭该复合微调配置并保留原V23。−.014501点不称为已证显著损伤；AP75上升、AP50和密集AP下降说明并非所有质量区间同时改善，不能用较高AP75替代完整AP。 正向重点：继续保留实际正收益V23，分离已有候选适配与已有风险器适配的作用，找出可以保留的有益更新及二者的兼容性。 优先依据：同一轮实际改变了39个候选state张量和原风险器，而整体AP近乎持平、AP75与AP50方向相反，现有对角结果无法区分候选改好但决策抵消、决策改好但候选抵消或交互失配。两种交叉组合以实际旧/新权重为干预，不增加网络或训练，优先于无分解地再改学习率、扩训或重加质量头。旧V24有限池见证支持保留有益候选机会，但不保证正常决策可学或该微调有效。 唯一下一步：另冻旧/新候选×旧/新风险器的2×2零训练组件交换；复用两个既有对角结果，只新增两种交叉组合的完整正常评价，再裁决应优化或保留哪一段。 改判条件：交叉组合只有较原V23至少+.1点且所有原护栏通过才具备保留其组件的工程依据；若均未通过，则停止该复合重加权方向的继续扩训，保留V23，并以候选—正常决策的可辨识性缺口重新裁决后续设计。无身份/重放合法性则只修相应工程项，不宣称科学结论。 下一项训练未启动。 |

## 同配置完整正常结果

| 模型 | Mask AP | 相对原生/点 | 相对V23/点 |
|---|---:|---:|---:|
| 原生N | 57.343500 | +0.000000 | -3.409778 |
| 已有V23 | 60.753278 | +3.409778 | +0.000000 |
| 本轮V23.1 | 60.738777 | +3.395277 | -0.014501 |

门槛为62.343500243941705及全部原护栏。相对V23至少+.1且护栏通过仅为本轮工程替换条件，实际为False，不表示统计显著。重复使用的dev不能代替封存test的独立确认。

| 护栏 | 实际结果 |
|---|---|
| all_bbox_statistics_exact | True |
| original_Box75_damage_zero | True |
| Mask_AP75_non_decreasing | True |
| all_scale_AP_at_least_native_minus1 | True |
| native_Mask75_damage_at_most21 | True |

## 所有来源与密集子集

| 来源 | N Mask AP | V23 Mask AP | V23.1 Mask AP | 对V23/点 |
|---|---:|---:|---:|---:|
| 1050s1132a1110s3003-5 | 68.994190 | 71.484571 | 71.226917 | -0.257653 |
| 1060a1000s1110s4001-2 | 16.783189 | 17.944389 | 17.496848 | -0.447542 |
| 1020s1220s2003-2 | 62.797709 | 69.957397 | 70.600716 | +0.643320 |
| 1050s1132a1110s3004-2 | 85.131134 | 83.592028 | 83.023310 | -0.568718 |
| ≥15GT的31图 | 16.783189 | 17.944389 | 17.496848 | -0.447542 |

四来源描述性重抽样只比较N与V23.1：{'source_groups': 4, 'draws': 256, 'unique_count_patterns': 35, 'delta_AP_percentiles': [0.6056050361629275, 3.2618866257069357, 6.699208751891533], 'scope': 'source-group descriptive resampling on previously-used dev; not independent confirmation'}。不是V23与V23.1的显著性区间，也不是外部泛化保证。≥15GT并不声称每头猪都相互遮挡。

## 对象能力与计算成本

实际末模型相对V23规范字段概率MAE=2.17510436e-08，二值变化0/589824格。正常匹配及实际应用地址分别统计，详见ACTUAL_USAGE和图05/23；活跃地址数不等于真实具名部位语义成立。

- whole_IoU: {'value': 0.7358503229466957, 'queries': 2147, 'scenes': 62, 'scope': 'known train-seen synthetic normal outputs; not real semantic generalization'}
- macro_part_IoU: {'value': 0.3840400487157917, 'queries': 2147, 'scenes': 62, 'scope': 'known train-seen synthetic normal outputs; not real semantic generalization'}
- task_ROI_IoU: {'value': 0.8316737901465392, 'queries': 2147, 'scenes': 62, 'scope': 'known train-seen synthetic normal outputs; not real semantic generalization'}
- known_pose_correct: {'value': 0.9438636060535124, 'queries': 2041, 'scenes': 62, 'scope': 'known train-seen synthetic normal outputs; not real semantic generalization'}

已知域部位观测宏召回=0.576184868812561。这些来自固定64张train-seen合成图，单身体与代理姿态限制保留；真实具名部位、物理遮挡和独立语义泛化未补齐，不以AP替代。

本轮实际累计1.223889GPU小时、峰值2A6000，含失败工程；全部登记作业已终态。候选训练峰值CUDA3399649280字节；部署实际成本为{'seconds': 2.835887453984469, 'peak_cuda_bytes': 1422819328, 'GT_used': False}，正常暖帧计时为[{'image_id': 320, 'outputs': 19, 'seconds_normal_native_plus_V23_1': 0.8970207439851947}, {'image_id': 1269, 'outputs': 35, 'seconds_normal_native_plus_V23_1': 1.5345930590410717}, {'image_id': 798, 'outputs': 19, 'seconds_normal_native_plus_V23_1': 0.6777469469816424}]。文件大小、显存、延迟和GPU小时分开解释。

568标签键工程失败、570串行SHA取消、574首次UL导入与DDP同步失败、588便携导出漏带源码清单均原样保留；572/576/592分别核资产、工程同步和便携打包修复。576的8步丢弃，不计正式科学训练；578/582才分别是本轮2000候选及1284风险更新。失败、取消的调度终态不被原始中间run字段覆盖。

## 实物与复现

- [实际最终权重](C:/Dpan/codexproject/paper-disc-organized/artifacts/spm_v23_1/dcd4ba2c8e7d/deployment/weights/MODEL_FINAL.pt)：SHA `f35abe4f78b530b1d9b6eb9c564632f6ec91e55bbba72e67bba9c8bfb1d6cb34`，30542866字节，30.54MB；包含C21锚，无优化器。MODEL_FINAL是固定训练终点命名，效用由上述验收决定。
- [正常RGB入口](C:/Dpan/codexproject/paper-disc-organized/artifacts/spm_v23_1/dcd4ba2c8e7d/deployment/USAGE.md)：实际便携子进程逐值重放；原生依赖108413518字节。
- [实物清单](C:/Dpan/codexproject/paper-disc-organized/artifacts/spm_v23_1/dcd4ba2c8e7d/MANIFEST.json)：1254轻量文件+2实际部署权重逐项SHA；完整优化器/成本缓存按DEFERRED_SERVER_FILES保留服务器，未冒称回传。
- [最终核验](FINAL_ITERATION_DELIVERY_VERIFIED.json) / [实际看图](VISUAL_QA_VERIFIED.json) / [完整图册](VISUAL_GUIDE.md) / [冻结协议](PROTOCOL_V1.md) / [统一理论D22](../../paper/manuscripts/spm_yoloseg/SPATIAL_OBJECT_THEORY_SUPPORT_20261025.md#d22-用户指定回到有益v23同架构真实域微调)。

本轮训练、验收和交付完成不等于整个goal完成。+5与独立确认、具名部位/姿态/物理互遮及真实资产的剩余要求按DESIGN_COVERAGE保留。未推送GitHub、未建定时器、未读取test426。
