# PCDCR Stage I 实验结果

**结论状态：STOP_STAGE_I。至少一个已锁定数值条件未满足；PCDCR第一版在本轮结案。**

组别：A=原模型；B=参数量匹配的静态MLP；C=普通条件拼接；D=PCDCR动态读出；M=同一D模型使用错配条件；S=开发集选定并冻结的单标量logit校准。D−B表示动态读出相对静态容量对照的差值。

**Q1：** D−B的95%区间跨越或接触0，当前比较不确定；不能把点估计方向当作已建立优势。 锁定实用量级为0.200 pp，Mask75替代条件单独列出。 B在dev选择了零初始化epoch0，因此本轮所选B输出等于A；该对照反映当前训练预算和选模规则的结果，不能代表静态函数类的最优水平。

**Q2：** D−C的95%区间跨越或接触0，当前比较不确定；不能把点估计方向当作已建立优势。

**Q3：** D−M的95%区间跨越或接触0，当前比较不确定；不能把点估计方向当作已建立优势。 即使为正，也只支持整套实例条件的对应关系有作用；e同时含P统计、c0和几何，不能据此证明prototype是唯一缺失变量。

**Q4：** 目前未形成同时超过单bias与提高连续像素排序的明确证据；不宣称已识别空间组合机制。 这不能排除所有其他校准方式，也不是完整oracle恢复证明。 实测D−A AUC为-0.0100 [-0.0188,-0.0011] pp；D−S原图macro IoU为-0.0645 [-0.1564,+0.0260] pp。 本轮AUC区间为负，没有支持像素排序改善。 D未建立超过S的优势，但这不证明D与阈值校准完全等价，也不能说其行为被阈值完整解释。

**Q5：** D相对A修复0个、损伤3个，净变化-3个Mask75候选。未检测到损伤率显著增加，但区间跨0不等于已经证明非劣性。

**Q6：** 至少一个已锁定数值条件未满足；PCDCR第一版在本轮结案。

## 主要比较

所有效应及区间均为百分点（pp）；主统计量是val图片macro，候选均值是辅助。95%区间为整图配对bootstrap，未作多比较校正。

| 比较 | 原图macro IoU差 [95% CI] | Mask75差 [95% CI] | AUC差 [95% CI] |
|---|---:|---:|---:|
| D−B | -0.0153 [-0.0685,+0.0506] | -0.2628 [-0.6494,+0.0000] | -0.0100 [-0.0188,-0.0011] |
| D−C | +0.0241 [-0.0227,+0.0775] | +0.4267 [-0.3745,+1.6767] | +0.0278 [+0.0126,+0.0466] |
| D−M | -0.0036 [-0.0097,+0.0023] | +0.0729 [+0.0000,+0.2187] | -0.0010 [-0.0037,+0.0009] |
| D−A | -0.0153 [-0.0685,+0.0506] | -0.2628 [-0.6494,+0.0000] | -0.0100 [-0.0188,-0.0011] |
| D−S | -0.0645 [-0.1564,+0.0260] | -0.1174 [-0.8570,+0.7234] | -0.0100 [-0.0188,-0.0011] |

## 六组原图质量

| 组 | 图片macro IoU | 候选IoU | Mask75候选数 | 修复/损伤（相对A） | Coverage | AUC | 框内非目标FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 79.9036 | 74.8540 | 831 | 0/0 | 91.3561 | 94.8710 | 21.5188 |
| B | 79.9036 | 74.8540 | 831 | 0/0 | 91.3561 | 94.8710 | 21.5188 |
| C | 79.8642 | 74.7987 | 825 | 4/10 | 90.9807 | 94.8332 | 20.9215 |
| D | 79.8883 | 74.7971 | 828 | 0/3 | 91.2539 | 94.8610 | 21.3734 |
| M | 79.8918 | 74.7977 | 827 | 0/4 | 91.2591 | 94.8620 | 21.3745 |
| S | 79.9527 | 74.8643 | 838 | 21/14 | 90.1201 | 94.8710 | 19.1164 |

FPR人群为预测框内所有非目标像素，包括邻居实例，不能单独解释为纯背景泄漏。

## 成功实例损伤

共同人群固定为A原图MaskIoU≥0.75；damage_X=1[MaskIoU_X<0.75]。正差代表D损伤更多。

| 比较 | 图片macro损伤率差 [95% CI] | 候选损伤率差 [95% CI] |
|---|---:|---:|
| D−B | +0.3366 [+0.0000,+0.7811] | +0.3610 [+0.0000,+0.8188] |
| D−C | -0.7767 [-2.0807,+0.0107] | -0.8424 [-1.6490,-0.1188] |
| D−M | -0.0748 [-0.2280,+0.0000] | -0.1203 [-0.3846,+0.0000] |
| D−A | +0.3366 [+0.0000,+0.7811] | +0.3610 [+0.0000,+0.8188] |
| D−S | -1.2270 [-2.5332,-0.3234] | -1.3237 [-2.2250,-0.5000] |

未检出增加不能代替非劣性证明；本轮没有事后新增损伤容忍界。锁定文本未明确损伤率比较基准，因此保留全表；若主要比较已经不通过，直接按该已确定失败条件结案，不以这项定义歧义代替失败原因。

## 训练、选择和运行核对

实际val：196张图 / 1346候选；协议范围196 / 1346。
训练：seed=0，epochs=12，lambda_delta=0.003。训练损失为官方同候选mask项 + lambda_delta·mean(||delta_c||²)，正则没有1/2。无oracle教师项。
正式训练耗时：0.073小时；预算12小时。
实际导入Ultralytics：8.4.100；路径：D:\coco_wire\py\ultralytics\__init__.py。运行记录中的发行包元数据：[{"location": "run_record.environment.execution.packages.ultralytics", "version": "8.4.27"}]。发行包版本（如8.4.27）与sys.path实际导入源码（如8.4.100）可不同，以核验的实际导入源码为执行依据。
历史baseline逐候选复核：[BASELINE_REPLAY_AUDIT.json](BASELINE_REPLAY_AUDIT.json)。审计记录：{"comparison": "PCDCR A normal-original-image IoU vs historical official TAL baseline on identical keys", "scope": "saved results only; no new inference", "candidate_count": 2748, "matched_count": 2748, "missing": [], "absolute_tolerance": 1e-06, "maximum_absolute_iou_error": 0.0, "outside_tolerance": [], "passed": true, "sources": {"D:\\coco_wire\\pcdcr_20261002\\source_PER_CANDIDATE.jsonl": "e9d11a6037ed63ba2c01ea25d7a5efda4b4b2186032f19096f05c96cf58c1a06", "D:\\coco_wire\\pcdcr_20261002\\runs\\RUN_stageI_seed0_20261002\\PER_CANDIDATE.jsonl": "5283d8f1367c855d06b9adb71ebb53d855392e82c0887607c5473b20c3c85f0e"}}。

| 组 | 参数数 | dev选择epoch | dev macro IoU | 训练记录轮数 |
|---|---:|---:|---:|---:|
| B | 35889 | 0 | 80.3027 | 12 |
| C | 35601 | 1 | 80.3301 | 12 |
| D | 35928 | 2 | 80.3195 | 12 |

S固定bias：-0.25；选择规则：max dev image-macro original-image IoU; tie smallest abs bias then signed value。
S−A AUC浮点审计：{"theory": "A shared additive scalar leaves pixel ordering and ROC AUC unchanged in exact arithmetic", "floating_point_note": "Actual FP32 logit addition may create ties. Measured S-A AUC differences are retained, never overwritten with zero", "defined_candidates": 2745, "undefined_or_not_evaluated_candidates": 3, "nonzero_difference_candidates": 94, "max_absolute_difference": 1.3656160979813592e-08, "mean_absolute_difference": 3.3661081997049053e-11, "mean_signed_difference": -1.5258397925037972e-11}。
条件计算均值：2.4129 ms/candidate（包含完整原型上采样及ROI池化；不含主网络、IO及H2D）。
D adapter均值：0.7416 ms/candidate。以上均不是端到端延迟。
协议没有给出‘明显过高’的数值截止值，因此不基于结果临时造一个延迟门槛。

## 分层与条件行为

P3/P4/P5、A原成功/失败及COCO面积分层保留在RESULTS.json，仅作解释，不能以有利分层替代val全体。完整逐候选记录在PER_CANDIDATE.jsonl，逐图值在PER_IMAGE.json。
门控各rank通道、true/mismatch gate差异与系数增量分布写入DECISION.json的intermediates；这里的sigmoid门控是动态读出内部结构，不是另加失败选择gate。

## 结论范围与结束条件

- 本轮没有oracle教师，只比较官方同候选mask监督和系数增量正则下的外挂读出结构。
- 结论限于本次166维复合条件、rank=8、seed=0、12轮训练和已冻结的样本/选模方案；不能据此判定所有prototype条件机制不存在。
- GT仅用于训练监督与离线评价；class_id和原成功/失败标签属于评价元数据，未进入condition。
- 固定官方TAL候选由GT条件的分配流程确定；本轮没有评价完整推理输出，也没有COCO AP结论。
- val未参加本轮训练或checkpoint选择，但已在历史研究中使用，因此不是从未查看的新盲测。
- M替换整个166维e，包含P统计、原系数c0和预测框几何；即使D优于M，也只能支持这套复合条件的实例对应关系，不能单独证明P是缺失信息。
- FPR统计预测框内的全部非目标像素，其中包含邻居实例；不能将其单独归因为纯背景泄漏。
- 旧固定表示共享仿射证书只提供研究背景：其正则为lambda/2乘系数位移平方，本轮为lambda_delta乘系数增量平方，且函数类不同，不能拼接目标值或计算跨实验回收比例。
- 旧证书不预先证明PCDCR有效，也不识别YOLO历史训练失败的唯一原因。

完整机器判定和来源SHA256见DECISION.json。脚本只整理现有冻结结果，不训练、不改checkpoint、不改bias，不执行Stage II。
