# 外部空间证据投影：正式结果与停止决定

**当前配置停止。** 冻结SAM ViT-B读取原预测框，其完整掩码相对原YOLO的图片macro IoU下降4.2256个百分点；压成8×8后经既有OGPS回投原YOLO原型，下降4.9903个百分点。投影优于直接渲染8×8网格，但没有超过原模型、SAM完整输出或同路径原模型自投影。预设继续条件未通过，不追加提示、模型版本、solver参数、gate或训练来挽救本配置。

执行与科学判断分别记录：Run成功完成、核验通过；研究方案未得到正向实用信号。总研究Goal仍未达成。

## 实际对象与实现

- Study：`STUDY_88774324e6f744dcae13436154d05633`；正式Run：`RUN_bb0160a72b9b4a93a43e5ffc7c4f3346`。
- 计划256张既有COCO train2017开发图，253张有效、1816个固定官方one-to-one TAL候选。无正样本图234138、284128、552870保留在清单。未按修正结果删除候选。
- 原模型为Ultralytics 8.4.100官方COCO预训练YOLO26m-seg；P、c0、预测框、分数、身份及真实letterbox几何冻结。GT只用于已有TAL对象与评价。
- SAM官方源码commit `dca509fe793f601edb92606367a655c15ac00fdf`；ViT-B权重SHA256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`。这是额外预训练模型和计算，不能称YOLO内部轻量等预算改进。
- 输入为缓存中同一640 RGB uint8，SAM内部官方1024预处理；原预测框单提示，`multimask_output=False`、连续logit返回640，每图一次embedding。未用GT框、类别、点击、SAM质量分数挑输出或迭代反馈。
- 原OGPS保持FP64、8×8、概率截断[.01,.99]、岭系数.003；优化系数增量后用原完整P及既定正常原图解码。不是BCE oracle，不计算跨对象的旧oracle恢复比例。

| 臂 | 操作 |
|---|---|
| A | 原c0与完整P正常解码 |
| BASE_SOLVE | 原完整640 logit→sigmoid→预测框pool8→固定OGPS→原P |
| SAM_FULL | SAM完整640 logit，固定原预测框支持及相同逆变换 |
| SAM_GRID | SAM概率pool8、clip/logit、8×8直接渲染 |
| SAM_SOLVE | 与SAM_GRID完全相同的q8和t8，经固定OGPS→原完整P |

## 主结果

以下变化均为**百分点**。原A的图片macro IoU为80.2592%。区间采用预先固定的1000次整图配对bootstrap；候选均值也整图重采样。复用开发集、多项描述性比较，不能当确认性显著检验或新盲测。

| 相对原A | 图片macro IoU变化及95%区间 | 候选平均IoU变化 | Mask75修复／损伤 |
|---|---:|---:|---:|
| BASE_SOLVE | −1.3153 [−1.5013, −1.1405] | −1.7393 | 15／108 |
| SAM_FULL | −4.2256 [−5.5229, −3.0225] | −2.4352 | 126／174 |
| SAM_GRID | −8.1986 [−9.6235, −6.8348] | −5.3069 | 91／251 |
| SAM_SOLVE | −4.9903 [−6.2118, −3.8625] | −3.9271 | 64／182 |

关键路径对照：SAM_SOLVE−BASE_SOLVE为−3.6750pp [−4.8572,−2.5243]；SAM_SOLVE−SAM_FULL为−0.7647pp [−1.1283,−0.4265]；SAM_SOLVE−SAM_GRID为+3.2083pp [+2.6017,+3.7987]。最后一项只说明本投影减轻了粗网格直接渲染的损失，不能单独作为有效方法结论。

SAM_FULL相对A的coverage／AUC／FPR分别−8.6384／−2.8629／−8.2735pp；SAM_SOLVE为−7.2680／−2.3764／−2.5525pp。误报减少是实际取舍，但伴随大幅覆盖与总体IoU损伤，未达到本轮预设实用要求。

## 框好掩码差对象

由A预先定义BoxIoU≥.75且MaskIoU<.75，共141图353候选。

| 相对A | 图片macro IoU变化及95%区间 | 候选平均变化 | 跨Mask75修复 | IoU下降实例 |
|---|---:|---:|---:|---:|
| SAM_FULL | −1.0237 [−3.1566,+1.1594] | +0.3915pp，区间跨0 | 100 | 145 |
| SAM_SOLVE | −4.1536 [−6.2150,−2.1082] | −3.3128pp | 44 | 217 |

此组原本全部低于Mask75，所谓“0损伤”仅指没有原成功候选可跌破阈值，绝不代表没有质量下降。不能用100个跨阈值修复掩盖145个实例IoU下降及macro主结果未改善。SAM_SOLVE对此组相对BASE_SOLVE亦为−2.3971pp [−4.6007,−0.2210]。

## 核验与解释边界

1. 原A五项指标与1816个历史同身份候选逐值复现，最大误差0；完整候选保留，未定义AUC/FPR各2例只对该指标计数，不删除IoU对象。
2. SAM参数及buffers前后相同，梯度关闭；1816个框均提示成功，fallback为0。输入资产前后hash一致。
3. 原logit不截断的自目标测试delta精确为0，原解码精确复现；正式solver最大normal-equation相对后向误差1.5318e−16。
4. P160池化与q640池化的整数边界存在已披露差异：仅14/1816候选离散支持完全相同，其余最大边缘差3个输入像素。该差异不能解释不经过solver的SAM_FULL下降，也不能据SOLVE下降证明原型容量不足。
5. FULL→GRID同时改变池化、概率截断和渲染，不是纯空间分辨率对照。BASE_SOLVE本身下降说明固定压缩/截断/岭回投不是无损操作；但SAM_SOLVE相对同路径BASE仍下降，因此不能把全部负结果归咎于共同处理。
6. 结论只针对当前SAM ViT-B、固定框单输出和旧OGPS配置；不否定所有外部模型、提示方式、无GT空间信息或原型表达能力。没有训练，不存在本轮教师蒸馏有效性的证据。

## 执行、资源和回传

本轮只在3080Ti服务器执行，未连接笔记本，桌面只代码／官方资产文件中转／结果记录。两次Git下载分别因HTTP2及HTTP1传输故障失败，独立Run日志保留；随后按同一官方commit及权重通过HTTPS文件中转，服务器独立验证后smoke与正式评价通过，研究条件未改。

正式评价122.28秒，SAM编码28.64秒、框提示5.37秒、solver9.88秒、指标34.89秒；峰值GPU allocated2.728GiB、reserved3.047GiB。这是本批次计时，不是完整部署延迟。pipeline于2026-10-03T17:34:01.748915Z完成。

五个Run共76文件SHA256核验回传，collection `COLLECT_620c935fd3eb4fb78c79661b1125d8b1`。每个Run有`transfer.json`和`manifest.sha256`。SAM模型／源码及逐图q8、系数回放张量保留服务器，远端路径及hash已在回传记录中列明；不声称大资产已全部本地化。

- [原始结果与全部分层](runs/RUN_bb0160a72b9b4a93a43e5ffc7c4f3346/REPORT.md)
- [机器汇总](runs/RUN_bb0160a72b9b4a93a43e5ffc7c4f3346/SUMMARY.json)
- [执行核验](runs/RUN_bb0160a72b9b4a93a43e5ffc7c4f3346/AUDIT.json)
- [协议](PROTOCOL.md)、[结果读取前的规则澄清](DECISION_CLARIFICATION.md)、[回传状态](CURRENT_REMOTE_STATE.json)

<!-- research-note: {"id": "DECISION_external_sam_projection_20261004", "kind": "decision", "title": "停止当前SAM框提示经固定OGPS回投配置", "scope": {"target": "253图1816个固定官方one-to-one TAL候选", "metric": "正常原图macro Mask IoU及Mask75修复损伤", "dataset": "既有train2017开发清单256图", "protocol": "experiments/external_spatial_projection_20261004/PROTOCOL.md"}, "verdict": "reject_current_configuration", "coverage": "declared", "evidence": [{"path": "experiments/external_spatial_projection_20261004/runs/RUN_bb0160a72b9b4a93a43e5ffc7c4f3346/SUMMARY.json", "revision": "db636e1fc9884d94bd00166eb4e30382a03bf33ec3f06397ad552a6f47888508", "relation": "derived_from"}], "limitations": ["复用探索集，不是独立盲测或COCO AP", "固定SAM ViT-B单框提示及旧OGPS，不否定所有外部空间信息", "投影受池化、概率截断、离散支持及固定岭正则共同影响"]} -->
停止当前配置，不扫提示、阈值、λ或自动训练；下一研究问题须另有依据。
<!-- /research-note -->
