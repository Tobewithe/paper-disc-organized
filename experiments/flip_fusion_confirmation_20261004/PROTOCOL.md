# 冻结翻转融合的方法开发外确认

本轮只确认上一轮普通MIX的预设失败组信号能否重复，不开发新系数网络。原投影PROJ已停止；不扫ridge、融合比例、框匹配条件、变换或阈值。

## 对象和新增证据

[原筛查](../flip_response_projection_20261004/REPORT.md)在253张已用开发图、1816个官方one-to-one候选上，MIX的框好mask差组macro IoU +0.5483pp，CI[0.0998,1.0515]；全体+0.0852pp不确定，29修复20损伤且coverage下降。只有局部初筛信号，不是已成立方法。本轮新增的是同一冻结计算在未参与其配置选择图片上的复现，不能按换图改名宣称新机制。

方法沿用增强一致性文献的普通双视图思路，[SAC](https://openaccess.thecvf.com/content/CVPR2021/papers/Araslanov_Self-Supervised_Augmentation_Consistency_for_Adapting_Semantic_Segmentation_CVPR_2021_paper.pdf)与[SEAM](https://arxiv.org/abs/2004.04581)只提供先例，不保证本对象成立；本轮不宣称原创。

## 数据与独立性

前瞻固定1536张COCO train2017，清单见CONFIRM_IDS.json，排除来源、算法、可知范围和未知范围见EXPOSURE_AUDIT.json、EXCLUSION_IDS.json。先对历史清单去重，再以固定seed=20261004和image_id的hash排序选取；不依据GT目标大小、失败率、预测或指标选图。冻结前不在新清单上试方法。

该范围只称“本次冻结方法开发外确认”。官方COCO预训练见过train2017，早期本项目也进行过全train2017其他模型训练；不能称主模型未见、全项目未用或全新盲测。明确清单核对不能排除所有未索引的历史查看。图像排除审计是条件性证据，不制造绝对独立性承诺。

POWER_PLAN.json由旧逐图差值规划：目标组0.5pp相对零、双侧alpha .05、90%功效需约397张目标有效图；出现率及长尾方差的保守敏感性组合约需1304张计划图，预先固定1536。它不是证明真效应至少0.5pp的功效保证。完成既定清单后停止，不因显著性、候选数或效果追加图片。

## 仅两个冻结臂

A是原官方COCO YOLO26m-seg / Ultralytics8.4.100。原输入按已核验YOLODataset生成RGB640canvas；无增强、FP32 eval、batch1、全部参数和BN buffers冻结。对原始权重执行一次官方one-to-one TAL，保留split/image_id/annotation_id/branch/raw_id/pyramid_level/target_gt_idx的完整身份，按官方转换排重和overlap排序绑定原COCO实例。TAL以后不随MIX重分配。

MIX完全复用原FlipSource：整张640输入（包括padding）水平翻转，翻转P与c配对组成完整连续logit，再逆翻。翻转source在同层、同原预测argmax类的全部raw中按逆翻预测框IoU最大、再分数、再raw_id选择；无候选时镜像raw回退并记录。无top-k/conf/GT选源，低分源不删除。融合z=(z0+zf)/2，固定alpha=.5。最终使用原预测框裁剪、zero-logit二值化和同一真实letterbox逆变换。原框/分数/候选身份不变，无训练、投影、校准、oracle或新增gate。

MIX施加于全部官方候选，不能用GT定义的失败组决定修不修。框更像不等于mask来源更好，旧MATCH反例仍成立；本轮不事后改选源方式。

## 必要核验与预算

smoke为旧dev前2张已有图，独立Run，240秒上限：由原始图片重建官方身份并核对历史A/MIX的五项指标误差不超过1e-12，c/P/box replay仍用原atol=rtol3e-5。这2张只用于接口等价，不纳入确认。

正式Run为冻结1536张，数据准备、原前向、融合与统计总共1800秒上限。流式处理，不迁移旧大tensor缓存，不在本地执行模型。每个A的连续logit解码必须逐像素复现官方process_mask(upsample=True)；所有候选保留，空掩码、无正样本图片、AUC未定义单独记录。新图没有历史结果，不编造基线重放证据。只修明确执行故障，失败Run日志保留，重试需独立Run；超预算不缩样本或延长来追求阳性。

## 指标、统计与决定

唯一主要效应：由A预先定义BoxIoU>=.75且原图MaskIoU<.75的候选，MIX-A图片级macro原图MaskIoU。每图先平均该组候选，再平均有该组的图片。配对bootstrap5000次，seed20261004，整图重采样；候选平均同样以图片重采样。95%区间与点估计必须同口径。次要对照/分层区间为描述性，不择优替代主要指标。

同时完整给出全体、原成功、原失败、P3/P4/P5及COCO大小的IoU、Mask75、repair/damage、coverage、连续logit的框内AUC和FPR、空mask及连续IoU升降/超过1pp的升降。原COCO mask是质量真值；层级不等于大小，失败组零damage不等于安全。

额外只标注原raw是否出现在官方one-to-one head top300且conf>.001中、输出类别和GT是否相同、rank；列出retained_anyclass、retained_exactclass及exactclass且GT类正确的次要分层。该输出范围只是候选→输出保留联系，不覆盖全部推理FP/漏检，不报告COCO AP或完整部署收益。

- 若目标组估计>=0.5pp且95%下界>0，同时全体IoU差95%下界>−0.1pp，保留“有实际量级的目标组信号重复且未见超出所定幅度的总体损伤”；不等同证明真收益至少0.5pp。Mask75、coverage等代价必须并列判断。
- 若目标为正但低于量级、全体护栏不通过或原成功存在明显代价，限制结论，不能凭目标组阳性宣布方法成功。
- 若未复现或区间仍宽，固定预算结案，不继续补图/选权重/新gate以挽救。
- 无论结果如何，普通MIX不成为新的系数方法；PROJ保持停止。只有实际结果另行支持时才规划进一步工作，不自动训练一致性头或完整AP。

输入/源码/配置/协议在启动前冻结，runner与SHA回传复用现有流程。本线程仅3080Ti服务器，笔记本归另一线程。完成报告、回传、记录后结束本轮，总研究Goal另行保留。
