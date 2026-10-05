# 外部冻结空间证据→原型系数：零训练兑现检查

本轮问题：独立预训练模型产生的无GT空间信号，能否经现有固定OGPS求解器在原YOLO原型中兑现为正常原图掩码收益？新增变量是空间信息来源。历史[去重与原文审查](../../project/research_notes/EXTERNAL_SPATIAL_EVIDENCE_REVIEW_20261004.md)未发现该外部信号的已执行记录。原YOLO局部视图、neck读取及选点已做过，不重复。

使用官方SAM ViT-B冻结权重，来源官方repository与其下载链接。额外外部预训练、编码器及推理成本单列；引入SAM/框提示不是新颖方法。原论文已用预测检测框提示SAM，存在部件/整体歧义与数据标注习惯差异。不会把阳性解释成YOLO唯一根因、等预算提升、已能蒸馏或已实现正常推理AP收益。

## 冻结对象与五臂

复用服务器fast-screen dev256清单、原官方one-to-one TAL候选（预期253有效图/1816候选），同P、c0、预测框、分数和真实letterbox几何。每个候选永久7键不变。原GT仅用于事先既有TAL身份和本轮评价，不进入SAM、求解器、框提示或输出选择。输入SAM为与YOLO相同的缓存640 RGB，内部1024预处理按官方；每图一次embedding，batch8框，multimask_output=False、return_logits=True，不给额外点/标签/反馈/GT挑选。

| 臂 | 输出 |
|---|---|
| A | 原YOLO c0与完整P正常解码 |
| BASE_SOLVE | 原完整640 logit→sigmoid→预测框pool8→同求解器→原P |
| SAM_FULL | SAM返回640连续logit，按相同预测框crop，零阈值，真实inverse |
| SAM_GRID | SAM_FULL sigmoid→同pool8→clip/logit→旧8格直接渲染，同crop/inverse |
| SAM_SOLVE | 与SAM_GRID完全相同q8/t8→旧OGPS→原完整P正常解码 |

`crop_pool_7o`是floor/ceil框裁剪+adaptive average pooling，不是ROIAlign。A8来自原P的160网格，q8来自640输出网格；二者映射同框但栅格不同，保留原实现差异，不能将回投损失直接归为P容量。`min_delta ||A8(c0+delta)-t||²/(2*64)+.003||delta||²/2`，t=logit(clip(q,.01,.99))，FP64固定lambda，无扫描/新loss/solver。尤其BASE也必须pool(sigmoid(full_logits))，不能用旧sigmoid(A8c0)冒充同路径对照。

有限退化/与640无交集框保留：SAM_FULL退回A原连续logit，SAM_GRID和SAM_SOLVE用原同路径q8，SAM_SOLVE=BASE_SOLVE，计数source_unavailable。nonfinite原资产直接fail，禁止造值/删候选。正常框不按效果筛除。

## 预检与预算

独立prepare/smoke/evaluation Run。下载源码与权重在服务器900秒预算内，保存commit/文件SHA/权重SHA及来源；源码只加路径导入，不升级Ultralytics或Torch。SHA是资产身份，不冒称官方发布校验。

smoke前2有效图300秒；formal全部256图3600秒，超时保留不完整，不按前缀形成结论。CPU6线程，FP32 SAM无autocast，CUDA3080Ti，峰值显存和embedding/解码/投影/评价时间单列。桌面只代码与传输，笔记本不访问。

核验缓存RGB生成链、SAM box-transform640→1024坐标、return_logits真正回640；source只接GT-free字段；权重与buffers前后一致、无grad；c0/P/box/input不变。原A五指标逐身份复现历史，tol1e-12；原型logit自目标solve_ogps_logits精确delta=0；普通解normal-equation relative backward error<=1e-10。保留所有候选和无正样本图，AUC未定义计数。失败先查明确执行故障，独立Run重试，不改研究条件或宽容差。

## 指标、判断及停止

主比较dev SAM_SOLVE−A、SAM_SOLVE−BASE_SOLVE；同时SAM_FULL−A、SAM_GRID−SAM_FULL、SAM_SOLVE−SAM_GRID/SAM_FULL分辨新来源质量和池化/clip/render/投影的损失。FULL→GRID同时改变池化、截断和渲染，不是纯分辨率效应。

正常原图macro IoU为主；候选均值、Mask75净修复/损伤、coverage、固定框支持连续AUC/FPR一起报告。原成功/失败/BoxIoU≥.75且MaskIoU<.75由A固定。整图配对bootstrap1000 seed20261004；复用探索集，多比较区间描述性，不作新盲测/确认性显著宣称。

继续信号：SAM_SOLVE同时超过A和BASE_SOLVE，整体macro IoU≥+.2pp且对应区间下界>0、净Mask75>0；或主失败组≥+.5pp且区间下界>0，同时整体相对A不低于−.1pp且损伤代价明确。信号只支持单独规划独立确认，不自动训练新头或蒸馏。

SAM_FULL好而SOLVE差：定位当前网格/clip/求解传递限制，不能宣布P容量不足；FULL本身无益：停止此教师/提示配置，不扫模型版本、提示或阈值。区间宽记不确定。不生成新GT oracle、不报告旧对象恢复比例，不增加gate、loss、样本或后续训练。正式结果报告与核验回传后结束本Study，总Goal不因此完成。
