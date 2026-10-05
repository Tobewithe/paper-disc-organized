# 原生原型—系数适配：有界快速对照

## 问题和既有证据

刚完成的固定P静态选点与局部视图投影均没有得到值得扩大投入的净收益，不再修补这些结构。改检验：让原型本身适配当前one-to-one系数与监督对象，是否比只更新系数分支更有效？

8.4.100的Segment26向one-to-many传递带梯度proto，向one-to-one传递proto.detach()。这一事实已在coco_mask_spatial_probe_20260916/OFFICIAL_TRAINING_AUDIT.md:56–71记录，不是本轮的新发现。历史去重未找到同数据/目标下的原生proto-only和proto+cv4训练对照；旧冻结特征替换不等价，旧parameter_probe真正可训练的仅one2one_cv4。

本轮显式允许当前固定one-to-one实例mask损失更新原生Proto26，改变的是该监督到原型的梯度路由。这不是官方完整双分支训练复现，也不预先认定detach导致原模型失败。

## 四组与复用

| 组 | 操作 |
|---|---|
| A | 原始预训练权重，既有原图掩码参照 |
| N | 刚完成的fast-screen原生one2one_cv4-only，固定epoch3参数与逐候选评价直接复用，不重跑 |
| P | 原始初始化，只更新原生Proto26的实际mask路径；系数固定c0 |
| PC | 原始初始化，同时更新Proto26实际mask路径与原生one2one_cv4 |

不新建预测头、solver、gate或损失。backbone/neck/框/分类/候选assignment不变；全部BN运行统计固定原checkpoint，参与训练的原生模块BN仿射参数按N口径可训练。Proto26未参与当前mask输出的semseg支路冻结，无额外语义目标。

候选与数据严格复用fast-screen：1024计划fit/1012有效图/7172官方one-to-one候选，256计划dev/253有效图/1816候选，永久身份、标签、raw索引与框不重绑定。dev已被研究查看，只作探索筛选。

## 损失、预算与执行

训练完全沿用N的官方实例mask BCE目标：当前原型按既定bilinear规则到640，系数相乘，官方GT框裁剪/归一化面积/原增益，按有效batch候选等权；无位移正则、教师项或新loss。必须同时取得dL/dc与dL/dP并链回指定模块。旧缓存固定P的operator只能作为原始核验，不能作为P训练的梯度。

原始官方checkpoint初始化、seed0、3epochs、microbatch2图/effective16图、原生参数学习率1e-4、weight_decay1e-4、warmup1epoch、cosine终值比例0.1、gradient clip10，顺序与N一致。P与PC各自固定epoch3为唯一评价checkpoint，无dev选模。两臂合计最多40分钟训练预算，评价另最多20分钟；超时记未完成，不以提前停止模型冒充epoch3。

只在授权3080 Ti运行，桌面不跑模型。各自2张fit图smoke独立Run：初始化P/c与原缓存一致；固定P时目标/dLdc与旧N参考一致；完整小例官方loss与分块双梯度一致；小步后指定模块有有限梯度与参数变化，其余权重/buffers保持不变。真实通过后才开始正式训练。

所有正式执行使用runner，失败保留原Run，不自动重试。训练Run、smoke及评价身份各自独立，原fast-screen记录不覆盖。不增加训练轮数、样本或事后只保留某尺度。

## 评价和停止规则

A/N原图指标同永久身份复用，A用原完整P与c0重放核验。P/PC必须使用各自更新后的完整原型和系数，沿同正常原图解码评价，不能用缓存原P算新模型。实际原图IoU/coverage用原始COCO mask，AUC/FPR使用既定连续logit与固定预测框支持。

主比较P−N、PC−N；同时报告P−A、PC−A和PC−P。图片macro原图IoU为主，同时给候选均值、Mask75修复/损伤、coverage/AUC/FPR、原成功/失败及框好掩码差组，按图片配对bootstrap1000次。正常输出的框、分类、候选集合保持固定，不作COCO AP或独立确认主张。

继续投入参考量级与N同：全体macro IoU+0.2pp，或目标组+0.5pp且全体没有超过−0.1pp的明显代价；结合指标意义及区间判断，不要求所有指标同涨。没有有价值信号就结束本轮，不扫学习率/epoch或外挂新模块。如果有信号，只能称当前固定对象/预算下的原型适配作用，再独立规划确认；不能归因为原历史训练的唯一机制，更不能拿更新P的结果套用旧固定P证书。
