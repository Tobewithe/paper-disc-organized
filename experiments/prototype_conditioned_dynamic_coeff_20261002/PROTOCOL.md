# PCDCR Stage I：冻结模型的读出方法对照

用户执行定义以 [原始指令](USER_EXECUTION_INSTRUCTIONS.md) 为准；实际常量集中在 [RUN_CONFIG.json](RUN_CONFIG.json)。Stage I 不重做此前的可达性诊断。已有证据限定于相应目标/安全内集，不能预先当作原型条件缺失的证明。本轮方法比较才检验条件化作用。

## 对象与训练

复用官方 YOLO26m-seg / Ultralytics 8.4.100 one-to-one TAL 缓存。原始 checkpoint、全部 h/P/c0、框、类别和 assignment 固定。fit/dev/val 为 796/197/196 图、6058/1402/1346 候选；逐身份核对已有官方 manifest。val 仅用于最终评价，但曾参与历史研究，称保留集评价，不称全新盲测。

三个训练组从等效原输出初始化，seed=0、12 epochs、4 image/batch、AdamW lr=0.001、weight_decay=0.0001、cosine 到0.0001、FP32、无增强。无增强保证缓存和 raw 身份不变。每批官方实例 mask loss 按候选等权平均，保留缓存 segmentation_gain；总损失加 0.003*mean(||delta_c||²)，没有1/2，不把参数weight decay冒充该正则。

训练监督使用原官方 overlap 标签、GT框支持与面积归一化；原图质量使用原始 COCO annToMask。两种标签用途分别保存。A完全不训练；B Static hidden123；C Concat hidden45；D rank8，每层独立。参数数 B=35889、C=35601、D=35928。所有原生参数和buffers保持冻结。

## 条件和初始化

e固定166维，顺序为 fg32/bg32/unc32/response32/c032/预测框宽高面积3/尺度one-hot3。支持区为640输入网格上的原预测框，完整原型按官方bilinear align_corners=False插值。空预测框仍保留，四种池化描述为零，分母固定epsilon=1e-6；不得删候选。

D=U(sigmoid(G(e))*Vh)，U零初始化，alpha固定1。首步U有梯度，后续V和G才有梯度；smoke检查这个真实的梯度过程。B/C输出层同样zero-init。

M错配e来自同分片、另一图片、同尺度，优先同预测类别，以预测框面积最近5项中固定seed取1项。GT类别、面积和失败标签不参与错配选择。完整匹配清单在训练前保存。

## 模型选择与对照

每轮用正常原图dev图片macro IoU选checkpoint，候选包括epoch0，平局选最早。三组完全相同，不看val选epoch。固定一个lambda，不做搜索。单标量bias在dev的[-2,2]、步长0.25网格选择图片macro IoU最高者；平局绝对值最小，再取较小有符号值；val冻结。bias在固定预测框支持内施加，b=0逐像素复现原输出。

## 评价与放行

主比较D-B、D-C；机制D-M；参照D-A、D-S。主指标图片macro原图Mask IoU，候选均值也以整图bootstrap5000次给95%CI。Mask75、修复/损伤、coverage、连续logit AUC、预测框内背景FPR同时报告。P3/P4/P5、A组成功失败、COCO面积small/medium/large仅作解释。

用户提出的实用量级为D-B至少0.002，或Mask75一致向好；D-C和D-M须为正，D-A须改善，且damage不显著增加。置信区间跨零的比较保留不确定性，不把微小正点估计当成已建立机制。D若不能超过单bias且AUC基本未动，不宣称空间排序改善。是否满足Stage II另在本轮结果表显式判断，失败或证据不足即停止；不自动加rank/特征/模块。

## 执行和范围

仅笔记本28358lan执行GPU实验，原主机只编辑及接收记录。8张fit图smoke检查官方身份、原末层系数复现、历史原图baseline、条件finite、零初始化、后续梯度和冻结参数、正常解码。正式训练累计预算12小时，smoke单列；预估超预算先停。全量缓存按图准备，训练只载入fit/dev的低分辨率原型，不驻留640原型全集。

交付含逐候选和逐图结果、所有选模轨迹、错配表、bias选择、参数量和资源。此轮为固定TAL候选方法验证，不报告完整COCO AP。
