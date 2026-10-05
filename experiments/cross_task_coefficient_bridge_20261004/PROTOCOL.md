# 跨任务隐藏表示到原生系数分支：短预算对照

本轮只问：检测分支已经学到的内部表示，能否为原生系数分支提供可兑现的掩码信息？它不是再改变原型、空间取样、GT分配或损失。使用服务器3080Ti；笔记本归另一线程，本地桌面不运行模型。

## 依据、去重与反例

- 7I/7L已读取one2one_cv2末层前64维，研究失败识别/收益门控；不能说检测隐藏特征从未被使用。
- 7N的10,000图、71,269候选五折近邻已加入检测隐藏特征，好框坏掩码组作用方向余弦增量仅+0.0004，CI[-0.0084,+0.0082]。其直接target/gradient Ridge没有该检测块，系数训练仍h-only。因此不重复检索，而检查mask监督的原生分支联合学习。
- 20260911旧结构探针捕获过one-to-many cv3/cv2空间图，属于同图GT辅助像素可分性，不是本轮one-to-one同raw跨图系数生成。
- 8.4.100实际结构：同一F和raw网格，cv4系数隐藏64维，cv2框隐藏64维，cv3分类隐藏256维。启动时从模型读取核对。框输出64→4和分类256→80均压缩了表示，最终框/分数不能等同其隐藏特征。
- [CondInst](https://arxiv.org/abs/2003.05664)、[BlendMask](https://arxiv.org/abs/2001.00309)的[官方FCOS实现](https://raw.githubusercontent.com/aim-uofa/AdelaiDet/master/adet/modeling/fcos/fcos.py)已有从bbox_tower生成mask参数的先例。这支持尝试但限制新颖性；一般跨任务连接不是原创主张。
- 强反例：同类不同实例可以拥有相似分类表示；定位表示可能忽略内部孔洞/细部。三种表示来自同一F，本轮没有新增图像信息。即使真源优于错源，也不能自动认定缺失信息是唯一根因。

## 模型、对照和独立问题

全部保留32维系数与原始P，完整原生one2one_cv4参与训练，读取当前h而非旧h缓存。其他原模型参数和全部BN运行buffers冻结；cv4 BN affine可训。

\[
h'_i=h_{\psi,i}+U_l\frac{t_i-\mu_l}{s_l}+v_l,\qquad c'_i=W_lh'_i+b_l,\qquad M_i=P c'_i.
\]

U和v初始化为0。仅fit真实源特征按尺度估计mean/std，std下限1e-6，保存实际钳制通道；不PCA、不使用GT拟合特征压缩。原生前级卷积、末层与桥接同步训练；检测隐藏特征来自冻结原权重，不能随cv4更新。

| 臂 | 操作 | 用途 |
|---|---|---|
| A | 原始权重 | 原输出 |
| N | 已完成的原生cv4普通微调 | 复用同数据/预算/初始参数，不重训 |
| T | 当前raw的分类hidden256→64桥接 | 本轮主方法 |
| M | 同形分类桥接，改读另一图候选的hidden | 同参数量、同预算的来源对应控制 |
| R | 当前raw的框hidden64→64桥接 | 次要来源参照；与T容量不同，不作纯来源因果比较 |
| TW | 冻结T，评价时换成M来源 | 同参数源依赖检查，不训练 |

T/M新增49,344参数；R新增12,480，实际启动核对。N原生854,880参数。M/TW在每个split内固定查找另一图片、同尺度、同原预测类别的候选；类别来自原模型argmax而非GT。无法匹配时回退同尺度另一图，逐项标记。全体与严格同预测类别子集分别报告。它检验候选对应，不能区分所有空间/外观因素，不能宣称就是同图邻居归属能力。固定映射不随训练变化，fit/dev不交换源。

## 数据、目标和预算

复用fast-screen已审计的1024fit/256dev清单及官方one-to-one候选。候选身份含split/image_id/annotation_id/branch/raw_id/level/target_gt_idx；不重新TAL，不按质量过滤。dev为历史复用开发集，不是盲测。fit效用只评价原清单前128图，不冒充全部fit。

训练目标仅原官方实例mask BCE，完整P上采样640、原overlap标签归属、GT框crop、面积归一化与原gain；先候选求和再除当前16图累积组原候选总数。复用已验证的dense_runtime exact coefficient VJP、权重全1。无额外正则、教师、gate、阈值或新loss。

T/M/R都固定seed0、3epochs、micro2/effective16、原生及bridge lr=1e-4、AdamW(.9,.999,eps1e-8)、多维weight_decay1e-4/其他0、warmup1轮/余下cosine eta=.1、clip10。与N相同图像顺序、原生初值和更新次数。固定epoch3，不看dev选epoch。等更新不等实际梯度/计算量，均记录。准备900秒、smoke300秒、每训练臂1800秒、评价1800秒硬上限，超时保留失败不自动续训。无LR、宽度、损失或训练轮数搜索。

prepare/smoke/train_T/train_M/train_R/evaluation各独立Run，由runner包裹。准备缓存仅保存small hiddenfeatures/身份/映射/标准化，不传大旧缓存。启动前核原始Run和PID，禁止重复。

## 必要核验

1. hidden和原c0同raw，重建最终分类/框输出、P/输入未变；prepare源参数与buffers全哈希不变。
2. 零bridge逐候选复现原c0（atol=rtol=3e-5）；对相同loss，其native全参数梯度与N等价：abs1e-5/rel3e-4，整体相对L2≤1e-4。bridge梯度非零，临时一步后native及bridge改变；临时模型丢弃，不作为初始化。
3. 训练固定源与BNbuffers不变，bridge和native实际更新，原native初始化和每轮图像次序对应N；记录梯度、更新和残差幅度。
4. A/N dev五项历史逐候选指标复现，atol1e-12；使用既有完整P正常process_mask(upsample=True)和真实letterbox逆变换，不改解码。

## 指标、继续与结束

主比较T−A/T−N/T−M，主指标dev图片macro原图MaskIoU。R−A/N及T−TW为次要。候选均值、Mask75修复/损伤、coverage、AUC、FPR、原官方BCE一并报告。预先分原成功/失败、框IoU≥.75且原maskIoU<.75、P3/P4/P5；层级不等目标尺寸。所有区间整图配对bootstrap1000次，探索多比较不当独立确认。

预设有用信号：T对A和N均满足全体macro IoU≥+0.2pp、CI下界>0、净Mask75>0；或主失败组≥+0.5pp、CI下界>0且全体不低于−0.1pp。此外T−M的对应效用需正且CI下界>0，才支持当前实例源提供额外作用。其他指标的明确有价值改善可单独审查代价，不要求全面上涨，不以偶然子组掩盖整体损伤。

T>M但不胜A/N，只是使用对应信息，不是有效方法；T≈M提示本配置不能区分源对应与通用增加参数/训练路径。短预算路径幅度不足时不能宣称整类机制无效。R单独有信号也不能将T/R差解释为来源纯效应。无实用信号即停止当前配置，不扩宽、不加attention、不延长训练；本轮不做COCO AP或自动启动大数据确认。
