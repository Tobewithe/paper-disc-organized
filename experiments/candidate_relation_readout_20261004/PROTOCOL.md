# 候选关系读取：其它预测响应的实例归属增量

2026-10-04，运行前固定。用户授权笔记本 `28358lan`；本机只编辑、记录和传输。

## 问题与依据

检验其它预测候选在当前实例空间位置上的响应，能否为系数生成提供可学习的归属信息。现有原型具有部分oracle恢复空间，但静态自身ROI、原型统计、迭代自身反馈和本轮原型共同微调没有给出正常任务收益；这些事实不证明“原型完全充分”或“系数头一定缺信息”。

依据：`../prototype_feedback_readout_20261003/runs/RUN_17e5267de70f4364a8fde3ecc4c7a60d/REPORT.md`、`../native_proto_coadaptation_screen_20261003/REPORT.md`、`../supervision_output_domain_audit_20261003/REPORT.md`；去重入口 `../../project/research_notes/READOUT_INTERVENTION_DEDUP_20261003.md`。

[K-Net (NeurIPS 2021)](https://arxiv.org/abs/2106.14855) 提供条件化实例kernel读取的相关研究背景；它不证明YOLO当前失败机制，也不构成本实验的新颖性保证。本轮只检验受控、有限实现，不复刻其完整架构。

## 不变量与样本

沿用笔记本 Ultralytics **8.4.100** 实际源码和官方 COCO YOLO26m-seg 权重。冻结上游三尺度F、原型P、框、类别、原有one-to-one TAL分配、BN running buffers。训练native one2one_cv4与新读取模块，GT仅参与原有监督及评价。没有新的排斥损失或互斥softmax。

原缓存INDEX的前1024张fit与前256张dev，均为COCO train2017历史开发图；不重复/不跨split。保留零正样本图的数量；其没有候选级掩码指标。固定seed0、3epochs、microbatch2/effective16、native lr1e-4/new lr3e-4、AdamW decay1e-4（bias/norm不decay）、warmup1epoch、cosine末端0.1、clip10、FP32、无AMP/TF32。继续使用既有官方full640 overlap-mask监督、GT框内area-normalized BCE与gain9.83241；不重写标签、不声称无实例背景监督。

## GT-free候选池和方法

对缓存F执行冻结的官方one-to-one box/class/coefficient分支，先核对既有目标的框/系数重放。按每个raw anchor的最大类别概率取前300，概率至少0.05；它是自定GT-free上下文池，不声称等于官方最终top-k输出。单raw只出现一次。对每个目标排除自身raw，从有正框交集的候选中，按score×boxIoU取最多4个；跨类别均保留，不使用GT决定邻居或剔除同实例重复。

在目标自己的16×16原型ROI算子A上计算其它系数的sigmoid(A_i c_j)，用邻居预测框对ROI bin中心形成支持，乘冻结score。聚合三通道：逐点max、按有效邻居总数的mean、逐点支持数/4。目标query、当前h/c、A/K和原框不变。使用已有64D融合ROI读取器，新增三通道输入；输出r=4tanh(raw/4)，仅一步 c'=c+K r。初始最后层为零，初始输出复现原模型。冻结关系图不反传到邻居或上游。

|组|含义|
|---|---|
|A|原权重，不训练，正常裁切重放|
|N|普通native系数分支微调|
|S|同样邻居集合/框/score，但每个邻居系数换成自身冻结c0，提供自身响应对照|
|T|真实其它候选响应|
|M|相同真实邻居，在每个邻居与目标ROI相交的离散矩形内分别roll半高/半宽，再聚合；支持、score及每个邻居响应值多重集不变|

S/T/M完全相同参数、初始化、优化步骤及预算。S包含邻居几何，T−S只检验**其它候选响应的增量**。M可能产生环绕接缝且小/常量支持无法改变；记录实际changed_fraction，不据效果补样或改扰动。T>M单独不足以证明因果机制。邻居可以是同实例重复，不能把任何重叠都叫泄漏。

## 评价与判断

固定最终epoch3，不选最佳轮。fit只评前128张用于拟合诊断，dev全部。原COCO annToMask、原预测框、原图normal decode；复核原A逐candidate IoU与历史记录一致。本轮是固定TAL候选的掩码质量诊断，**不是最终检出召回或COCO AP**。

主比较T−S图片macro IoU；同时T−M/T−A/T−N。报告框好掩码差（原BoxIoU≥.75且原MaskIoU<.75）、原成功/原失败、有邻居/无邻居；统计IoU、Mask75修复/误伤、coverage、FPR、BCE、1000次图片配对bootstrap；分组由A及GT预先固定。无邻居也保留，不掩盖方法可用范围。

短程正信号需：dev全体T相对S/M/A/N的macro IoU均正；并达到全体相对A至少+.2个百分点或目标组至少+.5个百分点；全体相对A不低于−.1个百分点；全体Mask75净修复非负。门槛是预设筛选条件，不等于统计显著或正式泛化结论。即使通过，也要独立图像/完整输出/原生AP另行确认。

若没达到，停止当前实现，不自动增加轮数、种子、超参数、loss或架构；负结果仅针对本实现与预算。保存预测池覆盖、同类邻居计数、M实际改变比例，避免把没有有效干预当机制反证。

## 执行与记录

prepare、smoke、N/S/T/M训练、evaluation各自Run ID；runner冻结配置及代码，每轮checkpoint保留，失败Run不覆盖。准备/冒烟分别30分钟上限，四组训练合计2小时，评价90分钟资源上限；这些是硬上限，不是预计时长。不触碰已有进程；本任务超预算/失败不自动重试。完成自动写SUMMARY/REPORT，启动本地回传器同步自身状态与Run（不执行额外研究）。
