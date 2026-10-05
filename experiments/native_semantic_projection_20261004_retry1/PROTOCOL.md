# 原生类别语义响应回投系数：冻结快速筛查

## 本轮问题、历史与决定

新增信息是官方预训练 Proto26 的原生语义分支在候选预测类别上的空间响应。旧 semantic_residual_missing_instance_20260914 只做漏检峰提案，cross_head_class_rescue_20260914 把响应池化成类别分数，均未在当前固定官方 one-to-one 候选上改掩码。旧7O预测另一个ownership头，不能替代这里的现成原生信号。此前原型投影、局部系数、crop支持、重计权和翻转试验不重跑；借用算子，不把新信号包装成新投影数学。

源代码：8.4.100 Proto26 的 semseg(feat) 与 feat_fuse/Proto 同源，但位于 feat_fuse 之前；本轮使用其原checkpoint参数，全部eval，不调用fuse、不切train、不求梯度。语义能区分类别和背景，不能分离同类实例；旧框好掩码差组语义AUC约0.697，还是GT类别、排除同类邻居的全图统计，不能视为本轮强信号。

文献依据：[MaskLab CVPR2018](https://arxiv.org/abs/1712.04837)把语义与方向分别用于异类/背景和同类实例区分，提供信号用途与反例；[YOLACT ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.pdf)提供原型组合背景。本轮不是重现这两篇方法，也不宣称语义融合本身新颖。

## 完全冻结与七组

原官方8.4.100 YOLO26m-seg COCO权重SHA256见RUN_CONFIG和运行INPUT_MANIFEST。所有候选、原型P、系数c0、框、类别、分数、BN参数及buffers冻结。无GT进入语义响应选择、融合、投影或阈值。

z0=interpolate(Pc0)；s为原raw候选预测类别的原生semseg logit。语义网格先选通道，再双线性上采样640，align_corners=False。a为原生网格80通道max后同样上采样。S为原预测框在640整数网格的官方crop_mask有效支持。t为s在S矩形内旋转180度，支持外保持s；支持内值集合不变。不得换GT类别、GT框、置信度阈值或依结果换候选。

| 组 | 正常解码前的响应/系数 |
|---|---|
| A | 原c0、完整P，正常官方解码 |
| SEM | s在原框支持内以0阈值直接解码，仅信号质量诊断 |
| MIX | 0.5(z0+s)，原框、原letterbox几何、0阈值 |
| PROJ | c0+δ(s)，完整P正常解码 |
| ROT_MIX | 0.5(z0+t)，其余同MIX |
| ROT_PROJ | c0+δ(t)，其余同PROJ |
| AGN_PROJ | c0+δ(a)，一般前景对照，非标度匹配类别消融 |

δ(v)=argminδ mean(u∈S)[P_up(u)^Tδ−0.5(v(u)−z0(u))]^2+0.003||δ||²。

复用full_projection.py：完整640支持，精确分块累积FP64 32×32正定方程；不量化、不8×8压缩、不使用GT、不优化网络。λ=.003、融合=.5预先固定，不扫权重、温度、步长或阈值。输出系数投回原160完整P后正常上采样/预测框裁剪/零阈值/真实letterbox逆变换，与直接logit组同几何。空支持保留，返回零增量。

语义与实例logit未保证同标度；固定融合失败只能否定这个配置，不能宣布语义无信息。ROT保留直方图并破坏空间排列，但中心对称对象可能变化很小，记录s−rot(s)支持内RMS。PROJ>ROT_PROJ只支持本投影利用这一个框内的空间排列；不是实例归属已证明。AGN的max会抬高logit，不能把PROJ>AGN_PROJ单独解释为类别机制。PROJ>MIX仅说明这一个ridge投影优于同一个50%融合，不是一般投影优越性。

## 对象、指标与预算

复用fast-screen固定开发清单256张train2017开发图片，预计253有效、1816官方one-to-one正样本；无正样本图保留，不补图。逐候选永久七字段身份校验。这些图已经用于历史选择，是探索筛查，不是盲测，不报告COCO AP，不是全部推理候选部署。GT只用于已有入组与评价。

主指标图片macro正常原图MaskIoU；主要比较PROJ−A、PROJ−MIX、PROJ−ROT_PROJ。次比较MIX−A、MIX−ROT_MIX、PROJ−AGN_PROJ、SEM−A。报告候选平均、Mask75修复/损伤、coverage、预测框支持连续AUC/FPR；未定义AUC另计，不删除IoU候选。原成功/失败、原框好掩码差(BoxIoU≥.75,MaskIoU<.75)、P3/P4/P5和实际面积大小分别报告；预测类别正确性及同类邻居只评价分层。候选均值也整图配对bootstrap1000次，seed20261004，多臂区间仅探索解释。

实用筛查信号：相对A的全体macroIoU≥0.002且区间下界>0且净Mask75>0，或主失败组≥0.005且区间下界>0且全体不劣于−0.001。多指标有价值变化可结合代价如实保留，不能用偶然子组掩盖整体损伤。投影方法价值另外要求优于MIX，空间排列解释另外要求优于ROT_PROJ。达信号只允许另行登记独立确认，不能直接宣布总Goal完成。

授权3080Ti服务器运行；桌面只文件、记录、隐藏传输。独立smoke2张有效图上限180秒；通过后正式完整清单上限1800秒。固定CPU6线程。每阶段runner独立Run，保留输入hash、代码/配置快照、退出状态、失败日志。不更改失败Run原件，不自动重启。没有实用信号停止此配置；不追加训练、gate、新loss、权重搜索或扩大数据。

## 运行前/后核验

1. 原始远端状态、Run和实际PID/GPU检查，禁止重复或抢占。
2. 原输入字节与缓存SHA、七字段身份，原P/c/box/预测类/分数核对；原P/c/box容差atol=rtol=3e−5，原基线任务指标最大差≤1e−12。
3. 全程eval，semseg真实存在，状态前后原样；记录原生语义分辨率。只向source传GT-free allowlist。
4. 零目标投影返回原c0，方程相对反向误差≤1e−10；先组合后插值与重排算子atol1e−4/rtol3e−5；禁止为了通过改容差。
5. ROT支持内值集合不变、支持外未改、记录实际扰动强度；类别和邻居GT不进入source。
6. runner完成不等于科学成功；回传manifest逐文件SHA核验，独立Study报告给出结果、代价与停止决定。

## 执行入口

`python scripts/pipeline_semantic.py --root /root/native_semantic_projection_20261004_retry1`

Study/Run永久ID见RUN_IDS.json，所有预算/输入路径见RUN_CONFIG.json。原历史Run均保留，不覆盖。

