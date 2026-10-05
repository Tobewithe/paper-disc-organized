# 整图翻转响应与原型投影：零训练判别

2026-10-04预先锁定。本线程3080Ti服务器执行，不访问笔记本，不在桌面运行模型。

## 问题、依据与去重

固定官方模型在同一图像水平翻转后，是否产生有互补价值的掩码响应？这种信息能否通过原图已有prototype变成更好的系数，而不增加训练网络？本轮先区分“第二视图有无信息”与“投影到原基底是否有额外价值”。

历史[local-view](../local_view_coefficient_replay_20261003/REPORT.md)实际采用1.5倍框上下文裁切再放大到640、top300选位、8×8概率池化截断及Ridge，原图IoU没有提高。[candidate-consensus](../candidate_response_consensus_20261004/REPORT.md)研究同图其他候选的响应，未建立共识收益。本轮改变的是整图反射，保持尺度、上下文、缓存输入的全部像素与完整连续logit，不把失败方案换一个倍率重跑。检索未找到此处完整对照的既有Run。

[CVPR2021 SAC](https://openaccess.thecvf.com/content/CVPR2021/papers/Araslanov_Self-Supervised_Augmentation_Consistency_for_Adapting_Semantic_Segmentation_CVPR_2021_paper.pdf)及[CVPR2020 SEAM](https://arxiv.org/abs/2004.04581)是增强/等变一致性的原始先例。它们不证明YOLO的此方案有效；翻转集成不是本项目原创。本轮是低成本机会判别，尚非新方法贡献。反例：两视图可能重复同样错误；反射raw可能换了负责对象；原P投影可能丢失另一视图的有用响应。

## 固定对象和来源

原官方COCO YOLO26m-seg、Ultralytics8.4.100；复用fast-screen 256张已查看的dev清单（预期253有效图/1816官方one-to-one候选，实际按原INDEX核对，不补删）。所有永久身份、预测框、类别分数及最终原图逆变换固定。没有训练、TAL重算、GT选源、oracle或全COCO AP。

对缓存RGB uint8 640×640整张canvas沿宽翻转，包含padding；这只是精确整数像素置换，先正常解码完整640连续logit再逆翻，恢复原canvas后仍按原预测框裁剪。无需重做JPEG缩放或letterbox。两次翻转必须逐像素复原。

翻转输出的prototype与coefficient必须配对使用。不能把翻转系数直接乘原P，因为跨视图通道坐标并不保证等价。

两种来源：
- MIRROR：原raw所属80/40/20网格中横向镜像位置。只称同物理网格中心，不宣称同一实例。卷积步幅采样相位及候选归属可能改变。
- MATCH：仅依赖原预测类别、原预测框，在翻转输出中选择**同pyramid level、同原预测类别**且逆翻BoxIoU最大的raw；同IoU时按原预测类别分数、raw索引排序。不进行top-k/conf/NMS过滤，不用GT类或mask。无同类别位置则回退MIRROR，记录，不删候选。

原图自身用同一MATCH规则重选作为ORIG_MATCH控制，记录是否回到原raw。翻转镜像/匹配源的框IoU、类别一致、重复donor及回退数全部保存；低对应质量不剔除。

## 六组及唯一投影

设原logit为z0，配对翻转源逆翻后的连续logit为zf，原P的双线性上采样为Pup。

| 臂 | 做法 |
|---|---|
| A | 原P和原c0正常解码 |
| ORIG_MATCH | 原图同规则重选的c，以原P解码、保留目标框 |
| MIRROR | 镜像raw的完整zf，保留原目标框 |
| MATCH | 框对应raw的完整zf，保留原目标框 |
| MIX | (z0+zf_MATCH)/2，普通固定等权连续响应集成 |
| PROJ | 只更改原系数，将同一MIX增量投影回原P，再正常解码 |

PROJ唯一固定计算：在原预测框的官方整数像素支持S内，FP64精确分块累积

\[
\delta=\arg\min_d\;\frac1{|S|}\sum_{u\in S}\left(p_u^T d-\tfrac12(z_f(u)-z_0(u))\right)^2+0.003\|d\|_2^2,
\qquad c'=c_0+\delta.
\]

32维正定正规方程直接求解；不是7D的GT-BCE有限oracle目标，也不借用其证书。没有8×8压缩、sigmoid池化、clip、阈值校准或选择融合权重。空支持按协议返回零增量并计数。self-target零增量必须精确返回c0；正规方程相对后向误差≤1e-10。全640插值P后乘c与先乘后插值的误差预先固定atol1e-4/rtol3e-5，记录误差；评价总用原版先组合再插值，不将浮点交换误差当方法收益。

## 核验、预算与评价

先2张有效图smoke（180秒），再固定256图一次正式评价（1800秒），两独立runner Run。只修明确执行故障，原Run/日志保留；不能改科学配置救负结果。源码、协议、配置和输入版本在启动前冻结。原P/c0/box回放atol=rtol3e-5；A与既有五指标逐候选误差≤1e-12；未过核验不解释结果。

主效用为MIX−A、PROJ−A的正常原图图片macro Mask IoU；PROJ−MIX回答原P投影是否超越普通集成。MATCH−ORIG_MATCH帮助分离视图和原图重选；MATCH−MIRROR只诊断位置对应。预先保留所有组的IoU、Mask75、repair/damage、coverage、AUC、框内FPR、空mask及source误差。AUC使用完整连续logit，原始COCO标签映射到输入空间、固定原预测框支持；未定义条目保留IoU。图片macro与候选平均均整图配对bootstrap1000次seed20261004；多个对照区间是探索性描述。

预定分层：all、A成功/失败、原BoxIoU≥.75且MaskIoU<.75、P3/P4/P5及COCO小中大。不能把原失败组零damage当全体安全证据；层级不等于目标大小。

## 决定

若MIX或PROJ相对A整体macro IoU≥0.2pp、配对CI下界>0且Mask75净修复>0，或主失败组≥0.5pp且CI下界>0、整体不低于−0.1pp，则有继续独立确认的初筛信号。其他有意义指标改善须明确代价与对象，不能仅挑一次偶然上涨放行。

PROJ只有明确优于MIX，才支持投影的额外效用；若MIX有效而PROJ没有额外优势，只保留普通多视图信息事实，不能宣称系数求解创新。若翻转来源差、匹配来源好，降低固定网格等变假设；若均无有用收益，结束当前整图翻转及固定投影配置，不扫变换、融合权重、匹配门槛或ridge值，不自动训练一致性头。

这是旧开发集上的候选条件诊断，成功也须独立图像与正常输出验证。失败不证明所有增强或等变学习无效。完成回传、报告后结束本轮，研究总Goal另行保持。
