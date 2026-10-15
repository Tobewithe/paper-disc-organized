# 实验三：2×2 子区域系数的有限恢复机会

2026-10-15；用户授权顺序完成实验2、3、4。本项在E2结案后启动。GT辅助有限oracle，不训练部署模型，不作为S2/COCO AP结果。使用原S1已冻结的256条同输出配对及原全局32维oracle结果，避免重复运行已经完成的全局oracle。

## 主张与预注册判定

H3：同一固定P和最终检测框内，允许不同子区域采用不同系数，能找到全局32维有限oracle未找到的Mask75解。理论依据是空间加权的四组系数扩大函数族；已有S1证实固定P有可恢复机会，但不能由此直接推断空间扩展收益或可学习性。

两个固定候选：Q-all允许四象限残差（128维）；Q-local只允许与指定邻居框实际相交的象限残差，其它象限残差为0。Q-all是主机会检验，Q-local检验提案“仅改受邻居侵入区域”的限制是否保留该机会。未改/无邻居行保留全部256分母；另报告有邻居子集但不得替代主分母。

以旧同配对global32的122/256为有限预算参考。Q-all找到同输出原图IoU≥.75的比例至少20%，且与旧global32比较净增加至少13/256（预注册约5个百分点的最小有用增幅），才支持值得开发空间系数预测器。成功率/增幅不够且所有求解收敛则关闭本固定扩展配置；求解精度不足时保留已找到的下界，负向裁决标为预算/精度阻断。Q-local达到相同净增标准才支持“仅改邻居象限”限制；否则不从Q-all结果转移结论。

原global32也只是有正则、有限预算所找到的下界，不是数学上限。门槛是本新参数化的开发机会筛查，不修改历史S1或S2 Gate。无论结论如何，按授权继续独立E4，不自动训练128维adapter或扩大数据。

## 固定对象、空间权重与求解

复用 `RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949` 的 TARGET_PANEL、ORACLE_RESULTS 和同一S0原生缓存/权重/GT。保持256个GT、baseline_output_detection_id、raw_id、原始框/类/score和配对。真实smoke用该正式面板前4条，仅工程验证，之后正式256条；不会据smoke效果选配置。

几何标签统一用原生FP32计算，求解用float64权重、解码用FP32权重。在实际proto格点按预测框中心的水平/垂直线划分四象限，划分覆盖整个proto画布；边界右/下归右/下象限。各one-hot权重做3×3均值滤波，边缘replicate padding。权重非负且逐点和为1；不在proto框边缘截断权重，避免额外改变插值支持。实际裁剪仍由原生decoder在原顺序完成。

`z_new(q)=z0(q)+sum_s w_s(q)*P(q)^T delta_s`，z0用原生整图系数矩阵乘法，零残差必须完全重放原生RLE。Q-all四组32维delta全可求解。Q-local邻居继承E1同类/不同raw/更高score优先、BoxIoU>.1的Top1规则；在未羽化的目标框内象限，若存在格点落在邻居框内，该象限可调，否则delta=0。最多四组；无可调象限直接原生。羽化边界最多一格支持相邻区，因此不宣称逐像素严格零扰动。

GT投影和损失ROI沿原S1：原图annToRLE、实际整数resize/pad、nearest到proto，半开预测框ROI；损失 `mean_ROI BCE(z0 + Design*delta, GT) + .003/8 * sum_s ||delta_s||²`。Design为各激活权重乘原P。每组等效L2系数 .00075；四组相同delta时正则恰与旧global32的 .003/2*||delta||²相同。Q-local也用同一组系数，不按激活数量重新归一化。

CPU单线程float64、零delta初始化，原LBFGS max_iter120/max_eval150/history20/lr1/strong_wolfe，tolerance_grad1e-9、tolerance_change1e-12、梯度无穷范数≤1e-6才称收敛。两候选每目标独立零初始化，不warm-start旧解，不增加步数。原始proto完整ROI不降采样、不抽像素。

求解后FP32 CUDA生成新logits，沿原ops.scale_masks→logit>0→crop_mask→scale_masks至原图→byte路径；不替换阈值或resize顺序。记录每条完整原图RLE与同一GT的IoU。该空间参数化已超出原单一32维c，仅参数化是研究变量；原权重、P、检测身份与native版本不变。

## 工程与证据

先测试分区和为1、零残差恒等、均匀残差的代数等价、局部象限支持、同正则求解器已知非零解正控；真实smoke≤900秒，正式≤5400秒，RSS≤4GiB、GPUreserved≤6.5GiB、产物≤1GiB。真实每张图全输出原生重放；缺缓存、输出身份/GT变化、非有限解、decoder失败立即工程阻断，失败Run保留。

每Run单独协议/源码/参数/输入SHA快照、命令与环境、原面板、逐目标求解状态/迭代/梯度/系数/原图RLE/IoU、SUMMARY/DECISION/CONFIG_AUDIT。运行前后核全部实际缓存、权重、GT、参考Run与配置源码。独立复算所有256条两候选RLE交并、恢复计数和净增；按固定首8条从系数和原缓存重建空间logits/RLE，检查浮点解码一致性。返回当前机器并逐文件SHA核验。
