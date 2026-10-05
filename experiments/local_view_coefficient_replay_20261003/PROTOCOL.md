# 冻结模型的局部视图证据 → 原图原型系数：零训练短测

六臂静态选点screen没有显示值得扩大的新增任务收益，见 ../prototype_readout_fast_screen_20261003/REPORT.md。本轮改变GT-free空间证据来源，不训练模型、不扫solver参数、不加gate/loss/更宽头。研究Goal持续授权下的一次独立、最多1小时的探索重放。

## 对象与唯一新变量

复用完成screen的256张dev图及其1816个官方one-to-one TAL固定候选（253有效图片），不重新分配。原P、c0、目标预测框与候选身份不变；不增加样本，不按效果剔除候选。原始GT只参与现有候选身份及最终评价；新视图source函数不接收masks/owners/rows/GT类别/GT框。

对每个原预测框，经真实letterbox几何映射到原JPEG，取中心不变的1.5倍上下文整数ROI（向外floor/ceil，裁到图像边界，至少2像素），独立letterbox到640，再用同一个冻结官方8.4.100/COCO预训练模型前向。使用官方one-to-one head的top300池，不另加conf阈值、GT或类别条件；以与原目标预测框的最大几何IoU选择对应source候选，平分时按分数及固定顺序。新box仅用于对应，所有mask仍按原目标预测框裁剪。记录source对应IoU、选择身份、裁图/缩放/填充及耗时，不将几何对应可靠性预设为已证明。

使用选中source的未二值化mask logit，逆新视图letterbox与ROI变换后映回原640输入框架；无新视图观测区域保留原P*c0，不能补零伪造背景抑制。保留原model参数和所有buffers不变，不读取先前短训权重。

## 七臂输出与固定solver

A：原始c0完整P正常解码。
BASE_SOLVE：旧OGPS自投影，q=sigmoid(A8*c0)，经过同一截断和solver；这是校准/投影控制，不要求其二值mask恒等。
ORIG_RESELECT：原640输入也执行同一个冻结模型、官方top300与同样的BoxIoU→score→raw身份对应，选中系数在原P中正常解码，最终仍裁原目标预测框。它隔离“换raw位置”本身的作用，原P必须与缓存一致。
ORIG_RESELECT_SOLVE：该原图重选候选的完整连续logit先sigmoid再按同一crop_pool_7o池化8×8，使用同一OGPS solver，以原目标c0为中心。这与VIEW_SOLVE使用完全相同的pool(probability)流程。
VIEW_FULL：映射后的新视图连续logit，原框裁剪、阈值0、相同原图恢复。
VIEW_GRID：新视图probability按旧crop_pool_7o池化8×8，q8夹[.01,.99]后转t8，t8双线性贴回旧floor/ceil框，原框裁剪、阈值0、相同原图恢复。
VIEW_SOLVE：同一个q8走旧OGPS solver后在原完整P上正常解码。

完全复用 ogps_solver.py：A8为原预测框内原型native160尺度floor/ceil截边平均池化8×8得到64×32；FP64求解 delta=(A8^T A8/64+0.003I)^(-1) A8^T(t8-A8*c0)/64；正则在系数增量，无额外jitter/调参。注意 pool(sigmoid(z)) != sigmoid(pool(z))，两条路径按以上定义分别记录。另用无截断t=A8*c0检查delta接近0，只是代数核验，不能代替正式BASE_SOLVE。

## 执行与评价

只在3080Ti远端执行，不在桌面运行模型。先2张预定有效dev图smoke，随后完整256张，正式Run硬预算3600秒；未做完标记未完成，保留部分产物，不缩样追结果。两阶段用独立Run和runner，已有运行不重复启动。代码预检、源模型/标签/基线复现只复用或针对新增路径做必要检查，不重新跑旧oracle/训练。

A必须按永久候选身份复现刚结束screen中的原图IoU、Mask75、coverage、连续logit AUC/FPR；source几何以带padding非方图及坐标ramp作检查，冻结状态必须不变。直接输出与系数输出保持相同原预测框支持、阈值、输入二值mask→真实ratio_pad原图恢复顺序，full-view与grid-view区别包括池化/截断/插值，不能只解释为输出分辨率。

主量为dev图片macro原图Mask IoU；同时报告候选均值、Mask75修复/损伤、coverage/AUC/FPR，框好掩码差组固定由A判定。图片bootstrap1000次，探索性、无新盲测/AP。主要比较VIEW_SOLVE−A及VIEW_SOLVE−BASE_SOLVE；VIEW_SOLVE−ORIG_RESELECT_SOLVE和VIEW_FULL−ORIG_RESELECT专门检验增加局部视图后是否超出原图重选候选，不能把换位置的收益归给放大信息。VIEW_SOLVE与VIEW_FULL/GRID区分新增视图和固定basis投影的作用。GT不进入source或solver，不以子组单独成功替代整体结论，也不要求所有指标同涨。

## 有界判断

参考前一轮实质量级：全体+0.2pp或目标组+0.5pp，结合其他指标及代价判断是否值得后续独立确认。新视图没有有价值的信息则停止，不追加新的学习头救场；视图直接有效但投影不能保持收益，则保留视图结果并停止当前投影；若相对自投影没有优势，不归因新增视图；若两者有效也不把TTA/计算增加的收益全部归给投影。归属错误与计算成本如实记录，不自动扫上下文倍率或matching阈值。此短测仅决定投入优先级，不完成最终研究Goal。
