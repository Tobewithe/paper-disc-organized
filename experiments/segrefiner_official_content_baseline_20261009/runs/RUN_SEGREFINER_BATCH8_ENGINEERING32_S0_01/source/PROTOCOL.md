# 官方 SegRefiner LR 内容细化强对照

2026-10-09，B授权的下一独立baseline Study，用已有RGB内容细化方法界定质量、损伤与成本的权衡；不称本项目创新，不重开F或补TriFlow，不训练、不扫参数。原七臂/多角度/成本及其已接受结果保持。

作者来源为 [SegRefiner官方仓库](https://github.com/MengyuWang826/SegRefiner)，执行前锁 revision `53419a2d38ea3da0b6e2be77e5b45e139195a0b3` 的源码与 `configs/segrefiner/segrefiner_lr.py`、`segrefiner_coco.py`，LR权重采用README官方链接并记录下载来源/文件SHA/可核checkpoint元数据，不以替代随机模型通过工程。LR配置声明训练 `lvis_v1_train.json`，官方实际训练图像范围与本项目COCO val5k的精确交集必须先审计；配置声明不等于checkpoint训练manifest已证实。交集未知/非零先返回事实给B决定评价范围，不包装成干净独立确认，不根据GT质量挑子集。

环境明确为桌面RTX5060Ti、`C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe`。旧依赖隔离或做有源码/数值记录的忠实适配，不覆盖现有pytorch环境；模型加载、预处理、随机采样与解码规则按锁定作者代码核对。先做有界依赖/算子检查，不无界修旧框架。实际推理与当前桌面独立成本串行，下载/小源码核对可先并行，重CPU解析/编译不得污染正在进行的正式时延测量。

直接读 `experiments/frozen_native_mask_comparison/runs/RUN_FROZEN_NATIVE_5K_INFERENCE_S0_02/baseline` 的原图RLE及同JPEG字节RGB，不重跑YOLO改变跨设备baseline。主比较固定原first64/protoROI可作用集合，使用封存native资格/顺序而不是从四舍五入score重新筛选；范围外全部保留。沿官方COCO配置的原area/crop/空mask规则，`object_size=256`、`pad_width=20`、`batch_max=32`、`area_thr=512`，原LR六步配置不调；代码核对后明确每种回退并保存全部正常框、类、score、顺序、空ordinal。官方发表的评价范围不充当本项目实测指标。

推理进程只接触图像、baseline预测、native资格及必要图像元数据，不读GT或按GT挑候选/调结果；GT仅供独立训练来源审核和评分进程。运行记录实际解释器、设备、输入输出、原权重/源码/配置与适配锁，保留失败；独立评估和重试新Run。若官方推理含随机性，执行前记录其原采样规则并固定可重现的种子/状态，不挑seed。

固定原5k清单原顺序前32张为独立工程，验证真实GT-free路径、完整身份/空/fallback、输出与实际统一端点成本；工程质量不能代表能力结论。只有训练来源/评价范围清楚、忠实工程通过且成本可执行，才按固定官方配置自动继续允许范围的完整5k评价；若需因来源改变范围，先由B基于具体事实确定，不擅自把已观察32图改成盲测。

正式比較采用既有完整baseline、冻结RCMC与refiner，保留原fixed detection→GT队列及正常COCO全输出评分，并读出修复/损伤、candidate/image macro coverage/purity/FP-G/FN-G与固定Boundary .02。空purity/null和成对有效分母沿当前研究线规范，未实算区间未知；GT关联/普通AP/Boundary分开。统一部署端点为已读原始图像与冻结正常掩码输入→全部原图binary masks CPU，加载/参考/审计/RLE/写盘/评分与部署耗时分列；若需要与旧cost端点换算，记录可比边界，不能直接混表。

公开LR预训练预算与本项目20k训练不等同，不作同预算消融主张。权重不可得、训练范围未知/重叠或忠实实现存在实质障碍时保存来源和具体阻塞回报B；不临时加gate、步数、阈值或改模型来追32图效果。当前仅授权现成强对照；集合AP效用的新训练目标尚未启动。
