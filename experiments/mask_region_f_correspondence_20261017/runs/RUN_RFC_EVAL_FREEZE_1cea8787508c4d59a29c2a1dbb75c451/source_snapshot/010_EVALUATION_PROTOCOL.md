# RFC-v1 固定最终 checkpoint 评价执行冻结

2026-10-17 用户授权：“查下笔记本上 A/Q 训练跑完没有，完成后复核配置，按冻结协议做评价，不追加训练”。这扩展旧 EXECUTION_PROTOCOL 的“训练结束即停”执行范围。科学主张、比较、数据、Gate、停止出口均继续使用 PROTOCOL_DRAFT.md 第6—8节和原 DESIGN_SPEC.json，不重写历史训练协议或旧Run。

## 固定输入与时序

- A：RUN_RFC_TRAIN_A_59b094a35f9b4fc3a45f7a86862ec677，last SHA `1383818808b7c52594ba5dfb49ae8f4b858a900b02fb65dccd5f1c55e6bf1963`。
- Q：RUN_RFC_TRAIN_Q_ed1e5e49f60c42908d23822e5804f668，last SHA `877a3b221fb0b81fa20772495cfbade530402252202a8fff86a2cf816c058942`。
- G：原 soft-IoU checkpoint + 原生面积≥9216门控；复用 RUN_S3_FULL_670fffd010664a648e096572fc7c476d/predictions_mixed.json，SHA `15d8cc2884777854a7120099e4ee218b8750ee59f63d2b3fcc4571fa2a191590`，逐行非Mask身份与门外RLE核验。开发端点须复现原T+.010024491543988425、全配对+.0008909711434980322。
- 先复核两训练Run全部产物、实际图片顺序/GT join/初始化/17参数张量及17条有效梯度/各1500步；冻结 EVALUATION_SOURCE.json、推理合同和checkpoint。随后用系统随机源一次抽取22957原成功GT中的1024条，无放回、不补样、不重抽。此前没有读取本次科学预测或效果。
- 原dev512和新损伤图并集≤1536；T655/257有效图、全部bbox75配对2925/508有效图不变。只补导该并集需要的同层区域F，沿用完整原生批次、输入指纹与全部输出身份。每图P/c0/单点F及zero-delta原图RLE必须精确重放。

## 方法、评价与判定

N/G/A/Q正常完整预测覆盖整个并集；dev512另有J及AeqQ/QeqA/AeqJ/JeqA。所有新方法均仅在原生掩码面积≥9216的行修改系数；门外RLE逐字保留。J只改变同一A权重的区域F排列；等幅控制调用原冻结FP64 RMS算法，双方降至较小幅度，零方向/空ROI都回原生且保留分母。布局置换固定点、常量F、幅度前后及全行系数保存。

每组完整图片的全部GT/预测重新执行冻结COCO bbox/segm独立匹配。固定配对以原bbox75匹配检测身份为准，不因新Mask匹配改配对。主指标为原图同检测IoU；先图内均值，再257/508图宏平均。复用原image_effects/bootstrap：FP64、image_id升序、NumPy2.4.3 default_rng1729、2000次同索引配对图重采样、linear百分位区间。COCO ledger过程附带产生的AP仅为注明图集合的审计副产物，不用于与全量S3或官方指标比较，不增加AP决策门槛。

唯一主方法A同时要求：T上A−G≥.005且CI下界>0；T上A−N≥.005且CI下界>0；全2925配对A−N≥−.001；新1024样本的精确单侧95%超几何损伤上界≤1%；身份/配置全通过。机制支持另要求A−Q、A−J、AeqQ−QeqA、AeqJ−JeqA四项T区间下界全>0。G/Q共享面板只报告比较，不能择优替换A。

效用通过才建议同A checkpoint全量S3（本轮不启动）；机制未过则仅保留复合效用，不作空间归因。效用失败或精度不足即关闭RFC-v1投入、保留G并把本次证据并入论文；不补训练、样本、seed、网格或loss。工程错误保留失败Run，仅修复可核实工程问题，重复抽签被禁止。

## 执行和资源

已授权笔记本28358lan，原Python3.12.3/torch2.5.1/Ultralytics8.4.100环境、TF32/确定性实际值全部匹配G。三个有限阶段分别登记Run：EVAL_FREEZE、EVAL_INPUT、EVAL_FULL；无定时、无自动后继训练。

正式F导出总10800秒，已消耗训练F Run的914.672秒，评价F最多9885.328秒；正式基础图forward总≤7000（2000训练+≤5000原生验证批次）。科学预测/匹配/统计总14400秒。CPU RSS≤8GiB、CUDA allocated/reserved≤7GiB、起始可用RAM≥3GiB；原64GiB磁盘预留按空余+本Study已写入核验，F≤48GiB、其余≤8GiB。

结案前：全部完整预测、行映射、固定配对、GT账本、幅度和配置收据回传并核SHA；另行独立复算统计、损伤和Box身份；更新Study/ACTIVE_PLAN并执行check_active_plan.py。不自动提交推送，不修改其它会话文件。
