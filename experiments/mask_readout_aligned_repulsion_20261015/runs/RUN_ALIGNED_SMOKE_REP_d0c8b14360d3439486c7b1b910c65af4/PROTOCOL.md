# 实验四：原生输入解码尺度监督与背景邻居排斥

2026-10-15。用户已授权依次完成实验2、3、4；本项在E3结案后启动。使用原B adapter、原初始化、原2000图缓存。两臂各一次3轮/1500步，不增加训练规模、epoch搜索或阈值扫描。两臂训练结束后才评价任何一臂。

## 主张和必要对照

H4：在原生输入解码尺度直接监督GT，可以减少proto监督到最终二值mask的错位；在目标GT背景的邻居重叠位置施加固定排斥项，可在此基础上改善修复。两条是待检验主张。已知S1有限系数解存在，旧BCE与soft-IoU pilot失败；不据此断言错位或邻居是唯一原因。

- R：原B结构167872参数；GT BCE从proto格点改为实际native输入解码尺度，原proto保持BCE权重.1不变。
- REP：同R，额外加权重.1的下述GT背景排斥项。除这一项权重，R/REP的源码、初始化、图序、配对、预算全部相同。

原BCE已完成的Run仅作历史参考，不重复训练。R/REP同批新对照可直接检验排斥项增量；它们不引入E2球面/门控/邻居网络，不继承旧soft-IoU checkpoint或9216面积门控。

这里对齐的是**原生输入尺寸、硬阈值之前**的连续logits，不是对最终原图二值RLE进行可微优化。原图后续resize/byte仍不可微，不能声称解决所有解码差异。GT掩码限制排斥的直接像素梯度，但共享系数可改变别处，不能保证正常实例零损伤。

## 冻结损失与分母

按原class/box50 score-order ordinary GT一对一训练join；只对原proto ROI非空的匹配检测监督，实例集合与旧B相同。所有原proto有效检测参与保持项；无匹配图仍参与训练与图片分母。

对匹配检测i，原始COCO GT按实际缓存记录的整数resize/pad，用nearest变换到native输入；不经过proto降采样再放大。其尺寸必须同时等于input_hw与native_decode_hw。预测logits由原P和新系数得到，经原版本ops.scale_masks到native_decode_hw；半开预测框定义ROI，验证其非空。

GT BCE先在每实例完整ROI取均值，再在该图全部匹配实例取均值；保持项仍完全使用原proto `vector_grouped_loss` 的独立实例均值/图片分母，目标为detach(sigmoid(z0))，权重.1。

邻居规则与E1/E2相同：同类、不同raw、更优(-原生未舍入score,rank)、BoxIoU严格>.1，Top1按IoU/score/rank。R/REP均计算同一项但R权重为0。邻居原生logits以原ops.scale_masks升到native输入再sigmoid并detach。

`Lrep_i = mean_ROI_i[ sigmoid(z_i_new) * stop(sigmoid(z_j_native)) * 1_(pixel in B_j) * (1-Y_i) ]`。

无邻居时项为0，仍计入该图全部匹配实例分母；不缩成“有邻居/有冲突”集合，不按正像素数量重新归一化。总损失R为GT+.1保持；REP为GT+.1保持+.1排斥。不改变GT权重、置信度、检测集合或评价分母。

native尺度监督按最多4,000,000像素/块处理匹配检测，逐块累加相同标量目标梯度，再加保持项；合成控制验证与不分块的梯度一致。每图1次forward，图微批1累积4，分块不改变有效图片batch或实例权重。

## 训练、运行和停止

每臂从 `RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2/initial.pt` 全量逐字节加载，seed0，优化器全新。原2000图、3轮6000图访问/1500 AdamW步、原IMAGE_ORDERS和TRAINING_JOIN字节一致；lr.001、betas(.9,.999)、eps1e-8、weight_decay.0001，其余原布尔参数false。FP32、无AMP/增强/scheduler/early-stop，仅last checkpoint。

基础模型权重/P/c0/F、Box/class/score/排序/raw和output身份、框裁剪/阈值/插值/native解码、Ultralytics8.4.100及原batch16/rect=True/imgsz640缓存、数值环境、GT和全部评价器冻结。改动白名单仅上文损失项。推理adapter不读取GT或匹配信息；不重新forward基础模型。

先合成控制：半开ROI、GT前景上排斥直接梯度为0、无邻居项为0但分母保留、分块与整批梯度一致。真实smoke两臂各原fit前8图/dev前8图zero-delta全输出RLE核验及2步，均丢弃参数；≤900秒/臂。正式R后REP各≤7200秒，单臂评价≤3600秒，RSS≤4GiB、GPUreserved≤6.5GiB、每Run产物≤1GiB。资源预检失败不偷偷换数据/减图/降精度。工程失败保留Run，正式已消耗步数不能被重复完整训练掩盖。

## 评价和唯一方向裁决

两臂均导出完整原1314图/185621条最终检测，原512dev/2925固定Box75配对/T655，原成功面板1024/N22957，未修复行仍入分母。每图先完整zero-delta原生RLE重放，只有segmentation字段可变；bbox与segm COCO匹配独立。

原S2三项门槛：T图宏均值≥.005且2000次paired-image bootstrap95%下界>0；所有bbox75图宏均值≥−.001；精确超几何单侧名义95%原成功损伤上界≤1%。seed1729、linear分位数及原面板字节不变。两臂都按原Gate独立判定；不以组间差异代替绝对损伤护栏。开发面板已暴露，不称独立确认或新的95%覆盖保证。

用同图同实例的 `delta_REP-delta_R` 计算T图宏均值和相同bootstrap区间，作为排斥增量；只有其下界>0且REP三项Gate通过，才支持本固定排斥项有新增收益。R通过且REP不满足此增量证据时选择更简单R；只有REP通过则保留REP方法包，但不自动归因排斥。两臂均失败则关闭两个固定配置。结合E1—E3及已有9216面积门控结果，最后给一个有证据的方向；不自动进入S3或20k。

## 证据与复核

每臂smoke/train/eval独立Run，冻结协议/源码/EVAL_CONTRACT/环境/输入与缓存SHA，逐图三项loss、实例数量与实际图序、优化器状态/白名单、initial/last、TRAINING_JOIN、完整预测/配对/GT账本、SUMMARY/DECISION/CONFIG_AUDIT。训练后独立核6000图访问、1500实际Adam步、原初始tensor、GTjoin、完整loss组合；评价后独立核全部非mask字段与系数加法、2925条RLE IoU、bootstrap和损伤上界，固定前8张图从序列化checkpoint重放全部系数。R/REP对照另核损失权重以外合同一致，配对差值不丢零行。完整产物回传后逐文件SHA核验。
