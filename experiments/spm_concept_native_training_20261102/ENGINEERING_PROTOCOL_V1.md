# 原生参数—梯度—TRAIN-only接口资格V1（正式训练前冻结）

## 对象与执行

仅登记a25023104@192.168.71.10服务器、冻结Python/Torch/Ultralytics8.4.100与N权重。一次Slurm：1GPU、6CPU、24GiB、30min；最多2次已定位工程重试，全部计入Goal 24GPUh/180CPU-coreh。当前不读dev/test、无新合成、无正式优化。最多64次native scratch更新，权重不得作为正式初始化。

输入：1272 TRAIN-only manifest、原640预处理RGB、原160独立实例Mask及归一化xywh；保留真实重叠与设施填入，不重画GT。核全部源字节、数组形状/类型/非空和框界限，固定最小4个image_id用于梯度资格。合成使用发布90图manifest，只读取72 train目标，固定000/001/002各A/C六张用于三档资格；QA不进优化。读取所有TRAIN数组是正常监督资格，禁止回流生成器。

## 参数与损失候选（训练前可按本协议确定λ，不看dev）

- 原生原本可训练的非BN参数参与适配；所有BN affine/running stats/counters固定，原生固定常量保持。精确白名单/计数实际导出；不沿用Mask-only白名单。
- 原生训练分支保留全部正常任务损失及O2O detach。F=proto.cv2输出，几何4通道和部位6通道无BN 1×1头分别1028/1542参数。G用几何头，Q/C两个头，Q/C参数初值同源。
- 真实任务单独前向4图；合成2图单独前向，仅辅助损失，0合成原生target。所有组BN均固定。
- 辅助归约：逐图、逐owner（背景单独一个单元）、逐有效类/通道的正负支持分层均值；空项为连接图的0并记录。部位CE只用part_valid；每owner各存在类均权，再owner均权，前景owner平均与背景平均各1/2（无背景则只前景）。几何BCE每通道正/负各1/2，owner/背景单元均权，再4通道均权。各valid独立，未知不作负例。
- Q逐owner空间排序保留每类数量/背景/unknown；不把Q当完美反事实，记录其训练难度与梯度。
- 原生候选AdamW lr峰1e-5，辅助1e-4，矩阵wd1e-4、bias无wd，betas(.9,.999)、eps1e-8、clip10、FP32/TF32off；100步线性warmup从峰值/10到峰值，余弦到峰值/10。原生criterion epochs=2000作为更新时钟，O2M .8→.1。
- λ采用事前确定的TRAIN-only标定：在固定4real与3个合成A/C对上，分别取原生task及几何/正确部位对proto.cv2.conv.weight的L2梯度范数中位数；λg/λp各为clip(.05*task_norm/(.8*aux_norm),.001,1.0)。仅执行一次，无候选搜索，Q与C共享λp、G/Q/C共享λg。总loss=Ltask+wO2M*(λg Lg+λp Lp)。梯度范数不是效用/机制证明，Q幅度不同如实报告。

## 必须通过

1. TRAIN1272/原实例与像素重叠保留，split与字节正确，合成targets不能进入原生criterion；零分母和padding/unknown安全。
2. 初始N正常输出与既有原生参考严格一致；原生可训练参数仅白名单、BN与固定常量逐值保持。
3. 实测F为[batch,256,160,160]，正常任务及概念loss对共同上游参数梯度非零且有限，O2O proto保持detach；真实任务与合成辅助分开，没有真实部位伪标签。
4. R/G/Q/C scratch分别≥4有效更新；共同初值、Q/C辅助初值一致，更新/criterion时钟正确。固定参数无梯度/无变更，获准上游参数实际更新。
5. 独立fresh原生导出/重载输出逐值一致；不宣称Box/score身份训练前后不变。辅助头不进入推理checkpoint。
6. 记录实际耗时、显存及有效参数；用实测制定正式walltime预算，不引用旧Mask-only成本。

无效/非有限/身份或接口错误：保留Run，定位后版本化修复；不提交正式训练。全部通过仅允许冻结完整正式合同，不算科学阳性，也不自动打开test。
