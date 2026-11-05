# V28 实际对象编码—记忆—解码接口工程

单独一次性Slurm，已登记个人账号、1A6000/4CPU/24GiB/最多5分钟，独立本次上限1/12GPUh，0优化。不借旧Run预算、不激活后续训练/缓存作业；所有失败/原日志保留。前置为官方源码blob、权重SHA、CPU strict加载与构造核心通过。

实际4张real train查询图按图SHA排序固定；从13个real train来源按来源/图SHA稳定选最多2图/来源，前16张各取第一个有效GT对象作为支持经验。支持图/mask仅来自train，查询自身来源完全排除；无dev/test。支持GT是学习对象经验的合法训练资料，当前查询GT不进前向。支持与查询共用同SAM2.1 Hiera-base-plus编码器，同FP32→half→FP32特征路径，编码器/高分辨率预投影/提示编码器冻结。

核四图fresh FrozenNative与原native缓存的检测/系数logits/ID/ordinal精确；全部合法正常检测按chunk2验证：自己实现的box+native-logit提示、单mask解码与官方predictor在同量化特征/同提示/同chunk下full640 logits和binary逐值精确。原生Box/Class/score和行身份不改，记录数量。

对象记忆context输出的最终投影0，初始完整memory模型等于上述强底座；它不是原生YOLO权重或正式科学基线结果。真实支持feature/mask、排除矩阵、检索行与strict重载可追溯。选第一个按原成本owner标记可靠的query，仅在loss处使用其GT，核对象自重建/正确known处理和mask解码梯度；再仅对副本投影固定.01I检查正常任务梯度抵达共享检索键。0optimizer，副本丢弃，原模型/底座state不变。

此阶段只证明真实接口/梯度/精度契约，不声称原型学好、named real anatomy或+5。正式K=256/完整train缓存/完整训练吞吐尚未由16支持工程证明；真实部位/姿态unknown不作负例。后续完整数据、训练/正常评价/独立重算/重载/图册代码与两轮同源审查必须完成才可训练。
