# 一次固定执行批量适配

2026-10-09，B在原batch_max32真实工程的WDDM专用/共享内存证据后授权此硬件适配；旧协议/源码、CPU/GPU核验和正在执行的32图工程保持原字节并正常收尾，失败仍保留。原32工程不直接扩大5k，也不称纯设备显存驻留成本；观测不能精确归因全部慢速。

新独立Run唯一配置改变为batch_max=8；官方LR全部455权重、FP32、6步5→0、前5步随机采样、crop/pad20/resize256/原paste/area512和base_seed20261009+image_id规则均保留。这是公开披露的执行批量适配，不按质量选择batch/seed，不扫描其它批量、精度或步数；不能继续声称全部官方默认配置未改。

分批改变torch.rand消费分配，因此新输出不能要求或宣称与batch32逐位一致；只在同batch8/同输入/权重/种子及RNG下重新核验未改作者核心与scope wrapper张量/最终mask一致。该核验共享有据framework shim，不称独立完整旧MMDet/MMCV环境复现；同时明确真实/合成case是否覆盖511/512阈值、8以上有效候选第二批及empty/tiny/ordinal边界。

原GPU32工程结束后串行新GPU核验与同一固定前32图batch8工程，记录真实WDDM DedicatedUsage/SharedUsage/TotalCommitted、CUDA allocated/reserved和部署端点耗时；不让旧热缓存或另一GPU进程污染资源证据。参考/初始化/warmup/RNG记录/审计/RLE/写盘分列，来源/源码/配置/种子留Run。

batch8忠实与身份/GT-free工程通过且实际资源可行后，自动固定完整5k一次随机推理；范围、first64/protoROI资格、原正常输出与空ordinal不变，不根据工程AP选模型或加风险gate。若仍有真实阻塞回报具体事实，不无限改框架或继续扫batch。报告分别列原默认32与适配8证据，未测质量/区间保持未知。
