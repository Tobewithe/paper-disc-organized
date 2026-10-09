# 单方法增量成本小面板

2026-10-09，B 授权补足同作用范围的部署成本；原七臂推理、AP、固定关联及多角度读出不变。本协议在成本面板执行前固定，属于工程成本测量，不作能力或泛化结论。

固定原 `val_full.txt`（SHA256 `b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db`）原顺序前32张，不按任何结果选图。面板图片及尺寸/字节摘要、实际顺序在 Run 留存。五臂为 baseline、RCMC_first64、multi_local_first64、global_minus025_first64 和原 TriFlow final8；统一正常输出、first64/protoROI 支持、原生640/FP32口径、官方权重和 fallback/空预测身份。global contraction 指 logit−.25，即阈值+.25，不是 F 的阈值−.25扩张。

实际使用原离线笔记本 `ssh 28358lan`、`C:/Users/28358/anaconda3/envs/pytorch/python.exe`（Torch2.5.1）、`D:/coco_wire/vendor_8.4.100` 与官方权重 SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。每臂只执行自己需要的动作、特征和 gate；在计算这些方法路径前限制 first64，保留必要的完整 native GEMM 以保持数值。TriFlow 使用原终轮固定 head，不训练、不选 epoch。不能将原七臂联合工作均分成单臂成本。

正式3次重复，每块五臂顺序从上述列表循环左移 repeat_index 位；每 arm×repeat 使用一个新子进程，GPU不并发。各进程独立加载官方模型与本臂必需资产，固定前4张各warmup一次后，原顺序测32张。初始化、读图、warmup及子进程启动各自记录，排除部署计时；重复块顺序用于减轻漂移，不称完整位置均衡。

共同部署端点为：计时外已读取的原始RGB ndarray → 预处理 → 本臂官方 forward/native后处理与方法 → 全部正常候选的原图binary mask均已导出CPU，身份/顺序/空/fallback齐全。每次开始前CUDA同步，结束确保CUDA完成；以wall clock报告端到端，保留每图每重复原始值、均值/中位数/std/p10/p90和同image/repeat相对baseline的有符号增量，不截断负增量。RLE编码、逐字节/身份审计、写盘、COCO评分均在此端点外分列，不混进部署成本；本成本Run不跑COCO评分或GT计算。

GPU测前重置 allocated/reserved peak；CPU Windows累计peak不能伪称可重置，按各新arm-repeat进程独立peak记录并明确包含初始化/warmup。若另采样post-warmup RSS，注明采样间隔、采样范围及非精确峰值的限制。不能把先前七臂或另一进程的峰值分摊成本。

每次测量的真实完整输出在计时外与已冻结对应 first64/原 final8 面板输出核对身份、空ordinal及binary/RLE parity，保留审计与来源SHA；任一不符保留失败，不以近似路径或共享联合时间代替。新增独立成本Run，记录实际解释器/设备/输入输出、当前协议与源码/资产锁；原Run不改写。若工程需要修复，失败保留，新Run重试，原算法/阈值不变；无法隔离某臂则报告该实质限制。

结果只描述这32张、此设备/环境/端点的实测开销。启动、审计/保存与部署时间分别解释，原TriFlow709.739ms的不同诊断边界及七臂联合计时不参与部署快慢排名；未测置信区间保持未知。
