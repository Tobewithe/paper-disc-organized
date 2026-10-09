# 统一冻结 native 掩码比较

2026-10-09。本固定比较已完成：完整5k主要评分、像素/Boundary读出，笔记本及桌面独立单方法成本与各项独立核验全部通过。原失败Run保留；结论是已知方法的质量、损伤、覆盖与成本权衡，不是新增方法或独立盲测确认。

本轮复用官方冻结权重，共享一次 native forward，比较七个预先固定 arm；没有训练、参数搜索或重新挑选 epoch。主要范围为完整 COCO val2017 5,000 图、555,445 个正常原生候选。全部七臂保留相同框、类别、分数、顺序及空预测。first64 选择 229,596 个候选，其中 228,974 有原 proto ROI 支持；另 622 及范围外保持 baseline。

## 实际来源

- 推理与完整回传：[RUN_FROZEN_NATIVE_5K_INFERENCE_S0_02](runs/RUN_FROZEN_NATIVE_5K_INFERENCE_S0_02/run.json)。笔记本 28358lan，官方权重 SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`，vendor 8.4.100，640、FP32、one-to-one、conf>.001、max_det=300。
- 评分：[RUN_FROZEN_NATIVE_5K_SCORING_S0_01](runs/RUN_FROZEN_NATIVE_5K_SCORING_S0_01/SUMMARY.json)，实际桌面 CPU 执行、completed/exit0。SUMMARY SHA256 `313794cb3944ba329848cc513b10bd1d625f692a4f783d5e0224acc9ef8f62aa`。
- 独立核验：[RUN_FROZEN_NATIVE_5K_VERIFY_S0_02](runs/RUN_FROZEN_NATIVE_5K_VERIFY_S0_02/VERIFICATION.json)，completed/exit0、passed=true；VERIFICATION SHA256 `b3811dbbd8099a212add6fd44856887d49bbf3b167e56c40137c73a9bf005a3d`。从保存的累积数组重建 AP/AR，重聚合固定配对、图与 GT 统计及来源锁；没有重新执行 COCO matching、GT mask 计算或模型前向。
- 固定关联来源是 TriFlow final8 的 `RUN_TRIFLOW_20K_EPOCH08_EVAL_S0/paired/INSTANCES.jsonl`。本轮完整 baseline 预测与它的完整 baseline SHA 一致，12 项普通 segm 指标最大差 1.11e-16；因此可在相同 first64 可作用范围作有限比较，不能假定两种方法实际修改集合相同。
- 主协议：[PROTOCOL.md](PROTOCOL.md)；补充协议：[MULTIVIEW_READOUT_PROTOCOL.md](MULTIVIEW_READOUT_PROTOCOL.md)。失败与重试来由见 [EXECUTION_NOTES.md](EXECUTION_NOTES.md)，原失败 Run 和封存产物保留。

## 普通 COCO 指标

表中 AP/AR 均乘100；差值为 AP 点。所有 arm 的 Box AP=52.116074，框/类别/分数身份相同，bbox 只实算 baseline 一次。

| arm | AP | AP50 | AP75 | APsmall | APmedium | APlarge |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 43.685372 | 66.353710 | 47.294844 | 22.401825 | 47.670209 | 63.571734 |
| RCMC_full | 44.133110 | 66.508714 | 48.076407 | 23.032815 | 48.013874 | 63.581845 |
| global_minus025_full | 43.856341 | 66.327627 | 47.645164 | 22.624427 | 47.910351 | 63.473884 |
| RCMC_first64 | 44.133829 | 66.512778 | 48.059405 | 23.038589 | 48.015386 | 63.582006 |
| global_minus025_first64 | 43.858340 | 66.337348 | 47.641812 | 22.620409 | 47.920694 | 63.485337 |
| multi_local_full | 44.075839 | 66.514622 | 47.997985 | 23.036020 | 47.896070 | 63.593236 |
| multi_local_first64 | 44.074901 | 66.505283 | 47.981448 | 23.042068 | 47.892619 | 63.593711 |

| arm | AR1 | AR10 | AR100 | ARsmall | ARmedium | ARlarge |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 33.956846 | 55.421677 | 59.810376 | 41.433729 | 65.541988 | 76.727269 |
| RCMC_full | 34.208838 | 55.907211 | 60.364487 | 42.486284 | 65.652565 | 76.710783 |
| global_minus025_full | 34.091494 | 55.620319 | 59.975754 | 42.096349 | 65.545084 | 76.387615 |
| RCMC_first64 | 34.207267 | 55.923267 | 60.362887 | 42.534380 | 65.701390 | 76.713736 |
| global_minus025_first64 | 34.087160 | 55.621924 | 59.984400 | 41.975307 | 65.655562 | 76.516985 |
| multi_local_full | 34.182608 | 55.919731 | 60.397626 | 42.580041 | 65.679706 | 76.725542 |
| multi_local_first64 | 34.181184 | 55.945918 | 60.382798 | 42.578972 | 65.714980 | 76.733624 |

## 固定身份的修复与损伤

固定队列 86,600 个 detection：baseline Mask IoU≥.75 的成功 36,266 个，失败 50,334 个。损伤是成功跨到<.75，修复是失败跨到≥.75；相应率使用各自原分母。Mean IoU 使用全部固定关联候选。关联到 33,644 个唯一非 crowd GT，其中 21,210 个 GT 关联超过一个候选，不能把候选修复/损伤计数当作唯一 GT 数。

| arm | ΔAP 点 | Δmean IoU | 损伤数/36,266 | 损伤率 % | 修复数/50,334 | 修复率 % |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.000000 | 0.000000000 | 0 | 0.0000 | 0 | 0.0000 |
| RCMC_full | 0.447737 | 0.001675272 | 1250 | 3.4468 | 1826 | 3.6278 |
| global_minus025_full | 0.170969 | -0.006542669 | 1368 | 3.7721 | 1319 | 2.6205 |
| RCMC_first64 | 0.448457 | 0.001658204 | 813 | 2.2418 | 1280 | 2.5430 |
| global_minus025_first64 | 0.172968 | -0.001704471 | 762 | 2.1011 | 882 | 1.7523 |
| multi_local_full | 0.390466 | 0.002259890 | 1108 | 3.0552 | 1842 | 3.6596 |
| multi_local_first64 | 0.389529 | 0.001941907 | 757 | 2.0874 | 1312 | 2.6066 |

这些点估计显示权衡：first64 的 RCMC 提供约 +.448457 AP 点，但损伤率 2.2418%；同范围 multi_local 为 +.389529 点、损伤率 2.0874%。原 TriFlow final8 为 +.171018 点、损伤率 .9182%（333/36,266），保留它的原来源和原区间；本次未重跑完整 5k TriFlow 质量评估，32 图独立成本面板另行实际执行了 TriFlow。不能仅凭 AP 或修复数声称全面优胜。

full 相对 first64 的 AP 变化均小于 .002 点，同时增加修复与损伤。固定收缩虽然 AP 上涨，固定队列 mean IoU 下降，说明两种指标回答不同问题。当前 multi_local 是冻结门控选择五个原全局 mask 动作，不等于 F 的单连通分量局部动作。

## 像素与 Boundary

正式 [多角度读出](runs/RUN_FROZEN_MULTIVIEW_5K_READOUT_S0_01/SUMMARY.json) 与 [独立核验](runs/RUN_FROZEN_MULTIVIEW_5K_VERIFY_S0_01/VERIFICATION.json) 均 completed/exit0、产物完整，核验 passed=true。SUMMARY SHA256 `3dffb22124ee1e6cade098c51081424fbfddc33eda3b6f629911fca880352c4f`，VERIFICATION SHA256 `bd1d3fa24045f8b8662ff80e80f9b6895a83bf0b9d45bd1776928647737d52ee`。像素部分直接解码真实预测及GT沿原固定关联计算；核验重新聚合保存TP/FP/FN公式、候选/逐图表与Boundary累积数组，不独立重解码掩码、重构边界或重跑COCO匹配，框/分数身份继承原Score。

下面是全部固定配对的候选宏平均（每个有效detection等权）；coverage/IoU/FP/G/FN/G的分母为86,600。purity空预测未定义；其差值只用双方均有效的候选，不能直接把各自不同分母的均值相减。未关联468,845个候选保留身份并记GT指标null。

| arm | IoU | coverage | purity | FP/G | FN/G | Δcoverage | Δpurity | Δpurity有效分母 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.667597 | 0.809476 | 0.788392 | 0.275537 | 0.190524 | 0.000000 | 0.000000 | 86596 |
| RCMC_full | 0.669272 | 0.792887 | 0.805487 | 0.243510 | 0.207113 | -0.016589 | 0.017095 | 86596 |
| global_minus025_full | 0.661055 | 0.783880 | 0.803029 | 0.237782 | 0.216120 | -0.025596 | 0.014559 | 86561 |
| RCMC_first64 | 0.669255 | 0.799133 | 0.799531 | 0.255089 | 0.200867 | -0.010343 | 0.011140 | 86596 |
| global_minus025_first64 | 0.665893 | 0.796845 | 0.797408 | 0.255019 | 0.203155 | -0.012631 | 0.009021 | 86589 |
| multi_local_full | 0.669857 | 0.792075 | 0.806708 | 0.240255 | 0.207925 | -0.017401 | 0.018316 | 86596 |
| multi_local_first64 | 0.669539 | 0.798197 | 0.800667 | 0.252757 | 0.201803 | -0.011279 | 0.012275 | 86596 |

图像宏平均先在图内按该字段有效候选平均，再按有有效值图片等权；下表有效分母是图片数，完整三组（all/baseline_success/baseline_failure）的字段数值、原像素和分母保留在 `PIXEL_SUMMARY.json` 与候选/逐图产物中。

| arm | IoU | coverage | purity | FP/G | FN/G | Δcoverage | Δpurity | Δpurity有效分母 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.730019 | 0.847295 | 0.834275 | 0.213318 | 0.152705 | 0.000000 | 0.000000 | 4946 |
| RCMC_full | 0.731401 | 0.835628 | 0.846534 | 0.191090 | 0.164372 | -0.011667 | 0.012259 | 4946 |
| global_minus025_full | 0.725511 | 0.827684 | 0.846551 | 0.185497 | 0.172316 | -0.019612 | 0.012250 | 4946 |
| RCMC_first64 | 0.731415 | 0.838660 | 0.843644 | 0.196667 | 0.161340 | -0.008636 | 0.009369 | 4946 |
| global_minus025_first64 | 0.728052 | 0.834157 | 0.843800 | 0.193658 | 0.165843 | -0.013139 | 0.009527 | 4946 |
| multi_local_full | 0.731813 | 0.834529 | 0.847928 | 0.188345 | 0.165471 | -0.012767 | 0.013653 | 4946 |
| multi_local_first64 | 0.731665 | 0.837462 | 0.845011 | 0.194299 | 0.162538 | -0.009833 | 0.010736 | 4946 |

global full新增91个全输出空mask，其中固定配对新增35个；global first64新增15个空mask，其中固定配对新增7个。其purity有效分母相应下降，不能以丢掉空预测后的单侧均值声称无代价变纯。RCMC与multi保持原33个全输出空mask（固定配对4个）。这些指标来自同一像素记账，不作独立机制证明。

Boundary 固定 dilation_ratio=.02，使用完整5k正常COCO关联及 crowd/ignore/maxDets=[1,10,100]，不是固定pair关联。六个实际vendor Python源内容锁通过，上游commit仍null；该backend普通baseline12项AP/AR与本次Score实际值核对通过。以下AP/AR乘100。

| arm | AP | AP50 | AP75 | APsmall | APmedium | APlarge |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 29.571367 | 60.175578 | 26.083138 | 22.379697 | 41.126481 | 32.019551 |
| RCMC_full | 29.984555 | 60.353591 | 26.838783 | 23.010111 | 41.536639 | 32.044843 |
| global_minus025_full | 29.897245 | 60.008092 | 26.754514 | 22.606076 | 41.583370 | 32.333765 |
| RCMC_first64 | 29.982693 | 60.358389 | 26.818901 | 23.015773 | 41.534315 | 32.044986 |
| global_minus025_first64 | 29.895137 | 60.008488 | 26.748093 | 22.601930 | 41.586194 | 32.336837 |
| multi_local_full | 29.931268 | 60.335069 | 26.729445 | 23.012588 | 41.503928 | 31.972200 |
| multi_local_first64 | 29.928266 | 60.333097 | 26.715324 | 23.018620 | 41.501199 | 31.972173 |

| arm | AR1 | AR10 | AR100 | ARsmall | ARmedium | ARlarge |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 24.535739 | 42.697683 | 46.825296 | 41.372572 | 58.652001 | 44.123242 |
| RCMC_full | 24.811334 | 43.240879 | 47.449689 | 42.430455 | 58.899781 | 44.148915 |
| global_minus025_full | 24.818823 | 43.133307 | 47.251452 | 42.056763 | 58.962101 | 44.253318 |
| RCMC_first64 | 24.809616 | 43.248136 | 47.431749 | 42.477419 | 58.915846 | 44.150717 |
| global_minus025_first64 | 24.813953 | 43.120467 | 47.222069 | 41.934559 | 59.031279 | 44.310922 |
| multi_local_full | 24.812606 | 43.275982 | 47.502659 | 42.523152 | 59.007535 | 44.114473 |
| multi_local_first64 | 24.808495 | 43.297833 | 47.482141 | 42.522187 | 59.041155 | 44.128659 |

first64 RCMC 的 Boundary AP相对baseline提升约+.411326点，multi约+.356899点；面积与边界AP提升仍伴随coverage下降和原成功损伤，不形成全面优胜结论。

## 实际成本与结论边界

共享七臂推理总耗时 6,431.450 秒（约107.2分钟），GPU模型171.162秒，动作/特征/门控及RLE相关段5,560.720秒；分段存在包含关系，不全相加，也不分摊为单臂部署时长。CUDA allocated/reserved峰值1,742,643,200/5,505,024,000 bytes、推理父CPU peak1,506,361,344 bytes。桌面Score/Verify_02实际434.894/323.120秒，Multiview/Verify实际2210.815/198.884秒；其中pixel1016.754、Boundary1112.814秒，读出父CPU peak4,441,374,720 bytes不含vendor子进程。此前TriFlow709.739ms包含不同诊断整理边界，不与下面实测作排名。

完整推理包35,047文件、5,366,495,041 bytes已逐文件回传核验；压缩包SHA256 `1e9aa3efe42b467d6eca03bf221f2ed299ccbd587459169bb6e37867de4b9158`。原 remote 路径保留为 provenance，本地消费者按真实本地 Run 读文件，不改写原终态元数据。

当前七臂 AP 与固定配对置信区间未计算，保持未知。历史 val 前500曾用于规则开发、旧4500也已用于探索；本轮5k是固定配置统一比较，不称新的盲测、独立泛化确认或新方法贡献。当前结果不证明完整 raw 池、TAL监督机制或连续像素域能力。该比较与已停止的有限 F 局部动作机会试验独立，不据此扩张 F 动作或给 TriFlow 加补丁；B已接受完整有限结果，并另立官方SegRefiner强对照与同容量监督目标实验，不改变本Study判据。

## 独立单方法成本

[原成本协议](COST_PANEL_PROTOCOL.md) 固定原清单前32张、每臂4图warmup、3重复，共480个测量（每臂32张×3，不是96张独立图）、15个串行fresh进程。端点为计时外已读RGB→该臂预处理/官方forward/必需动作与特征/gate→全部正常原图binary masks CPU；初始化/warmup、RLE/真实参考审计/写盘都单列在端点外。每臂只算自身必要路径，不用七臂联合计时均分。原图与所有候选/空身份保留；57,195条保留mask记录经独立核验重构RLE/uint8 hash与参考一致。

笔记本来源：[PANEL_S0_01](runs/RUN_FROZEN_NATIVE_COST_PANEL_S0_01/SUMMARY.json)，SUMMARY SHA256 `d280e83eafbfc25bb69fb2ab8aebf2c82d6951786a02f9b34ee4806efb757206`；[独立核验](runs/RUN_FROZEN_NATIVE_COST_PANEL_VERIFY_S0_01/VERIFICATION.json) SHA256 `b7ee364ab0a06be5a577d1c98ef9ac229fce0155779f13e0784a26311d85c5ba`、passed=true。实际Torch2.5.1/CUDA12.4/NumPy2.4.3/RTX4060 Laptop、threads4/TF32false。该32面板输出与原冻结first64/final8参考严格一致；算法对应原5k质量结果，但没有把32时长外推成5k全量成本或声称优化隔离路径在5k逐字节重新验证。

| arm | mean ms | median ms | std_population ms | p10/p90 ms | mean paired Δms | median paired Δms |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 59.032 | 54.059 | 15.696 | 43.128/80.435 | 0.000 | 0.000 |
| RCMC_first64 | 77.364 | 79.388 | 20.242 | 51.588/101.153 | 18.333 | 20.311 |
| multi_local_first64 | 348.072 | 422.668 | 122.256 | 152.454/454.752 | 289.041 | 360.052 |
| global_minus025_first64 | 72.590 | 69.285 | 19.787 | 48.037/104.842 | 13.559 | 13.545 |
| TriFlow_final8 | 737.545 | 831.915 | 254.279 | 337.924/963.743 | 678.513 | 769.630 |

桌面来源：[LOCAL_PANEL_S0_01](runs/RUN_FROZEN_NATIVE_COST_DESKTOP_LOCAL_PANEL_S0_01/SUMMARY.json)，SUMMARY SHA256 `30638e098fe9eac8da1bc994771c035a1c7afa51c8e278bac69dad1047db1349`；[独立核验](runs/RUN_FROZEN_NATIVE_COST_DESKTOP_LOCAL_PANEL_VERIFY_S0_01/VERIFICATION.json) SHA256 `c8cfc86a1666eb8982aca842a2dcf97b19a62775ad335a6df93e202eb04183b4`、passed=true。实际Torch2.9.1+cu128/CUDA12.8/NumPy2.2.6/RTX5060Ti、threads4/TF32false；使用独立 [桌面同机参考协议](COST_PANEL_DESKTOP_LOCAL_REFERENCE_PROTOCOL.md) 和原七臂/原TriFlow路径生成的32图参考，没有由待测隔离executor反向生成自己参考。共同baseline/input/native身份先严格通过；此表只表示本机实现成本，不拼成已验证桌面5k质量—成本点、不混合两机样本。

| arm | mean ms | median ms | std_population ms | p10/p90 ms | mean paired Δms | median paired Δms |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 36.732 | 33.745 | 7.156 | 29.862/49.083 | 0.000 | 0.000 |
| RCMC_first64 | 47.951 | 47.405 | 10.662 | 34.074/63.308 | 11.219 | 12.695 |
| multi_local_first64 | 196.261 | 238.831 | 67.326 | 90.214/256.346 | 159.529 | 201.672 |
| global_minus025_first64 | 41.475 | 39.544 | 8.635 | 31.855/55.532 | 4.743 | 5.107 |
| TriFlow_final8 | 247.107 | 293.217 | 82.391 | 116.606/319.280 | 210.375 | 257.875 |

旧桌面01在GPU前path-map字节SHA守卫失败；02真实GPU前向后严格跨环境record parity失败，都保持failed。首图139的输入和全部300个binary/RLE确相同，类别/ordinal相同，但296条记录有框/置信度数值差异（raw_conf最大2.07126e-6、raw box最大.00036621）；该结论仅首图，不扩张为全32或5k跨设备质量等价。新same-machine参考/工程/正式各Run独立保留，未放宽旧判据。

独立核验重构保存mask/RLE、来源/真实输入绑定与有符号成本统计，不重新执行GPU/forward或重测墙钟/内存。负增量样本保留；均值/分位数分布包含图像差异与重复噪声，CI未测。3个循环左移顺序减轻漂移但非完整位置平衡。TriFlow使用已锁minimal诊断核心，不称零诊断；child process_wall是整个生命周期，纯启动时段未直接测定保持null。GPU测前reset的peak包含resident/暖cache，reserved是热态容量，非单图新增或最低需求；CPU fresh-process lifetime peak包括初始化/warmup/此前审计，不作纯部署RAM峰值排名。完整各臂/重复峰值及phase值留原统计与PROCESS_PHASES。

桌面正式Boundary重CPU负载已结束后运行，前置和child资源采样heavy标志为0；采样覆盖整个child生命周期，不对齐纯endpoint，采样开销未知，不证明完全无负载干扰。真实工程4图无unsupported ROI；正式面板出现108条跨方法/重复合计unsupported行，但没有empty样本，不冒称实测覆盖空分支；一般空/fallback保障沿原源合同。

