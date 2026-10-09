# 统一冻结 native 掩码读出对照

用户已授权 A/B 持续协作和资源调用；A 实际执行，B 负责独立科学审查及方法推进。沿用 research-loop 3.0.0 固定正文 SHA256 `4d33b67d0a2fc22c63ecee2a1cb268303b82ef92ecae99336de5e71f2514eba2`。本问题是已有冻结读出方法的统一比较，不是新增方法、训练或参数搜索。

## 固定输入与环境

实际执行使用离线笔记本 `ssh 28358lan`；解释器 `C:/Users/28358/anaconda3/envs/pytorch/python.exe`，vendor `D:/coco_wire/vendor_8.4.100`，官方 `D:/coco_wire/models/yolo26m-seg.pt`。官方文件 SHA256 为 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。one-to-one、640 输入、FP32、conf>.001、max_det=300，各 arm 共用同一官方 forward 与原生候选，框、类别、分数、候选顺序与空 mask 身份保持。

正式范围为原 TriFlow `data/val_full.txt` 的 5,000 图，SHA256 `b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db`；图像来自 `D:/coco_wire/data/images/val2017`。原始标注 `D:/coco_wire/data/annotations/instances_val2017.json`，SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。不使用 val 调动作、门控、cutoff、样本或轮次。

历史 response 与 multi_local 的拟合/选择来自 train2017 的 1,500/500 划分；response 的实际 ID 已核对与全 val5k 无交叠（reproducibility/calibration_split.json SHA256 `1b75dfea2ca011dc43aaa8a7b863900c3bae913472b849f76934361366d9d8ce`）。旧 val 前 500 曾用于更早规则/阈值开发，旧 4,500 亦已用于探索，所以本次完整 val5k 是固定方法的统一比较，不称新的盲测或独立泛化确认。

工程清单固定为上述文件前 64 图，仅检验执行与真实成本，不作能力结论。实际清单、解释器、远端输入/输出、源码快照及摘要记录到独立 Run。工程与正式评价、独立核验分别建 Run，失败保留。

## 七个 arm

| arm | 固定行为 |
|---|---|
| baseline | 原生官方零阈值掩码 |
| RCMC_full | 对全部正常输出应用原冻结 response HGB；tau(A)=.75/(1+(A/2304)^2)，predicted_gain>0 才采用 trial；trial 新空回退 baseline |
| global_minus025_full | 全部正常输出的 logit 统一减 .25，即原 input-grid logit>.25；新空保留 |
| multi_local_full | 原冻结 18 特征模型预测五个固定动作 smooth、.25、.5、.75、1 的收益；空 trial 不可选；原动作顺序首次 argmax，max gain>0 才改，否则 baseline |
| RCMC_first64 | RCMC，同 TriFlow first64 和 protoROI 支持范围 |
| global_minus025_first64 | global−.25，同 TriFlow first64 和 protoROI 支持范围 |
| multi_local_first64 | multi_local，同 TriFlow first64 和 protoROI 支持范围 |

full 不套用 TriFlow 的 protoROI 支持条件；它们可以修改 input640 有支持但 proto 没有整数中心的细框。first64 固定为原 eligible 输出顺序的前 64，ROI 不支持则保持 baseline，不补位、不选择更优范围。两种范围均完整导出全部候选，范围外保持 baseline。

response 文件 SHA256 `eb2fca8b9bbb0192d62723e3da086c303729752772aa1a357ea64f61f70888b9`。multi_local 实际远端资产位于 `D:/coco_wire/runs/mask_boundary_route_20260914/RUN_a893d59e5d16441fadd42ef21e81a842/multi_local.json`，已只读核实为 196,839 bytes，SHA256 `71874d3f23e6cbe2179ce0962c751b7afd04b9e29581a0c3227515ca48b76987`；缺失/不符则执行失败，不重训替代。

严格沿原 portable gate、特征顺序与浮点处理。RCMC 的面积/宽/高使用原图导出二值像素，compactness 使用 input640 二值网格，不能改成 proto160。multi_local 18 维顺序为原五个 shape/response 特征、该动作的 tau、原 local12；原 local_features SHA256 `a0bdefe767f2fa38536e7c6261e32a1bc066238cc90bb7585de70490e8b0caf6`。显式 ratio_pad 与原生解码可能改变旧 gate 的特征域，所以结论是“旧冻结 gate 在统一 native 5k 的结果”，不声称旧 4,500 图评价字节复现。

## 身份、解码与验收

复用原 `frozen_io.py`（SHA256 `cff75eec547d9ec12eda5409bd273231ab99c16cdc6119d75f6d413098363293`），使用当次实际预处理 ratio_pad。保持原框裁切、input 网格阈值、逆变换、byte 和 COCO RLE 顺序；不能替换成连续 logits 先缩放到原图再阈值。

工程零动作必须与当次官方 native 掩码逐 byte/RLE 相同；全量正式 Run 的每图 baseline 必须与已核验官方缓存逐候选身份及 RLE 相同。原缓存 `D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/official`，也是 TriFlow 终轮的 baseline。检查原生 row/raw 身份、类别、confidence、bbox、导出顺序、empty 与 first64/unsupported 不变量，任何不符立即保留失败事实。

只在 baseline 全量一致条件下，与固定 TriFlow final8 的既有完整预测比较；不重跑 TriFlow、不重新选 epoch。TriFlow 来源 `STUDY_TRIFLOW_20K_TRAINING_20261007/RUN_TRIFLOW_20K_EPOCH08_EVAL_S0`，头状态 SHA256 `9bbe280e1de3c61a92a2c4595aeb3b5e855665a105112b39ae9646a5602772e8`。此前训练和推理时间不能充当本次统一计时。

## 指标、成本与结束条件

标准 COCOeval Mask AP/AP50/AP75/大小指标为主要部署比较（表内乘100，增益为 AP 点），覆盖全部 5,000 图 GT 和全部正常输出，保留 crowd/ignore 与 maxDets=[1,10,100]。baseline Box AP 评价一次，所有 arm 框/分数/类别相同另作身份断言。

固定身份机制表沿原 final8 的 paired `(image_id,detection_index,annotation_id)` 关联计算各 arm 的正常原图 Mask IoU；保留原 success>=.75、修复/损伤的各自成功/失败分母，不以 GT 重复候选数冒充唯一 GT 数。新正常任务 COCOeval 与固定身份机制表分开，缺失字段保持未知。按完整图像配对重采样可复用原定义，若未实际运行则区间未知；不以 repair 区间替代 AP 区间。

工程先测同步 forward、action 解码/特征/gate、逆变换/RLE、落盘及评价成本；正式只共享一次 forward 与可复用动作计算，日志保存实际耗时和峰值内存。工程估计用于给出正式成本，不将旧耗时或计划值当实测。单次后台运行，不创建定时器或监控自动化，不停止其他环境任务。

所有固定 arm 完成且核验通过后结束本比较；不可用或身份失败则记录实际失败并修复执行口径，不扫阈值、重训 gate 或增加动作追收益。本结果用于决定新方法应超过哪个已知读出对照。训练侧局部动作机会试验在 B 的独立 Study 登记，清单、标注用途与 val5k 严格分开。
