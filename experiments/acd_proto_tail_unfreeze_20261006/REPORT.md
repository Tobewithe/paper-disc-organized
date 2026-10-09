# ACD prototype-tail unfreezing feasibility experiment

更新于 2026-10-06T23:29:37+08:00（Asia/Shanghai）。

结论：**本次可行性门槛未通过**。有效性核验通过，但至少一个预定义科学阈值未通过；保留本次未达门槛的结果并停止调参。

主比较为 ACD − 同范围 coefficient+proto-tail baseline；official 为额外参考。
negative 表示未达到预设门槛，不能据此断言方法整体无效。

| 协议 gate | 判定 |
|---|---|
| 协议锁 | 通过 |
| baseline 固定训练预算/版本/数据/BN | 通过 |
| baseline 四个 trainable prefixes/2,306,240 参数/全 BN 冻结 | 通过 |
| baseline 有限 live loss | 通过 |
| baseline 实际应用梯度有限且非有限 attempt 均确认由 GradScaler 跳过 | 通过 |
| baseline frozen 参数及 BN buffers 不变 | 通过 |
| baseline one2many 实际应用梯度有限且每个 prefix 更新 | 通过 |
| baseline one2one 实际应用梯度有限且每个 prefix 更新 | 通过 |
| baseline proto_tail 实际应用梯度有限且每个 prefix 更新 | 通过 |
| acd 固定训练预算/版本/数据/BN | 通过 |
| acd 四个 trainable prefixes/2,306,240 参数/全 BN 冻结 | 通过 |
| acd 有限 live loss | 通过 |
| acd 实际应用梯度有限且非有限 attempt 均确认由 GradScaler 跳过 | 通过 |
| acd frozen 参数及 BN buffers 不变 | 通过 |
| acd one2many 实际应用梯度有限且每个 prefix 更新 | 通过 |
| acd one2one 实际应用梯度有限且每个 prefix 更新 | 通过 |
| acd proto_tail 实际应用梯度有限且每个 prefix 更新 | 通过 |
| 实际模型 one2many 原型尾部梯度/one2one detach/官方单步诊断 | 通过 |
| 完整 prototype-tail FP32 overlay 的独立 loader 合同 | 通过 |
| 正式训练/评估/诊断已完整回传且原 source manifest 保留 | 通过 |
| 配对 transformed-input 完整流一致 | 通过 |
| 两臂 optimizer attempt 的 batch/LR/accumulate/AMP schedule 一致 | 通过 |
| DATA_RECEIPT 与 fit/internal-val/smoke/full-val 列表哈希及数量 | 通过 |
| 完整 5,000 张原始 COCO val2017 独立评估 | 通过 |
| 固定第 3 epoch EMA 四-prefix mask-state overlay 官方 FP32 推理 | 通过 |
| official/baseline/acd frozen FP32 tensors digest 一致 | 通过 |
| ACD/baseline box/class/confidence/detection identity 直接逐图一致 | 通过 |
| Mask AP：ACD − baseline >= +0.003 | 未通过 |
| Mask AP75/APsmall 不同时下降 | 通过 |
| 配对损伤诊断有 baseline-success 样本且图像 bootstrap 完整 | 通过 |
| baseline-success 损伤率 <= 1% | 通过 |
| baseline-success 平均 mask IoU 变化 >= −0.005 | 通过 |

未知不能当成通过。执行或 frozen/data/parity 核验无效时，不将 AP 差解释为训练迁移结论。
固定预算结束后停止；不自动加 seed、修改 auxiliary 设置或扩展训练。

## 固定范围及机制限制

本轮 baseline 和 ACD 均从相同官方 COCO yolo26m-seg.pt 重启，Ultralytics 8.4.100。
更新完整原生 cv4/one2one_cv4 两个系数头，以及 model.23.proto.cv2/cv3 的 non-BN 参数；
原型前端、upsample、refinement/fusion、semantic、backbone/neck、box/class 与全部 BN affine/running state 冻结。
官方 m 预期可训练参数：系数 1,708,224 + prototype tail 598,016 = 2,306,240，实际 names/count 必须核验。

保留官方 one2one prototype detach。原型尾部仅接收 one2many mask/ACD 路径的梯度；
one2one ACD 直接训练其系数头。三轮 one2many/one2one 权重为 0.8/0.2、0.45/0.55、0.1/0.9。
这项梯度不对称和后期衰减是当前配置的限制，不能由短屏结果推断 prototype 或 ACD 普遍无效。

保留 MuSGD 的官方参数名称分组：prototype cv3 为 3× group LR，cv2 为普通 group LR。
保留官方 global clip_grad_norm_(model.parameters(), 10)：新增可训练原型参数会改变联合裁剪范数，
可同时影响系数梯度缩放和 AMP overflow 轨迹。可训练范围、LR 分组及联合裁剪共同构成本次干预；
结果不能单独证明 prototype 表示能力的充分性或不足。

先做独立梯度/optimizer 与 loader 合同诊断，再各做 1 epoch smoke；有效后从官方权重重启各固定 3 epochs。
seed=0、batch=2、workers=0、imgsz=640，固定官方支持的超参数、AMP、augmentation 和 overlap-mask 语义。
正式 fit=796、内部转换标签 val=196；dev=197 不用于拟合或挑设置。1,194 input batches 不等于 1,194 optimizer updates。
fit cohort 及全 val2017 均为历史研究复用，包含选择限制；全验证集不是新 blind test。

ACD 不变：weight=0.05、tau=0.50、min gain=0.01、min coverage=0.80、每图每 loss branch 首 12 candidates。
接受候选中所有 positive-logit 背景像素参与 softplus penalty，限 assigned GT support box；
不只学习 action 删除的像素，亦不是完整 tightened-mask/logit distillation。训练 GT 选择辅助项，不进入推理。

最终插入固定第 3 epoch 的完整 coefficient+prototype-tail FP32 EMA state（mask_final_ema.pt），
kind=mask_coefficient_proto_tail_ema_final、epoch=3、audit_passed=true；四 prefix 共 96 state keys，BN 仍必须逐值等于官方。
不选 best.pt。全 5,000 原始 COCO val2017 独立评价采用 native one2one、FP32、tf32=false、square 640、scaleup=false、
conf=0.001、max_det=300，官方 JSON decoder/原图 scale_preds.byte()、COCO80→91、pycocotools maxDets=[1,10,100]。

完整协议：[PROTOCOL.md](PROTOCOL.md)。固定屏结束即停止，不扩 seed、不改 detach/auxiliary/backbone/neck、不延长训练。
有效但未达门槛为当前配置的负屏；缺失或无效证据保持未知。

## 原始 COCO 全验证集结果

COCO AP 以百分数展示，差值为百分点；未知表示缺失、未回传或非有限值。

| 评价 | 模型/比较 | AP50:95 | AP50 | AP75 | APsmall | APmedium | APlarge |
|---|---|---:|---:|---:|---:|---:|---:|
| Mask | official | 43.685 | 66.354 | 47.295 | 22.402 | 47.670 | 63.572 |
| Mask | baseline | 43.445 | 66.323 | 47.054 | 22.387 | 47.460 | 63.071 |
| Mask | acd | 43.488 | 66.386 | 47.036 | 22.446 | 47.523 | 63.188 |
| Mask | acd − baseline | +0.043 | +0.063 | -0.019 | +0.060 | +0.063 | +0.117 |
| Mask | acd − official | -0.198 | +0.033 | -0.259 | +0.045 | -0.147 | -0.384 |
| Mask | baseline − official | -0.241 | -0.031 | -0.240 | -0.015 | -0.210 | -0.501 |
| Box | official | 52.116 | 69.141 | 56.935 | 33.955 | 57.012 | 68.020 |
| Box | baseline | 52.116 | 69.141 | 56.935 | 33.955 | 57.012 | 68.020 |
| Box | acd | 52.116 | 69.141 | 56.935 | 33.955 | 57.012 | 68.020 |
| Box | acd − baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| Box | acd − official | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| Box | baseline − official | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

不提供 AP 置信区间；bootstrap 仅用于图像簇配对的 IoU/损伤诊断。
原型尾部学习和 baseline 普通微调都可能改变 Mask AP；ACD − official 也保留以显示微调是否退化。

## 输出层配对损伤与修复

配对定义：每个 baseline detection 以同类 box IoU ≥ 0.5 匹配最高重叠的 noncrowd GT；
baseline 原图 mask IoU ≥ 0.75 为成功，ACD 降至 < 0.75 为损伤，反向越过该阈值为修复。
允许多个 detection 匹配同一 GT，计数是 detection identity 的诊断，不能当成 COCO AP 一对一匹配。

配对可用性：通过。

| 计数 | 数值 |
|---|---:|
| 配对匹配 detections | 86600 |
| baseline-success | 35652 |
| 成功组损伤 | 207 |
| baseline-failure | 50948 |
| 失败组修复 | 240 |
| 评价图像 | 5000 |
| 含匹配 detection 的图像 | 4946 |
| unique matched GT | 33644 |

| 图像簇配对诊断 | 点估计 | 95% bootstrap 区间 |
|---|---:|---:|
| 所有匹配 detection 平均 mask IoU 变化 | -0.000313 | [-0.000369, -0.000255] |
| baseline-success 平均 mask IoU 变化 | 0.000216 | [0.000177, 0.000257] |
| baseline-success 损伤率（%） | 0.5806 | [0.4998, 0.6599] |
| baseline-failure 修复率（%） | 0.4711 | [0.4104, 0.5359] |
| baseline 平均 mask IoU | 0.662249 | [0.658890, 0.665579] |
| ACD 平均 mask IoU | 0.661936 | [0.658553, 0.665272] |

bootstrap：5000 次，seed=0；
单位为图像簇，两臂配对抽样，以簇总和之比统计。记录定义：supplied images, paired across arms; ratio of clustered sums。
区间描述当前固定模型与该验证集的诊断变化，不量化不同训练 seed 的不确定性。

raw score-independent 五状态分类、pixel AUC/FPR、crop-support ceiling：**未知，本屏未测量**。
conf/max_det 筛选后的输出配对不等于 raw GT geometry、最大一对一匹配、COCO recall 或 AP 不确定性。

## 与 coefficient-only 条件的描述性比较

与 coefficient-only 原始 COCO Run 的可比性：通过。
原始 Run=RUN_ACD_PAIRED_EVAL_S0；其独立读出链核验=通过。

原始数据来源：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_PAIRED_EVAL_S0/SUMMARY.json`。这里只使用该固定末 epoch 的原始 COCO 结果。
不纳入 best.pt 或其他历史方法的 AP 收益。

| 条件与模型 | Mask AP | AP50 | AP75 | APsmall | APmedium | APlarge |
|---|---:|---:|---:|---:|---:|---:|
| coefficient-only / official | 43.685 | 66.354 | 47.295 | 22.402 | 47.670 | 63.572 |
| coefficient-only / baseline | 43.577 | 66.284 | 47.198 | 22.417 | 47.659 | 63.301 |
| coefficient-only / acd | 43.614 | 66.279 | 47.282 | 22.522 | 47.679 | 63.301 |
| coefficient+proto-tail / official | 43.685 | 66.354 | 47.295 | 22.402 | 47.670 | 63.572 |
| coefficient+proto-tail / baseline | 43.445 | 66.323 | 47.054 | 22.387 | 47.460 | 63.071 |
| coefficient+proto-tail / acd | 43.488 | 66.386 | 47.036 | 22.446 | 47.523 | 63.188 |
| coefficient-only Δ_ACD=ACD−baseline（百分点） | +0.0372 | -0.0058 | +0.0837 | +0.1053 | +0.0203 | -0.0004 |
| proto-tail Δ_ACD=ACD−baseline（百分点） | +0.0432 | +0.0634 | -0.0186 | +0.0598 | +0.0632 | +0.1170 |
| Δ_ACD 交互=proto-tail效应−coefficient-only效应（百分点） | +0.0060 | +0.0692 | -0.1024 | -0.0455 | +0.0428 | +0.1173 |

可比性要求初始化 SHA256、fit/internal-val/full-val 列表及实际本地列表哈希、原始 annotations、图像 ID、
完整推理配置、版本/官方 decoder 源码与 original-official predictions SHA256 一致；
且本轮执行有效、原实验三个臂的 prediction/COCO metrics/独立读出链均已核验。
缺失或不一致时交互保持未知；上表各自原始数值不自动构成跨条件可比结论。
仅同一原始 COCO 全验证集、同初始化/列表/推理配置的描述性交互；单 seed，非显著性或独立因果分解。不纳入 best.pt 或其它历史改法的收益。

## 配对训练数值轨迹

| 正式训练臂 | 选用 Run | attempted | applied | confirmed overflow skips | observed unscaled nonfinite attempts | EMA updates |
|---|---|---:|---:|---:|---:|---:|
| baseline | RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0 | 69 | 66 | 3 | 3 | 69 |
| acd | RUN_PROTO_TAIL_ACD_FEASIBILITY_S0 | 69 | 66 | 3 | 3 | 69 |

overflow-skip 次数差（ACD − baseline）：+0。
已直接核验两臂 overflow-skip 的 attempt、epoch 和 batch 位置相同；损失与梯度值仍由各自训练条件决定。
非有限梯度观测不改称 finite；计数差异不自动归因于 ACD。
attempt 的 epoch/batch/LR/accumulate/AMP schedule 必须一致；applied/overflow-skip 数量允许按官方 AMP 行为分别记录。
EMA 保持官方时机：每次 optimizer attempt 都更新，包括 scaler 跳过 actual optimizer step 的 attempt。

## 数据列表交集与评价限制

按实际列表的 COCO 文件名数值 ID 和规范化路径字符串分别计算；未读取远端图片，也未核对图片字节。

| 集合 | 条目数 | 与 full-val 5000 的图像 ID 交集 | 路径字符串交集 | DATA_RECEIPT SHA256 匹配 |
|---|---:|---:|---:|---|
| [fit / 训练](data/fit.txt) | 796 | 0 | 0 | 通过 |
| [dev / 未用于拟合](data/dev.txt) | 197 | 0 | 0 | 通过 |
| [internal-val / 每 epoch 验证](data/val.txt) | 196 | 196 | 0 | 通过 |
| [smoke train](data/smoke_train.txt) | 32 | 0 | 0 | 通过 |
| [smoke val](data/smoke_val.txt) | 8 | 8 | 0 | 通过 |

现有列表可确认：fit 的 796 个图像 ID 与最终评估的 5000 个 ID 没有交集；
本轮列表没有显示把这些评估图像用于 fit 的证据。

内部验证的 196 张全部包含于 full-val（3.92%）；smoke 验证的 8 张也包含于其中。
converted-cache 与原始 COCO 图像目录不同，因此路径交集为 0 不能证明这些图像独立；同一 COCO ID 应视为重复图像身份。
这是内部验证复用，不等于 fit 泄漏。固定第 3 epoch、不挑 best 保留原设计，但 full-val 不能声称新的 blind heldout。

“独立评价”指原始 COCO annotations/decoder 的单独测量流程，不表示与内部验证、历史研究信息独立。
缺失列表或不可解析的 ID 保持未知，不把缺失计为零交集。交集数、列表 SHA256 与来源路径保存在派生 DECISION.json；
没有创建新的划分列表，也没有增加训练或评估。

## 独立诊断及未知包裹层时间

- 真实模型梯度/optimizer/EMA：原 receipt status=PASS，
  本地诊断证据核验=通过，与正式源码链接=通过。
  [RUN_PROTO_TAIL_NUMERICAL_VERIFY_S0](runs/RUN_PROTO_TAIL_NUMERICAL_VERIFY_S0/REPORT.md)；真实官方模型的 fixture GT/assignment + 一步 FP32 诊断，只证明梯度/optimizer/EMA/overlay 合同；不等于正式训练或 AP 结果。
- synthetic loader 合同：原 receipt status=PASS，
  本地诊断证据核验=通过，与正式源码链接=通过。
  [RUN_PROTO_TAIL_LOADER_VERIFY_S0](runs/RUN_PROTO_TAIL_LOADER_VERIFY_S0/REPORT.md)；synthetic loader fixture 的 epoch=3 是合同字段，不是三轮训练事实；只验证正确/错误 payload 的接受与拒绝。

诊断 Run 的 planned 模板曾覆盖 wrapper metadata；随后依据 observed exit=0/PASS receipt 恢复 completed 状态。
精确 wrapper started_at/ended_at 保持未知，receipt 自身诊断时间另行记录；本报告不填造包裹层时间，也不将其未知解释成诊断失败。

## Run 记录

| Run | arm / purpose | 执行状态 | 训练/评价/诊断 receipt | 回传状态 |
|---|---|---|---|---|
| [RUN_PROTO_TAIL_ACD_FEASIBILITY_S0](runs/RUN_PROTO_TAIL_ACD_FEASIBILITY_S0/REPORT.md) | acd / feasibility | completed | 训练 complete | verified |
| [RUN_PROTO_TAIL_ACD_SMOKE_S0](runs/RUN_PROTO_TAIL_ACD_SMOKE_S0/REPORT.md) | acd / smoke | completed | 训练 complete | verified |
| [RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0](runs/RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0/REPORT.md) | baseline / feasibility | completed | 训练 complete | verified |
| [RUN_PROTO_TAIL_BASELINE_SMOKE_S0](runs/RUN_PROTO_TAIL_BASELINE_SMOKE_S0/REPORT.md) | baseline / smoke | completed | 训练 complete | verified |
| [RUN_PROTO_TAIL_LOADER_VERIFY_S0](runs/RUN_PROTO_TAIL_LOADER_VERIFY_S0/REPORT.md) | 未知 / diagnostic | completed | 诊断 PASS | verified |
| [RUN_PROTO_TAIL_NUMERICAL_VERIFY_S0](runs/RUN_PROTO_TAIL_NUMERICAL_VERIFY_S0/REPORT.md) | 未知 / diagnostic | completed | 诊断 PASS | verified |
| [RUN_PROTO_TAIL_PAIRED_EVAL_S0](runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/REPORT.md) | paired / evaluation | completed | 评价 complete / 5000 图 | verified |
| [RUN_PROTO_TAIL_PIPELINE_S0](runs/RUN_PROTO_TAIL_PIPELINE_S0/REPORT.md) | 未知 / pipeline | completed | pipeline COMPLETE | verified |

## 失败尝试

当前本地可读记录未发现失败尝试；未回传状态不据此推断成功。

## 核验与未完成项

- Mask AP：ACD − baseline >= +0.003：未通过；证据 `runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/SUMMARY.json`。

独立读出：[READOUT_VERIFICATION.json](runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/READOUT_VERIFICATION.json)；
其 AP/AP75/APsmall 来自 COCO accumulated precision 复算，配对 counts/点估计来自逐图/逐 detection 记录复算。
它不重做 COCO matching、GT/mask decoder 或 bootstrap CI；当前文件哈希与 audit evidence 再次链接。

机器可读三值判定与原始证据 SHA256：[DECISION.json](runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0/DECISION.json)。
报告脚本：[build_report.py](scripts/build_report.py)。仅写 REPORT/DECISION，不改 study/run metadata 或原实验。
