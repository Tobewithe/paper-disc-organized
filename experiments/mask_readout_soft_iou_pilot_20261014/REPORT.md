# 同 ROI soft-IoU 单变量复验结案

**拒绝当前 soft-IoU 配置进入 S3 或扩量。** 这次损失替换在相同开发图的目标组上优于原 BCE 配置，但原生基线参照下的原 S2 三项联立门槛全部未通过：目标均值达到最小增益，置信区间下界未过零；整体固定配对 IoU 下降超过容许值；原成功 GT 的损伤上界超过1%。这是该初始化、2000图、3轮预算下本配置的否决，不是 soft-IoU 或系数残差方法的普遍无效结论。

## 真实执行与回传

2026-10-14，Asia/Taipei：笔记本 YANGLEI 的独立后台任务于02:29:11完成全部评价、配置复核和打包；本次20:44:15现场核验任务退出码0，无Python进程、无Study锁或控制器锁。未追加训练，未启动S3。执行端根目录为 `D:\coco_wire\baseline_native_20261013`。

| 阶段 | Run | 实际终态 |
|---|---|---|
| 工程smoke | `RUN_SOFT_IOU_SMOKE_2cb8fd414d1540dcb4a16b34b0c0ac37` | 8张fit、8张dev零残差RLE一致；4项loss行为测试通过；2次更新仅作工程检查 |
| 正式训练 | `RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb` | 02:22:59完成；3轮、6000图访问、1500次AdamW更新；checkpoint/全日志独立审计通过 |
| 原S2评价 | `RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0` | 02:28:50完成；1314张评价图、185621条最终输出；工程及配置通过，科学Gate失败 |

训练51项、评价52项证据已回传并核对SHA；本机重新检查训练checkpoint、完整日志与原B的配置，并从逐GT/固定配对表独立复算bootstrap、损伤抽样及超几何单侧上界，结果与执行端一致。为复算成功总体另补回既有S0全GT账本，SHA与冻结合同相符。67,247,972字节的完整预测JSON保留笔记本，现场重新计算SHA与Run一致；缓存和数据未复制或重建。

## 原 S2 判定

效用均相对原生 S0，ΔIoU单位为0–1。原512张dev中，T=655个GT、257张图有目标配对；整体=2925个普通bbox75配对GT、508张图有配对。均按原先的图宏平均与2000次、seed1729、linear percentile bootstrap计算。

| 原门槛 | 本次结果 | 裁决 |
|---|---|---|
| 目标ΔIoU≥0.005，且95%区间下界>0 | **+0.006798976**；95% CI **[−0.001266561, +0.016392694]** | **失败**：均值过线，区间下界未过零 |
| 全部普通bbox75配对的ΔIoU≥−0.001 | **−0.002618361**；95% CI [−0.004032465, −0.001138054] | **失败** |
| 全部原成功GT总体的损伤单侧95%上界≤1% | N=22957；n=1024；损伤**53**；点估计**5.1758%**，上界**1477/22957=6.4338%** | **失败** |

同一512张dev、同一固定检测配对上，相对旧BCE配置的目标ΔIoU差为 **+0.016755282**（+1.6755个百分点），配对开发集95% CI为[+0.006562881,+0.028045183]。这支持本次loss替换改善了相对旧BCE的目标效果；它不替代相对原生基线的S2验收，不能因优于一个失败配置就推进。

新checkpoint冻结后按原损伤设计一次性抽样1024个GT，覆盖886张图；与dev合并为1314张图。旧BCE评价的1299张并集来自另一次损伤抽样，不能把两者的损伤个数或辅助AP当作同样本配对改善。**设计、预算、总体和门槛未变，损伤抽签按原协议重新进行**。1314图辅助AP不是全5000图AP，本报告不将它与官方发布值或全量原生基线拼表。

## 配置复核

| 检查对象 | 核验事实 |
|---|---|
| 初始化与训练数据 | 原B `initial.pt`、逐轮 `IMAGE_ORDERS.json`、`TRAINING_JOIN.json`逐字节一致；同2000图、同GT/ROI投影；全新AdamW状态 |
| 训练预算与数值 | seed0，3轮、1500步；微批1、累积4；AdamW lr=.001、betas=(.9,.999)、eps=1e-8、weight_decay=.0001及其余开关相同；无AMP/增强/调度/早停；最后一轮checkpoint；167872个新增参数，各参数记录1500步 |
| 唯一研究变量 | GT BCE→同ROI `1−(I+1e−6)/(U+1e−6)`；空GT沿同公式计入原分母；原保真BCE及0.1权重、归约和训练循环保持不变 |
| 原生输入与推理 | Ultralytics8.4.100、原模型/P/c0/F、实际batch16/rect/640缓存几何、FP32数值开关和decoder不变；无基础网络重新推理或重新分组；185621条输出的框/类别/score/顺序保持，bbox官方匹配检测身份不变 |
| 评价 | 原512张dev、T身份、bootstrap、COCOeval版本/忽略政策/maxDets、GT损伤总体及抽样设计不变；bbox/segm匹配仍分别处理 |
| 源码与输入 | 31个冻结源文件与快照及当前文件SHA一致；笔记本23项冻结输入、4个外部评价模块现场SHA核验全通过；Run内缓存前后核验通过 |

唯一历史工程差异已在**启动前协议**明示：独立审计脚本将已有JSON报告的list/tuple比较正规化为排序JSON比较，只有一行，不参与训练或指标计算。其历史/实际SHA已锁定，未借此豁免其它源码变化。

## 证据位置与身份

- [评价SUMMARY](runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0/SUMMARY.json) / [冻结DECISION](runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0/DECISION.json)。SUMMARY SHA：`b482453f9b7a7dbf45ecfc64eec407f12f0696334e58445901d20b5aee225891`。
- [评价配置核验](runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0/CONFIG_AUDIT.json)、[独立复算](runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0/LOCAL_INDEPENDENT_RECHECK.json)、[回传核验](runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0/LOCAL_TRANSFER_VERIFICATION.json)。
- [训练checkpoint审计](runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb/INDEPENDENT_CONFIG_AUDIT.json)与[原配置逐字段比较](runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb/ORIGINAL_CONFIG_COMPARE.json)。最终checkpoint SHA：`15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce`。
- [冻结协议](PROTOCOL.md)、[笔记本现场结案检查](LAPTOP_CLOSEOUT_20261014.json)、[本次回传复核收据](RESULT_VERIFICATION.json)。原生配置SHA：`ae8133e371d121c642e0a32070b3743f6f7657ded587d6ab0491456329198e45`。
- 本次 `B_predictions.json` SHA：`3157da232b5bc7384b8f7f99927b74d40b6f870c6ebadd21ef853f4f1c237de8`，位于笔记本同名评价Run；评价证据包SHA：`fa8e256b9efad45f529369aa8d3390c7728441bfd257631beec9b8065dc8bfd9`。

## 可信范围、推翻条件与唯一下一步

这是固定本地原生配置、COCO开发数据和单种子预算下的初筛；没有新增C对照、全5000图方法AP或独立确认。S0原生账本有效与官方发表AP复现未通过的双状态保持。

只有证实冻结输入、检测身份或评价实现存在错误，才撤销当前科学判定，先纠正该工程问题并用**同一checkpoint**复核；本次核验未发现该类错误，不以追加epoch、换样本或改阈值改判。

**唯一下一步：按事前失败后继，对本次同一checkpoint与现有缓存做同ROI→原图的固定检测配对诊断，重点定位原成功实例损伤发生在proto网格、原图解码还是最终COCO匹配阶段；零训练。** 该诊断尚未启动；当前配置停止在S2，不进入S3，不追加训练。
