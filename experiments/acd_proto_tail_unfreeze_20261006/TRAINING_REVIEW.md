# ACD 原型尾部解冻正式训练审核

训练完整性审核通过。此文件是原始审计的派生复核，不替代 Run 证据，也不判断最终 AP。

本次仅读取本地两个正式 Run；未修改原始 Run 文件，未连接远端或启动执行。

| 项目 | Baseline | ACD |
|---|---:|---:|
| 总可训练参数 | 2,306,240 | 2,306,240 |
| one2many 系数参数（变化 tensor） | 854,112（12/12） | 854,112（12/12） |
| one2one 系数参数（变化 tensor） | 854,112（12/12） | 854,112（12/12） |
| proto tail 参数（变化 tensor） | 598,016（2/2） | 598,016（2/2） |
| 训练 epochs / seed / batch | 3 / 0 / 2 | 3 / 0 / 2 |
| fit / 内部 validation 图像 | 796 / 196 | 796 / 196 |
| 处理 batch | 1,194 | 1,194 |
| optimizer 尝试 / 实际应用 / AMP 跳过 | 69 / 66 / 3 | 69 / 66 / 3 |
| 各 epoch 实际应用 | 41 / 13 / 12 | 41 / 13 / 12 |
| 冻结 state tensor | 878，live/EMA 保持原值 | 878，live/EMA 保持原值 |
| 原型 cv2 / cv3 真实变化 | 两块均变化 | 两块均变化 |

全部 26 个可训练参数 tensor 均真实变化。两个新增原型 Conv 各有一个可训练 weight tensor；所有 BN affine 与 buffers 冻结。

Baseline 的 one2many / one2one / proto_tail 最大参数绝对变化为 0.04886597394943237 / 0.017944544553756714 / 0.00935404747724533；ACD 为 0.057471632957458496 / 0.019830822944641113 / 0.008591696619987488。最大值属于整个组，不代表每一参数或每一原型块的变化量。

878 个冻结 state 的逐 tensor hash map 在 initial、final live、final EMA 间完全相等，且跨 arm 相同。冻结集合 digest：`72244ee9b2d7de8884e4832271ef7f73e747053c60e4f6878d0b25c150489d8b`。

两组输入的 1,194 行经过换行规范化后均匹配原始 audit digest，且彼此相同：`3bc535d443234cfcc45d057ebbde7f6aac090d29c0ba34d76cbd07c2d324380b`。源训练脚本、源协议和 optimizer 逐次调度及应用结果也相同。

| 同位 AMP 跳过 | epoch | attempt | 已处理 batch | scale 变化 |
|---|---:|---:|---:|---|
| 1 | 1 | 4 | 4 | 65,536 → 32,768 |
| 2 | 1 | 9 | 11 | 32,768 → 16,384 |
| 3 | 1 | 26 | 77 | 16,384 → 8,192 |

三次非有限 raw/unscaled 梯度都只出现在 `model.23.proto.cv3.conv.weight`。GradScaler 均没有调用 optimizer，所有非有限尝试都得到确认的 AMP skip；无非有限梯度实际应用，无未归类 skip，无异常。官方 EMA 仍在每次尝试后更新，总计 69 次。三个训练组各有 66 次非零、有限的实际应用梯度。

ACD 接受动作 one2many 4,663 次、one2one 4,136 次，共 **8,799 次**；惩罚 pixel terms 为 2,177,078 + 3,235,225 = **5,412,303**。这些是重复训练候选与像素项，不能当成独立实例或推理修复数量。

保留官方 MuSGD 命名分组规则：`proto.cv2.conv.weight` 的 base lr 为 0.00038，`proto.cv3.conv.weight` 为 0.00114（3×），源于官方 `(?=.*23)(?=.*cv3)` regex。每次实际 lr 仍随 warmup/调度变化，并已记录；两组完全相同。

最后一次 optimizer 尝试为 attempt 69，发生在第 1,177 个输入 batch 后。随后仍处理 17 个 batch，但完整日志没有更晚 step。因此这 17 个 batch 没有形成额外参数更新；此行为来自保留的官方累积方式，双方相同。

单 seed、796 张 fit 图及 66 次实际更新只支持短预算可行性筛查。正式 gradient audit 衡量总损失，尚不能拆分 ACD 与原 mask loss 的梯度强度或方向。官方 one2one proto detach 保留，新增原型尾部从 one2many 路径更新。

每组 `trainer/results.csv` 均有三轮内部验证记录。未生成 `epoch_metrics.jsonl`。196 图内部验证和使用 FP16 last.pt 的 trainer final metrics 不替代 5,000 图 FP32 overlay 主评估；本审核不包含最终效果判断。

详细派生数据、全部源文件绝对路径和 SHA256 见 [TRAINING_REVIEW.json](TRAINING_REVIEW.json)。

| 源文件 | SHA256 |
|---|---|
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\TRAINING_SETUP.json` | `5fa9191c5d3086e555ffbd4b821b1824fc3ee52bc2601751b8f6fe595653fabf` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\TRAINING_AUDIT.json` | `b47b8bb99dd62b370111c04960d79bbff2da5fd17d1ae8e641e112b49de55c7c` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\TRAINING_COMPLETE.json` | `56bfa9d52681e9ccf18a7d6772ca4df711edb18cac38ed1b2df42aef90d6b70a` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\optimizer_step_audit.jsonl` | `be34beb92f179789013b729c18445c7b697c2bc4cef7ca9322c9866f148af191` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\input_batch_hashes.jsonl` | `7bdc7c8c02b998399e6d7835a9fa5612a494737d6794e0ed310a8add20d6b9f5` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\trainer\results.csv` | `754a32f61a64f87deb944aed45724973dd73398584dfc16259a6806cade49243` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\trainer_final_epoch_metrics.json` | `1f52d57bc82b0d5fe70cf6d53e2ff203da9acbd0d9faec94e0b23727efe4142a` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\SOURCE.json` | `e13665f5b96b968f092ca707ffa8728b515c90bd096aa71e3132945170e8e654` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\source\train_acd.py` | `d5868a1d2dc3e57dded535561aec96554fab42f9b0b1ea71c1d863d3e009e763` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0\source\PROTOCOL.md` | `42059829e3e49175a093bb03d2df07e87ac2eb8adc398395ac0fbb22f351be0d` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\TRAINING_SETUP.json` | `474217a7d8ad62eec133db37987c5b88f04827098e1cec2a155793dea43642e3` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\TRAINING_AUDIT.json` | `8aa0702c6444e305ee7d6ad04b2a5489b4f3849948de407720e781794d576745` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\TRAINING_COMPLETE.json` | `4bd3a24d05f038f93142b666cf9644b253ab53a3593ea4e3cd638222fc82a2c5` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\optimizer_step_audit.jsonl` | `d05f353b34421fa89422b1fc365685be073ce99b747cfa84572bbe38fa62d056` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\input_batch_hashes.jsonl` | `7bdc7c8c02b998399e6d7835a9fa5612a494737d6794e0ed310a8add20d6b9f5` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\trainer\results.csv` | `005d116997f4045d2e308f9d2067383fab7ab38ac0fc4c0c6d2a18c7dd60f275` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\trainer_final_epoch_metrics.json` | `a36d096625403a7999d44a2c9c423dc1d07da7bbfdd83598f039969e2937a98a` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\SOURCE.json` | `dbb8ebcd573163ffbbfde5118d7a5eb24f7dddd59b85ddcd004905762759f074` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\source\train_acd.py` | `d5868a1d2dc3e57dded535561aec96554fab42f9b0b1ea71c1d863d3e009e763` |
| `C:\Dpan\codexproject\paper-disc-organized\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\source\PROTOCOL.md` | `42059829e3e49175a093bb03d2df07e87ac2eb8adc398395ac0fbb22f351be0d` |
