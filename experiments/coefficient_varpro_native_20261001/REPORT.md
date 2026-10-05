# 变量投影式原生系数表示训练：阶段记录

## 当前阶段结论

本 Study 先完成了 VarPro 前置的 `C+readout` 对照：从 Phase I C 组 checkpoint 的原生 one-to-one 系数分支特征出发，只重新求解三个尺度共享的 64→32 原生末层，其他特征、原型、框、类别和官方 TAL 候选身份保持不变。目标沿用官方参考 ROI BCE 加 `lambda=0.003` 的系数输出位移正则；训练/求解只使用 fit 的 6,058 个官方候选。

在修正面积归一化后，正式 Run `RUN_cplus_readout_full_seed0_retry5_area_fix_20261001`（796 张 fit 图、6,058 候选、4 个完整 fit pass、笔记本 RTX 4060 Laptop GPU）得到：

| 项目 | 数值 |
|---|---:|
| 原始模型 A：BCE | 2.5625872548 |
| 原始模型 A：评价目标 | 2.5625872548 |
| C checkpoint（Phase I 结束时） | 2.2881515788 |
| C + 重新适配共享末层：BCE | 2.2730948817 |
| C + 重新适配共享末层：正则 | 0.0128597614 |
| C + 重新适配共享末层：评价目标 | **2.2859546430** |
| 相对 A 的目标变化 | **−0.2766326118** |

这说明 C 组表示后仍有少量末层未充分适配空间（相对 Phase I C 目标再降约 `0.00220`），但它不是 VarPro 训练结果，也没有新图像掩码效用证据。不能把该结果写成突破旧表示限制或方法成功。

## 口径审计

首个全量 Run `RUN_cplus_readout_full_seed0_retry4_20261001` 使用了缓存行中的 `row["area"]` 作为分母；该字段不是官方参考损失要求的预测框像素面积。该 Run 已保留为审计失败记录，不能用于科学结论。修正版每个候选都由 `target_boxes` 按 `((x2-x1)/640)*((y2-y1)/640)*640*640` 重算面积；其原始 A 目标与既有官方报告的 `2.5625872548` 逐值对齐，故当前 C+readout 数值可作同目标前置参照。

100 图修正版 smoke `RUN_cplus_readout_smoke100_area_fix_20261001` 已通过，GPU 为笔记本 RTX 4060 Laptop GPU；本机没有启动训练进程。

## 仍未完成

- 未执行非精确 VarPro 的外层表示更新；
- 未执行同预算普通联合训练对照；
- 未在新图像上评价冻结后的 C+readout 或 VarPro；
- 未作完整 COCO 输出/AP 评价。

因此当前 Study 仍处于前置阶段。只有在实现并运行 VarPro 与等预算 joint 对照后，才能判断“共享读出可达性训练”是否有独立作用。

## 运行入口

- 协议：[PROTOCOL.md](PROTOCOL.md)
- 修正脚本：[scripts/cplus_readout.py](scripts/cplus_readout.py)
- 正式结果：[runs/RUN_cplus_readout_full_seed0_retry5_area_fix_20261001/SUMMARY.json](runs/RUN_cplus_readout_full_seed0_retry5_area_fix_20261001/SUMMARY.json)
- 传输与 smoke：[transfer/](transfer/)

## VarPro 首轮结果（2026-10-01）

在 C+readout 前置通过后，执行了一轮非精确变量投影：固定 backbone、neck、prototype、框和类别分支；只更新原生 one-to-one 系数分支的末层之前卷积；每次外层更新后，再用全 fit 候选重新适配共享 64→32 读出。外层使用 1 个完整 fit pass、学习率 `1e-5`、每 8 张图累积一次；随后内层读出使用与 C+readout 相同的 4 个 fit pass。

| 评价对象 | 候选数 | `J_eval` |
|---|---:|---:|
| 原模型 A（fit） | 6058 | 2.5625872548 |
| Phase I C + 读出适配 | 6058 | 2.2859546430 |
| VarPro 外层更新后、读出再适配（fit） | 6058 | **2.2741069750** |
| VarPro 冻结参数回放（val） | 1346 | 2.8348785961 |
| 原模型 A（val） | 1346 | 2.7712756770 |

VarPro 在 fit 上相对 C+readout 进一步降低 `0.0118476680`，说明外层更新确实改变了原生系数分支表示，并能在同一固定目标上继续拟合。可是冻结同一组参数回放到未参与求解的 val 图像时，`J_eval` 增加 `0.0636029191`。因此本轮是“训练域改善、跨图诊断目标恶化”，不满足协议的泛化放行条件，不进入完整掩码/AP评价，也不宣称 VarPro 方法成立。

100 图 smoke 同样验证了实现链：外层梯度能进入前置系数卷积；其后再适配读出后 `J_eval=2.3132980`，相同 100 图 C+readout 对照为 `2.3265901`。该结果仅作代码可行性验证。

## 失败与审计记录

- `RUN_cplus_readout_full_seed0_retry4_20261001`：使用了错误的 `row["area"]` 分母，已标记为口径审计失败，不用于结论。
- `RUN_varpro_outer_smoke100_20261001`：首次 smoke 使用 inference-mode tensor 直接回传，触发 PyTorch 错误；修正为 clone 普通 tensor 后，`RUN_varpro_outer_smoke100_retry2_20261001` 通过。
- 以上失败 Run 均保留，不覆盖、不删除。

## 本轮边界

当前证据只支持：在固定官方 fit 目标上，VarPro 外层更新具有拟合收益；该收益在 val 的同一诊断目标上没有迁移。还没有测量完整正常推理输出的 Mask IoU、Mask75 或 COCO AP，因此不能把 `J` 的 fit 改善写成分割方法收益。按预注册停止规则，不继续追加外层周期、gate、边界损失或新输入；若重新尝试，必须先解决跨图泛化，而不是单纯增加训练预算。

补充的 dev 回放也同方向恶化：1,402 个官方候选上，原模型 `J=2.5542800`，VarPro 冻结参数 `J=2.6372149`，变化 `+0.0829349`。因此不是只在 val 偶然波动，而是当前一轮表示更新在 fit 之外没有保持诊断目标。
