# 共享原型的候选重复计权审计

**Goal**：固定官方权重与 one-to-many / one-to-one assignment，判断 many 分支的“候选等权”相对“GT 等权”是否改变共享原型参数的局部更新方向，以及这种方向差异是否影响同图固定 one 候选的正常原图掩码。本轮是 **GT 辅助、逐图重置的一步诊断**，不是训练、独立泛化验证或可部署方法。

## 依据、去重与竞争解释

官方 8.4.100 的 [head.py](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.100/ultralytics/nn/modules/head.py) 将 one 分支的 prototype detach，many 分支承担原型 mask 梯度；[loss.py](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.100/ultralytics/utils/loss.py) 将官方逐候选 mask loss 累加后除以正候选数。因此，同图不同 GT 的正候选数可能改变它们对共享原型的相对权重。本轮实际导入代码、权重和 assignment 配置仍须锁定。

已有 [系数头梯度诊断](../coefficient_gradient_conflict_20260924/PROTOCOL.md) 研究的是系数预测参数；[原型联合微调](../native_proto_coadaptation_screen_20261003/REPORT.md) 只使用总梯度；[多位置监督](../mask_supervision_support_20261004/REPORT.md) 改变的是系数分支的监督位置。它们均未完成本轮“相同 many 候选、不同 GT 聚合、共享原型参数方向”的对照。

[Libra R-CNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.html) 提供样本平衡的先例；[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) 提醒梯度幅度本身会改变训练；[CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) 区分平均目标与各任务局部改善。这里不实现这些算法，也不将简单 GT 平均称为原创。候选数可能携带样本难度/质量信息，等权未必更好；负梯度夹角本身不证明性能根因。

## 对象与四臂

沿用已冻结 fit 清单的**前 128 张计划图**；smoke 使用其中前 2 张有效图。保存计划/有效图片、无 one / 无 many 正样本及原因，不补图。原始 COCO GT、官方 overlap 标签、真实 letterbox 和完整 prototype 沿用已核验路径；assignment 由原模型一次生成并冻结，不经置信度筛选、top-k 或 NMS。永久身份保存 image/annotation/branch/raw/level/target_gt_idx。

令图内 many 正样本集合为 \(S=\cup_tS_t\)，\(n_t=|S_t|\)，\(T=|\{t:n_t>0\}|\)。\(\ell_j\) 使用该 many raw 的**原始 many 系数**与官方标签、GT 支持、面积归一化及 mask gain；不替换为 one 系数。所有尺度合并后按 GT 分组。

\[
L_M={1\over|S|}\sum_{j\in S}\ell_j,
\qquad L_E={1\over T}\sum_t{1\over n_t}\sum_{j\in S_t}\ell_j.
\]

\(L_O\) 是原始 one 正样本及其原始 one 系数的同口径候选平均。只对共享原型模块的参数 \(\theta_P\) 求 \(g_X=\nabla_{\theta_P}L_X\)，不用对输出 P 的梯度代替参数梯度。

| 臂 | 同图操作 |
|---|---|
| A | 原参数 \(\theta_0\) |
| M | \(\theta_0-\eta g_M/\|g_M\|_2\) |
| E | \(\theta_0-\eta g_E/\|g_E\|_2\) |
| O | \(\theta_0-\eta g_O/\|g_O\|_2\)，仅诊断参照，不称 oracle |

统一 \(\eta=10^{-4}\|\theta_0\|_F\)，范数针对同一原型参数白名单展平计算。每臂从同一 \(\theta_0\) 出发，**每图恢复**，无优化器状态、无跨图累计。原系数、前缀特征、检测输出、BN 参数与 buffers 均冻结；仅临时原型参数改变，随后重生成完整 P。无新网络、loss、教师、阈值或 gate。

## 必须通过的检查与机制激活

FP32、TF32 关闭。smoke 验证 M 的 loss/梯度与本地官方实现一致，预设绝对/相对容差均为 \(3\times10^{-5}\)；A 的原输出复现沿用该容差。权重恢复、冻结参数和 buffers 要求 bit-exact；任何异常保留日志并停止，不调宽容差。参数名、实际/目标步长、梯度范数和退化状态逐图保存。零梯度不得除零，不删掉该图掩码评价。

保存每 GT 候选数、GT 总权重和 M/E 方向距离。若各 \(n_t\) 相同，则 M=E 是必然结果，不能作为“平衡无效”的证据。GT 有无 many 覆盖完整列出，one 评价集合不因此变化。同范数排除纯全局梯度倍率解释，但不排除重计权改变了 GT 难度/质量构成。

逐臂核对 one loss 的实际有限差 \(L_O(\theta_X)-L_O(\theta_0)\) 与一阶预测 \(g_O^\mathsf T(\theta_X-\theta_0)\)。保存残差，使用“绝对残差 \(\le10^{-6}\) 或相对残差 \(\le10\%\)”的固定局部忠实度判据；分母为 \(\max(|\text{预测差}|,10^{-7})\)。超界时保留有限步的描述结果，但不据夹角解释收益，不自动换步长。机制未激活、数值失真或身份/恢复失败，记为未充分检验，不能包装成机制阴性。

整轮局部解释的预设条件：至少 **90% 的有 one 正样本图片**，M/E/O 三臂对 one loss 的有限差分同时通过上述忠实度判据。分母包括所有有 one 正样本图片，不因失真删除图片再计算方向或掩码收益；零梯度及未定义方向另列。90% 是本次筛选的可解释性标准，不是理论保证。未过时保留全体有限步指标，但将“方向与效用联合解释”记为未充分检验，不宣称重计权无效、不扫 eta。

## 评价、预算与停止

机制主量为逐图 \(\cos(g_E,g_O)-\cos(g_M,g_O)\)；报告 M/E 原范数及所有未定义数。正常原图评价固定原 one 候选、预测框与原 COCO mask，沿用完整 P 解码；不使用求解支持替代完整掩码。报告 A/M/E/O 的图片 macro 与候选均值 IoU、Mask75 修复/损伤、coverage、连续 logit AUC、固定支持 FPR；全体、原成功/失败、框好掩码差及尺度分层均由 A 固定，空 mask 保留。

固定 seed 20261004，1,000 次整图配对 bootstrap；候选统计也整图重采样。主要效用比较 E−M，同时列 E−A/M−A/O−A；子组只作规定对象解释，不能冒充正常输出或 AP。

**值得另行规划正式训练的短 Gate**：通过上述整轮局部忠实度条件；方向差图均值 \(\ge0.05\) 且 95% CI 下界 \(>0\)；并且 E−M 满足“全体 macro IoU \(\ge0.001\)”或“框好掩码差 macro IoU \(\ge0.003\) 且全体 \(\ge-0.0005\)”。IoU 门槛使用点估计，CI 完整报告，不追加 CI 下界要求。同时披露成功实例损伤与各指标代价。通过只说明局部 GT 辅助诊断值得继续；不证明跨图训练、历史归因或部署收益。未通过结束当前配置；不扫步长、不追加图片、不自动训练 PCGrad 或重加权方法。

服务器运行；桌面不运行模型。smoke 240 秒，正式 1,200 秒硬预算，超时/异常为未完成，保留独立 Run 和日志。输出冻结输入/代码 hash、身份与梯度审计、逐图/逐候选指标、SUMMARY、REPORT、完成状态；回传附 SHA256 manifest。结案说明观察支持哪一项局部假设、哪些仍未知。
