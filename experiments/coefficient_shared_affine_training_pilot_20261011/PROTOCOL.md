# 官方 TAL 冻结特征共享系数末层训练 pilot

## 唯一问题

在已经固定的官方 one-to-one TAL 候选、特征和 prototype 上，真实的 mini-batch 参数更新能否找到共享仿射诊断已经发现的 mask 改善规律？本 pilot 只判断“优化路径是否能到达同一个共享目标”，不判断完整 YOLO 训练能力上限。

## 固定对象

输入是 `coefficient_official_tal_affine_20260930` 产生的官方缓存 `INDEX.json` 与 `images/*.pt`。每个候选固定：`image_id`、`annotation_id`、`branch`、`raw_id`、`pyramid_level`、`target_gt_idx`、`h`、`coeff`、完整 prototype、`target_boxes`、官方 overlap raster、`owners` 和 assignment。训练期间不重跑 TAL、不重新选候选、不改变 boxes/class scores、不解冻 backbone、neck、prototype、box/class branch。

小规模默认取官方清单中的 64 张 fit 图、32 张 dev 图、32 张 val 图，按固定 seed 分别抽样；可用命令行参数改变数量，但不得把 dev/val 用于拟合或超参数选择。该规模只用于 pilot，不能推广为方法能力上限。

## 训练目标

先用 fit 子集每个金字塔层计算 `mu_l`、`std_l`。训练参数是原生 one-to-one 末层的**增量** `A_l∈R^(65×32)`，从零初始化：

\[
  \Delta c_i = [(h_i-\mu_l)/\mathrm{std}_l;1]A_l,
  \qquad c_i=c_{0,i}+\Delta c_i.
\]

每个候选的目标严格复用官方 TAL 分割项：

\[
  F_i=9.83241\frac{\sum_{u\in B_i}\mathrm{BCEWithLogits}(p_{iu}^{\mathsf T}c_i,y_{iu})}{area_i}
      +\frac{0.003}{2}\|\Delta c_i\|_2^2.
\]

候选等权平均。`9.83241` 是冻结权重的官方分割增益；正则只作用在输出增量 `Δc`，不使用 optimizer weight decay。这样训练目标与 `coefficient_official_tal_affine_20260930` 的 B 解完全同式。

## 对照和优化器

脚本先在同一 fit 子集上用 FP64 full-batch L-BFGS 求出 `A_ref`，作为该子集共享仿射目标的数值参考；它不参与 dev/val 拟合。训练臂从零初始化同样的 `A_l`，支持：

- `adam`：默认 `lr=1e-2`、无 weight decay；
- `sgd`：默认 `lr=1e-2`、momentum=0.9、无 weight decay。

两臂使用相同图片顺序、batch_images、epoch 数和 fit 子集。默认 10 epoch、4 图一批、seed=20261011。每个 epoch 后只记录 fit 轨迹；最终参数固定为最后一个 epoch，不根据 dev/val 选 checkpoint。

## 评价和预注册门槛

对 native (`A=0`)、trained 和 reference 在 fit/dev/val 上计算相同的 `J`、BCE、正则，并保存候选级和 image-macro 统计。image-macro 的区间用整图 bootstrap，seed 固定为 20261011，默认 2000 次。

训练臂被判定为“找到共享规律”必须同时满足：

1. fit 目标回收参考改善的至少 90%：
   `J_train_fit <= J0_fit - 0.90*(J0_fit-J_ref_fit)`；
2. dev 与 val 的 image-macro `J_train-J0` 的 bootstrap 95% 区间上界都不大于 0。

若第 1 条失败，判定为当前优化器/训练路径没有找到已有共享规律；若第 1 条通过而第 2 条失败，判定为训练已找到 fit 共享规律但该规律在小规模 held-out 上未迁移。两种情况都只对本 pilot 的固定缓存和配置负责。

## 产物和 Run 记录

运行目录由 `--out` 指定，脚本写入：

- `run.json`：命令、配置、输入路径、选择的 image ids、环境版本和状态；
- `CONFIG.json`：冻结项与损失定义；
- `REFERENCE.pt`、`TRAINED_<optimizer>.pt`：参数和 fit 统计；
- `TRACE_<optimizer>.jsonl`：逐 epoch/step 目标、梯度和学习率；
- `SUMMARY.json`：三组统计、bootstrap 区间和唯一 gate 判定；
- `COMPLETE.json`：完成状态。

脚本只读取缓存，不修改既有实验产物。正式运行应在笔记本或指定 GPU 主机的新 Run 目录中执行；不要覆盖 `coefficient_official_tal_affine_20260930` 的任何文件。
