# 原生系数头末层共享可达性审计

本实验固定已有 YOLO26m-seg 8.4.100 one-to-one 正样本缓存中的 `h`、prototype 响应 `p`、原系数 `c`、GT 框内标签 `y` 和面积归一化。只优化三种状态：

\[
J_0=J(\Delta c=0),\qquad
J_{shared}^*=\min_{A_0,A_1,A_2}J(\Delta c_i=[h_i;1]A_{level_i}),
\]
\[
J_{ind}^*=\frac1N\sum_i\min_{\Delta c_i}J_i(\Delta c_i).
\]

目标为官方 GT 框内实例 BCE 加 7D 同口径的有限位移正则：

\[
J_i=\frac{\sum_{u\in B_i}\operatorname{BCE}(p_{iu}^{\mathsf T}(c_i+\Delta c_i),y_{iu})}{area_i}
 +\frac{0.003}{2}\|\Delta c_i\|_2^2.
\]

共享增量是原生末层的仿射增量，不新增网络；它在固定特征上具有凸目标。共享参数本身可能因特征秩不足而不唯一，判断依据是目标值、梯度和可实现的输出，不把参数坐标差异当成机制证据。原模型与共享最优的差值表示当前冻结特征下末层仍可回收的部分；共享最优与逐实例 oracle 的差值表示共享仿射约束下剩余的部分。

训练分片只用于求解共享参数，dev/val 只应用冻结后的共享参数作泛化评价。逐实例 oracle 只作为同目标的诊断参照，不进入推理。所有数值结果都是固定缓存上的 mask-BCE 诊断，不等同于 COCO AP 或完整 YOLO 训练效果。
