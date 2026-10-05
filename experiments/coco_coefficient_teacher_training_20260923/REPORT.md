# 阶段 I：原生 one-to-one 系数表示重学习

## 直接结论

阶段 I 的 seed 0 已完成，但没有满足放行条件，因此不进入阶段 II。C 组（普通原生系数分支微调）把固定官方目标从 `2.562587` 降到 `2.288152`，说明重新学习原生系数分支的前置表示确实能回收一部分旧固定表示的损失差距；但它仍高于旧固定表示证书门槛 `2.2529118805`。D 组加入有限 oracle 教师后为 `2.311606`，比 C 高 `0.023454`，预注册的教师额外作用门槛是 `C−D ≥ 0.01`，所以教师方案未通过。

本轮唯一允许的阶段 II 放行条件是：D 在固定评价目标上越过旧证书，并且 D 相对 C 至少改善 0.01。两项均未满足。当前结果支持“原生系数分支内部表示存在可回收空间”，不支持“本轮有限 oracle 教师在同预算下优于普通微调”。不追加更大 head、gate、边界损失或第二个 seed。

## 受控对象与执行

- 模型：COCO 预训练 YOLO26m-seg，Ultralytics 8.4.100；权重 SHA256 为 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。
- 候选：官方 one-to-one TAL，796 张 fit 图、6,058 个候选；身份包含 image、annotation、branch、raw、pyramid level 和 GT index。
- 固定：backbone、neck、prototype、框分支、类别分支、BatchNorm 仿射参数与运行 buffers、候选身份和原始系数锚点。
- B：固定旧的 64 维 `h`，只训练原生 one-to-one 末层，并加入教师项。
- C：重新运行冻结上游特征图，训练原生 one-to-one 系数分支的 Conv2d，只有参考 mask BCE。
- D：与 C 相同，增加固定有限 oracle 的像素 logit Huber 教师项。
- 训练：seed 0、8 epoch、batch 8、FP32、无 AMP；`MuSGD`，`lr0=0.00038`、`lrf=0.88219`、`momentum=0.94751`、`weight_decay=0.00027`。教师为 `beta=0.5`、Huber `tau=1.0`；教师项按每个 GT 框支持内像素平均，再按候选平均。
- 评价：固定原始 `c0` 的 `L_ref + 0.003/2 ||c-c0||²`，教师项不计入 `J_eval`。

远端第一次运行在训练前发现复现路径把一个会原地修改的 crop canvas 重用了；第二次发现缓存图像使用了取整后的 resize 尺寸；第三次完成了输入路径定位。最终 retry4 按缓存记录的有效 resize 几何、TF32 关闭和确定性 cuDNN 路径运行。全体 796 张图的特征/系数回放最大绝对差为 `h=4.86e-4`、`c=6.24e-4`，低于执行版 `2e-3` 数值回放容差；没有在对象不一致时训练。

## 固定目标结果

| 组别 | BCE | 位移正则 | `J_eval` | 相对 A |
|---|---:|---:|---:|---:|
| A 原始模型 | 2.562587 | 0 | **2.562587** | 0 |
| B 固定 h + 教师 | 2.570785 | 0.001119 | **2.571904** | +0.009316 |
| C 普通原生分支微调 | 2.275183 | 0.012968 | **2.288152** | −0.274436 |
| D 原生分支 + oracle 教师 | 2.276666 | 0.034940 | **2.311606** | −0.250981 |

教师项确实产生了独立训练信号：D 的训练日志中教师 Huber 从 `1.652962` 降到 `1.340102`，但它同时使最终 `J_eval` 比 C 高 `0.023454`。B 只训练末层且目标变差，符合其诊断控制定位。

## 预注册判断

```json
{
  "B_crosses_old_certificate": false,
  "C_crosses_old_certificate": false,
  "D_crosses_old_certificate": false,
  "teacher_delta_C_minus_D": -0.0234543796,
  "teacher_material_threshold": 0.01,
  "teacher_pass": false,
  "phase_II_pass": false
}
```

这里的“未越过证书”是相对固定 fit 候选和旧证书目标的判断，不是新图泛化结论。C 的正结果说明前置系数表示比固定旧 `h` 更有回收能力；D 的负结果说明在当前 β、Huber、优化器和 8 轮预算下，教师项没有给普通微调增加可测的目标收益，不能据此断言任何 oracle 教师都无效。

完整运行、逐候选结果和远端回传清单见：

- [阶段 I Run](runs/RUN_phaseI_native_representation_seed0_retry4_20261001/)
- [运行锁定配置](PHASE_I_RUN_CONFIG.json)
- [阶段 I 协议](PHASE_I_PROTOCOL_20261001.md)

