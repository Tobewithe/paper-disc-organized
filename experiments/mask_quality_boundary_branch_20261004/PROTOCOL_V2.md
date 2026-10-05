# 质量排序头收敛与独立复评协议（V2）

制定日期：2026-10-05（Asia/Shanghai）

## 研究问题

在保持 YOLO26m-seg 原检测、分类、框回归、原型和掩码解码能力不变的条件下，独立的 one-to-one 候选质量头能否学习候选掩码 crop-IoU，并改善候选排序？本轮首先检验质量头是否真正收敛，再检验保存 checkpoint 的独立 COCO 评价是否改善。

## 固定资产与运行环境

- 模型：`D:\coco_wire\models\yolo26m-seg.pt`，SHA256 `16B636F04E8FB6A325B3370F22DC5E5535FF473E384F4D041FD28D788F6EE9F5`。
- 固定运行时：`D:\coco_wire\py`，训练子进程实际打印 Ultralytics **8.4.100**；外层记录器探针版本 8.4.27 只作为环境差异保留，不能作为训练版本。
- 训练数据：官方 COCO 转换后的 `fit_full_pending.txt`，7,753 张 train2017 图像；标签由官方 `convert_coco(use_segments=True, cls91to80=True)` 生成。
- 最终评估：独立的 COCO val2017 5,000 图标准标签目录，不用训练期 GT 质量目标。
- 固定模型设置：`mask_ratio=4`、默认 `end2end`/one-to-one 分支、imgsz=640、workers=0、batch=2、seed=0、质量损失权重 0.10。

## 训练设计

- 总训练预算：**6 epoch**。官方默认 warmup 为 3 epoch，因此本轮包含完整 warmup 和 3 个非 warmup epoch；3 epoch 试跑只覆盖 warmup，不再作为收敛依据。
- 只训练质量头的 13,953 个参数；原模型全部参数冻结。
- 原模型 BatchNorm 的 affine 参数和 running buffers 均冻结；每轮 `model.train()` 后由官方冻结层机制重新置为 eval。
- 训练目标：官方 one-to-one assigner 选出的正样本，其当前 crop mask 与匹配 GT 的 IoU；训练阶段允许读取 GT，推理和最终评估不读取 GT。
- 每轮保存 checkpoint，并记录质量头 BCE、每轮目标数、训练损失和完整 stdout/stderr。
- 不使用验证 AP 选择 checkpoint；固定使用最后一轮 checkpoint 作为主要方法结果，同时保留每轮 checkpoint 做收敛曲线。

## 评价顺序

1. 对每轮 checkpoint 在 1,997 张开发子集上做轨迹评估，记录 Mask AP50-95、Mask AP50、Box AP50-95，以及质量分数与实际候选 IoU 的 Spearman 相关和分位排序指标。
2. 固定最后一轮 checkpoint，在完整 COCO val2017 5,000 图上复评；自定义质量头 checkpoint 必须先挂载质量头，再严格恢复 checkpoint state dict，禁止随机初始化附加头。
3. 官方预训练权重在同一 8.4.100、同一 `model.val` 协议下重新评估，作为配对基线。
4. 报告 `alpha=1.0` 主协议结果；`alpha=0.25` 仅作为已完成的强度诊断，不用于替换主结果或反复调参。

## 判定标准

- 收敛证据：训练期质量 BCE 在后续 epoch 不再持续下降，开发集质量排序指标和 AP 不再单调改善；若 6 epoch 仍明显上升，只增加一次预算到 9 epoch，不进行无界延长。
- 方法候选：完整 5k val 的 Mask AP50-95 不低于官方基线，且质量排序指标相对无质量头基线改善；若低于基线超过 0.1 个百分点或排序指标没有改善，停止质量排序主线，不把中性结果写成方法收益。
- Box AP 下降超过 0.1 个百分点时，即使 Mask AP 略升也标为候选筛选副作用，不能称为无损校准。

## 已知限制

本协议只验证候选排序层，不证明 prototype 或 coefficient 是唯一上游根因；总体 AP 的小幅变化不能外推为所有密集实例均改善。此前 3 epoch 未冻结 BatchNorm 的运行已作为实现缺陷保留，不与 V2 结果合并。
