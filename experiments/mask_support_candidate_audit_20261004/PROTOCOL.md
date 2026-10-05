# Mask-support candidate audit

固定 YOLO26m-seg COCO 预训练权重和 Ultralytics 8.4.100，使用 one-to-one 检测分支的 NMS 前候选。

对每张 COCO val 图片和每个 GT：

- `score_candidate`：同类别候选中置信度最高的候选，作为官方排序代理；
- `best_mask_candidate`：同类别候选中掩码 IoU 最高的候选，仅用于诊断上界；
- `mask_regret = IoU(best_mask_candidate, GT) - IoU(score_candidate, GT)`；
- 同时记录 Box IoU、候选层级、候选是否进入 top-k，以及 GT 面积/拥挤邻居数。

GT 只用于离线测量，不参与模型推理或候选选择。该实验不是 AP 结果，也不声称可部署收益；只有在候选身份与 mask regret 稳定相关时，才进入质量路由方法设计。
