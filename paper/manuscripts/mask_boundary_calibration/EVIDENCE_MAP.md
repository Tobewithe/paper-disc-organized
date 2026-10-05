# 稿件证据对应表

本稿引用独立实验 `STUDY_8fb3468ebb704682a2225ebed0e16206`。原始实验及 Run 不因论文写作而改名或复制登记。机器可读来源与证据版本见 [EVIDENCE_SOURCES.json](EVIDENCE_SOURCES.json)；小型复现副本及源码出处见 [COPY_MANIFEST.json](reproducibility/COPY_MANIFEST.json)。

## 核心主张及范围

<!-- research-note: {"id":"CLAIM_rcmc_selective_readout_20260916","kind":"claim","title":"试修正响应可用于选择掩码读出校准，在所测两种尺度上改善修复与误伤权衡","coverage":"declared","verdict":"supports","scope":{"target":"COCO-pretrained YOLO26m-seg and frozen transfer to YOLO26s-seg","metric":"COCO segmentation AP; fixed-slot R75 repairs and damages","dataset":"4,500 val2017 images; 32,831 ordinary GT; train2017 1,500/500 calibration split","protocol":"Ultralytics 8.4.100, one-to-one, imgsz640, conf.001, max_det300, FP32, fixed masks export"},"limitations":["Validation predictions had informed earlier exploration; not blind test-dev.","One fit and one architecture family; no AP confidence interval.","Gate adds latency and retains some harmful corrections.","Correctable readout does not establish a unique upstream network cause."],"evidence":[{"path":"experiments/mask_boundary_route_20260914/runs/RUN_e0e0d46defb74e3d9a90094f49ce26fe/SUMMARY.json","revision":"99c25bf7d41e5b37feebe64b2dff9133b77c82247e5b1aeabecccb5788736bd6","relation":"derived_from"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_f9df48632ac742248343184263262429/SUMMARY.json","revision":"ee81c06e8d76b36a34bbe113647845266ae4845b33ad8b7d4319d4a8e425f39a","relation":"derived_from"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_f0836430494141479d6a36da31db7df4/SUMMARY.json","revision":"a3fc7cd35d19cad0946941c30bb011fa3da6954f540f3809f31ad5ca9027b0cc","relation":"derived_from"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_44ab365e6ae64cb9a638557d60126b01/SUMMARY.json","revision":"e8477f7fe4131020e2c9eeb716e99ce2a86bd57fb9bf1b4be806590fcff7e4ec","relation":"derived_from"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_e7df55bcc56541ca8c449e5065c885a4/SUMMARY.json","revision":"8b6713bebec9d476b1a0e845625defb03ef1b7cae3b0fab7946a6768e7e880d4","relation":"derived_from"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_d01ffdef9f3c498fa3529d96cea875eb/SUMMARY.json","revision":"7767385e94e5c5178c5a4c0331d78d8337499cf9813d865a4739f852e5e6b315","relation":"derived_from"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_69cb2c16410147d6b6c683d751383f7c/SUMMARY.json","revision":"86bec9691f07642df16d1efb5e63cb5f907f0b80bed9eb784ff4d8f88bce99e3","relation":"derived_from"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_b0591a9bbbca4663995576db1448e1bd/SUMMARY.json","revision":"6ac5dddcdfee540cb68a92e31f482baad0fa30d710014796eeb8e6e6681c54e2","relation":"derived_from"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_8ecb45b277cc42c5a0d37217e010f5ab/SUMMARY.json","revision":"2d9b6867beb0aa124654c3db454a8d5ce271b13e6580ae4551d52efc81a879ec","relation":"derived_from"}]} -->
在固定 logits、预测框、类别、分数及候选身份的条件下，读取试收紧操作引起的变化可以帮助预测操作收益。门控在所测两种预训练尺度上改善正常掩码 AP，并减少固定平滑对原本成功实例的误伤。这支持“读出可修正且需选择性应用”的机制叙事，不足以证明原型或系数是唯一根因。
<!-- /research-note -->

## 正文主张与数字

| 正文位置／主张 | 证据与适用边界 |
|---|---|
| 摘要、Table 1：m AP 43.38 → 43.94；s AP 39.42 → 39.90 | m、s 最终冻结结果，4,500 图。不是 COCO 全量 5,000 图成绩，也不是官方公开排行榜复现 |
| Fig. 2、Table 2：m 误伤 576 → 445，s 556 → 389 | 固定 box 匹配下跨越 Mask IoU 0.75 的实例数，分别相对降低 22.74%、30.04%；同时修复数分别减少 106、112 |
| 门控相对固定平滑额外约 +0.11／+0.10 AP | AP 点估计；没有 AP 显著性检验。m 的 R75 对固定平滑差值 CI 包含零，s 的不包含零 |
| Table 3：m 响应门控拒绝组真实平均 ΔIoU = −0.0250 | 2,847 个固定匹配实例；仍有 39.66% 的拒绝实例本可受益，不能解释成完美识别 |
| Table 4：高覆盖失败组修复 26.87%／25.19% | 组条件为同类 Box IoU ≥ .75、实际 baseline mask coverage ≥ .95、mask IoU < .75；m=1,701、s=1,596。不是全体失败的修复率 |
| Mask 修改与过滤消融 | m filtering-only +0.000901 AP 点，boundary-only +0.448457；仅支持这个固定操作的收益主要来自掩码修改 |
| Table 5：响应门控 57.37 ms，固定平滑 35.60 ms | 4060 Laptop、80 图、4 轮、batch1 FP32；包括特征提取与 CPU 门控。不是零开销或 TensorRT 速度 |
| 补充实例图 | 开发集 500 图的固定平滑样例；明确包含误伤例，不将其冒充最终响应门控输出 |

所有 COCO AP/AR 表采用 0–100 量纲；连续 IoU 仍用 0–1，另有显式 ×100 的表按表注读取。固定匹配 R75 分母包含所有 32,831 个 ordinary GT，并非只算匹配成功的目标；它不是官方 COCO AR。

## 已完成运行

| Run／摘要原文 | 用途 | SHA256 前 12 位 |
|---|---|---|
| [RUN_e0e0d46defb74e3d9a90094f49ce26fe](../../../experiments/mask_boundary_route_20260914/runs/RUN_e0e0d46defb74e3d9a90094f49ce26fe/SUMMARY.json) | m 模型正常框 AP、固定匹配转移、配对图像 bootstrap | `99c25bf7d41e` |
| [RUN_f9df48632ac742248343184263262429](../../../experiments/mask_boundary_route_20260914/runs/RUN_f9df48632ac742248343184263262429/SUMMARY.json) | s 模型冻结迁移及相同评价 | `ee81c06e8d76` |
| [RUN_f0836430494141479d6a36da31db7df4](../../../experiments/mask_boundary_route_20260914/runs/RUN_f0836430494141479d6a36da31db7df4/SUMMARY.json) | train2017 选定的全局阈值 0.25 在 m 验证集上的对照 | `a3fc7cd35d19` |
| [RUN_44ab365e6ae64cb9a638557d60126b01](../../../experiments/mask_boundary_route_20260914/runs/RUN_44ab365e6ae64cb9a638557d60126b01/SUMMARY.json) | 定位好且高覆盖的失败组、纯度与覆盖率 | `e8477f7fe413` |
| [RUN_e7df55bcc56541ca8c449e5065c885a4](../../../experiments/mask_boundary_route_20260914/runs/RUN_e7df55bcc56541ca8c449e5065c885a4/SUMMARY.json) | 4060 Laptop 实际推理成本 | `8b6713bebec9` |
| [RUN_d01ffdef9f3c498fa3529d96cea875eb](../../../experiments/mask_boundary_route_20260914/runs/RUN_d01ffdef9f3c498fa3529d96cea875eb/SUMMARY.json) | 训练集门控拟合、选择集及冻结参数 | `7767385e94e5` |
| [RUN_69cb2c16410147d6b6c683d751383f7c](../../../experiments/mask_boundary_route_20260914/runs/RUN_69cb2c16410147d6b6c683d751383f7c/SUMMARY.json) | mask 修改和空候选过滤的因子分解 | `86bec9691f07` |
| [RUN_b0591a9bbbca4663995576db1448e1bd](../../../experiments/mask_boundary_route_20260914/runs/RUN_b0591a9bbbca4663995576db1448e1bd/SUMMARY.json) | 接受／拒绝组的真实 IoU 变化 | `6ac5dddcdfee` |
| [RUN_8ecb45b277cc42c5a0d37217e010f5ab](../../../experiments/mask_boundary_route_20260914/runs/RUN_8ecb45b277cc42c5a0d37217e010f5ab/SUMMARY.json) | 开发集上的固定平滑实例图；非最终门控可视化 | `2d9b6867beb0` |

## 模板、论文与源码

学术参考文献位于 [references.bib](references.bib)，正文明确区分 YOLACT、Mask Scoring R-CNN、PointRend、BPR 和既有自适应阈值方法。创新定位是“预测具体修正的收益并利用试修正响应选择操作”，不宣称阈值调整本身是首次提出。

投稿格式与分区来源见 [VENUE_AND_FORMAT.md](VENUE_AND_FORMAT.md)。全文、补充材料和图表均可编辑；PDF 通过编译与版面检查。作者信息和最终声明没有代填，本交付不执行投稿。

## 2026-09-20：与当前失败分类规范的对应

<!-- research-note: {"id":"REVIEW_5c6f0c04a3e54916bf12f6f609223c82","kind":"review","title":"新版候选集合分类与 RCMC 已有失败组结果的适用边界","coverage":"declared","scope":{"target":"RCMC m/s frozen exported-candidate evaluation versus current raw candidate-set taxonomy","metric":"conditional repair at Mask IoU .75; mean IoU; global COCO segmentation AP","dataset":"4500 val2017 images,32831 ordinary GT","protocol":"existing fixed same-class score-ordered Box50 association; not new score-independent raw classification"},"evidence":[{"path":"project/guidelines/EXPERIMENT_SPEC_STANDARD.md","revision":"20f68d0ce51dafc2d39a85709ea128c8296b76e42338a483ac62653428685a78","relation":"references"},{"path":"experiments/mask_boundary_route_20260914/runs/RUN_44ab365e6ae64cb9a638557d60126b01/SUMMARY.json","revision":"e8477f7fe4131020e2c9eeb716e99ce2a86bd57fb9bf1b4be806590fcff7e4ec","relation":"derived_from"}],"limitations":["No per-class AP under the new five-state raw taxonomy has been computed for RCMC.","Original failure group is conditioned on baseline output and must not be called a candidate-pool geometric failure.","AP is not additive over GT failure groups; repair fraction is not AP gain."]} -->
当前规范第 6.2.1 节采用五种 GT 级完整 raw 候选集合状态，并将几何能力与分数／类别／保留过程解耦。这有助于判断读出修正发生在“候选 mask 质量不足”还是“已有好候选但输出未使用”等不同层面。RCMC 不修改框、类别、分数或上游候选筛选；可能修正进入解码的候选 mask，却不能凭本实验解释或修复所有 raw 候选、框或筛选失败。良好框 IoU 不等于充分像素支持，需单独检查实际解码支持上界。

本稿的高覆盖失败组依据最终导出候选的同类、分数顺序、固定 Box50 关联，再筛选 Box75、实际 mask coverage≥.95、Mask IoU<.75。它是输出层的同候选诊断；不能由某条输出 mask 失败推断整个候选池都不存在 Mask75。当前没有按新规范重算本稿完整 raw 五类，也没有这些五类内的 AP。

既有组内改善：m 的 1,701 个高覆盖失败修复 457 个（26.87%），平均 IoU 65.40→68.72（+3.32 点）；s 的 1,596 个修复 402 个（25.19%），平均 IoU 65.19→68.35（+3.16 点）。这是相对基线的失败修复，且基线按定义在该组的固定匹配 R75 为零，不能称为“AP 提升 26.87 点”。组内同时包含覆盖率下降，完整数字沿用上述 Run，不以纯度提升代替无损修复。

扩大分母后，全部已匹配失败的修复率分别为 801/11,361=7.05%、729/12,050=6.05%；连同固定匹配中未关联 GT，则为 801/(11,361+2,405)=5.82%、729/(12,050+3,351)=4.73%。另有原本成功实例 445／389 个被误伤，净增达标实例 356／340，换算全体 GT 的固定匹配 R75 只增加 1.084／1.036 点。这说明总体增益小不能仅解释为容易实例稀释，还包含失败可作用范围有限和修复／误伤抵消。

若后续要量化某类失败对总体 AP 的贡献，可在预先冻结的基线分组上只替换该组对应候选的 mask，保留其余预测和所有 GT，再跑完整 COCOeval，报告“该组替换的条件 ΔAP”。此为使用 GT 分组的诊断，不能冒充可部署门控或可相加的 AP 分解；分组配对也必须注明来自 raw 几何还是最终输出。若做类内 AP，须另外定义非目标 GT、相关预测、重复和背景 FP 的处理，不能简单删 GT。当前未执行上述额外 AP 计算。
<!-- /research-note -->

