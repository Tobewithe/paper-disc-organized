# YOLO26m-seg v8.4.100：O2O raw–类别 Top-K 与 COCO 官方基线核验

日期：2026-10-13。状态：**源码与公布数值及官方权重 SHA 已核实；官方原生 model.val 数值复现尚未执行**。这是审计补充，不修改历史 Run、原始预测、原始评估数值或历史论文表格。

## A. O2O 两轮 Top-K + conf 的实际对象（已核实）

固定源码（仓库有 vendor 快照）：
- `shared/vendor/ultralytics_8_4_100/ultralytics/nn/modules/head.py`：`Detect.forward`、`Detect.get_topk_index`、`Segment.postprocess`；
- `shared/vendor/ultralytics_8_4_100/ultralytics/utils/nms.py`：`non_max_suppression` 中 `end2end` 的早返回；
- `shared/vendor/ultralytics_8_4_100/ultralytics/models/yolo/detect/val.py`、`models/yolo/segment/val.py`：输出验证和 Mask 解码。

条件：**`head.end2end=True`，`head.agnostic_nms=False`（Head 默认值），`max_det=300`，从 one-to-one head 推理。每个 raw 只产生一套原生框和一组 Mask 系数、80 个类别分数；最终不是先把每 raw 截断为最高分类。**

准确执行次序（输入形状随真实特征网格改变，8400 只适用于 640×640 等对应网格）：

1. `scores.sigmoid()`：每 raw 的 80 类分数。
2. 第一次 Top-K：`scores.max(dim=-1)[0].topk(K)`，按照每 raw **最高类别分数**选择最多 300 个 raw 位置。
3. 第二次 Top-K：对入选 raw 保留全部 80 类 `scores.gather(...).flatten(1).topk(K)`，最终选最多 **300 条 (raw_id, class_id, class_score)** 记录。一个 raw 可出现多次、不同类别共享其相同预测框和相同 Mask 系数；这不是 300 个互异物体。
4. `Segment.postprocess` 对每条入选组合根据 **可重复** `idx` 读取框和 Mask 系数。
5. `nms.non_max_suppression(..., end2end=True)` 直接执行 `pred[:,4] > conf_thres`（严格大于）并截断最多 `max_det`，**筛选对象是已经展开的 (raw,类别) 检测行，不是原始 raw**；此路径不经过普通 NMS，也不经过普通 NMS 的 `multi_label` 展开。
6. 分割验证器随后用这批保留行的系数、框和共享 Prototype 解码 Mask，并分别提交 bbox/segm 给官方 COCOeval。

**Head 中另一分支：** 如果明确将 **`head.agnostic_nms=True`**，源码在两轮展开之前改为每 raw 取最高类别，只做一次 `topk(K)`。这个属性和验证器传给后处理 NMS 的 `agnostic` 参数不应自动视为同一操作；需要确认实际 head 的属性、分支及调用。普通 NMS 的 `multi_label=False` 也不能取消已经在 Head 产生的重复 raw–类别输出。

**规范要求：** 评价或 GT 分类的日志至少记录 `(image_id, branch, raw_id_global, category_id, class_score, output_detection_id)`，其中 `output_detection_id` 对应最终筛选后的检测行。`raw_id` 是原始几何候选身份，不能当作唯一最终检测 ID；若 COCO JSON 不含 raw_id，必须保留推理阶段的索引映射，不能从输出编号反推。Box/Mask 都由同一输出行对应 raw 的框与 Mask 构成；bbox COCOeval 与 segm COCOeval 仍独立匹配 GT。

历史证据：`experiments/coco_raw_capability_5000_20260916/REPORT.md`：5000 图的 561660 条非空检测中 124995 条是重复 raw 的额外类别展开（`conf=0.001`，不能外推为较高部署阈值下的重复率）。

## B. 官方 benchmark 对比（2026-10-13 查询，权重已核实）

官方固定版本 `ultralytics v8.4.100` 的文档宏 `docs/macros/yolo-seg-perf.md` 和当前实例分割文档均刊载 `yolo26m-seg.pt` 在 `COCO val2017, 640, e2e, single-scale` 的 `Box mAP50-95=52.5`、`Mask mAP50-95=44.1`（0–100 AP 点，官方仅保留一位小数）。

官方模型下载来源：`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt`；**官方 GitHub Release Asset 元数据 SHA256** 为：
`16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`，大小 `54,750,385` 字节。

本项目 `RUN_180459232c434a8ebc385f51dec06f63/run.json` 的模型输入 SHA256 和大小与上述**完全相同**；因此排除了该 Run 使用的权重文件不是该官方发布物的疑点。

| 指标 | 官方发布一位小数 | 本项目历史摘要精确值 | 差值（本地−官方，pp） |
|---|---:|---:|---:|
| bbox AP50:95 | 52.5 | 51.97992350896097 | −0.5200764910 |
| segm AP50:95 | 44.1 | 43.51845912911208 | −0.5815408709 |

本项目来源：`experiments/coco_raw_capability_5000_20260916/runs/RUN_a7c840bb695d45b4b66bc346715ca1aa/SUMMARY.json`，官方 5000 图、36335 普通 GT，Ultralytics 8.4.100，一致权重；其原始 `PROTOCOL.json` 使用 **自定义 batch1/auto LetterBox/unfused FP32 前向、raw 重建与非空 Mask 过滤、自定义 COCO 预测导出、pycocotools COCOeval**，不是直接执行官方 `model.val()` 的基线。

**结论：权重严格一致；性能数值不一致。** 官方仅发布一位小数，即使考虑四舍五入，当前约 0.52/0.58 点的差距仍非同一发表数值。**Box AP 也比官方低约 0.52**，所以仅靠 Mask decoder 差异不能完整解释。需要同时检查输入缩放/矩形批次策略、Head 分支/Top-K/筛选、类别映射和预测 JSON 等；然后再细查 Mask 解码。

### 可证实的评估代码区别，未定位其定量影响

- 官方 `SegmentationValidator.init_metrics` 若 `save_json=True` 则选 `ops.process_mask_native`；本地 `scripts/evaluate_raw.py` 自行整图计算全部 raw 的 FP32 Mask logits，按预测框裁切/阈值化并 `scale_masks` 到原图。两条可能因 GEMM/解码边缘及缩放实现产生差别，但还没有成对预测核验。
- 官方 `DetectionValidator.coco_evaluate` 在此版本使用 `faster_coco_eval.COCOeval_faster`；本地聚合历史记录走 `pycocotools`。无证据宣称这个实现差异就解释全部差距。
- 官方公开基线的验证命令含端到端参数 `nms=False`；本地虽然宣称并使用 O2O Head，但没有正式原生 val 作交叉证书，故不能声称与官方基线严格对齐。
- 自定义实验有 `conf=0.001`、`head_topk=300`、`maxDets=[1,10,100]`，还有省略空 Mask 输出行为；以上均必须同官方实际运行参数核对，而不能凭名称默认相同。
- 自定义采用每图独立 `auto LetterBox`（实际张量高度宽度不必 640×640），官方 val 的 dataloader/rect batch/原图回缩路径需以实际命令和数据加载日志为准。

## C. 唯一最小补证动作：同权重原生 COCO val，不重训

在拥有 COCO val2017 全部 5000 张图及同一权重的本地环境执行一次 **Ultralytics 8.4.100 原生验证**，原权重和旧 Run 不覆盖。官方当前公开说明可复现：
`yolo segment val model=<PINNED_OFFICIAL_PT> data=<LOCAL_COCO_YAML> imgsz=640 device=0 nms=False save_json=True`

这里 `model` 指向 SHA 与官方 Asset 元数据完全一致的模型；`data` 明确指向本地全部 5000 图和官方原始 COCO 类别/标注数据，不误用 coco8/coco128。日志保存实际 `batch`、`rect`、`conf`、`max_det`、`imgsz`、`nms`、`agnostic_nms`、`save_json`、`half`、`fuse`、input H×W 分布、Ultralytics/torch/pycocotools/faster-coco-eval 版本、GT 标注 SHA、结果 JSON SHA。

两种预先裁决：
1. **原生 val 与官方刊载的 52.5/44.1（发表精度）对齐：** 说明权重与官方基线可重现，自定义导出/评分路径存在需定位的协议差异。首个检查量是同图**预测 bbox+类别+分数集合**是否与自定义导出一致（Box AP 已低 0.52）；只有 Box 对齐后再专门检查 Mask decoder。
2. **原生 val 仍相差显著：** 不对模型错误或官方指标错误下结论；记录模型版本/实际配置/包依赖/数据清单等，维持 `official_baseline_parity=unresolved`，锁定唯一差异，禁止将旧 51.980/43.518 说成严格官方复现。

**本次没有 GPU、没有新 native val Run，不能伪报已查明 0.52/0.58 点的因果原因。** 以前本项目的研究结果若使用完全相同的自定义 evaluator + 同一权重基线，对照差仍有其限定科学意义；不要因为比官网少约 0.5 点就删除历史数值。

## 来源

- 固定源码：https://github.com/ultralytics/ultralytics/blob/v8.4.100/ultralytics/nn/modules/head.py
- 分支与后过滤：https://github.com/ultralytics/ultralytics/blob/v8.4.100/ultralytics/utils/nms.py
- 官方标签版基线：https://github.com/ultralytics/ultralytics/blob/v8.4.100/docs/macros/yolo-seg-perf.md
- 官方 Segmentation 复现：https://docs.ultralytics.com/tasks/segment
- 官方 release SHA：https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0
- 本地固定报告：`experiments/coco_raw_capability_5000_20260916/REPORT.md` 及对应 `run.json`/`SUMMARY.json`。

