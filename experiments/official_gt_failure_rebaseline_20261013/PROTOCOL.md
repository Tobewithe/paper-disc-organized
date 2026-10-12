# YOLO26m-seg 官方 COCO GT 失败重基线：冻结执行计划（2026-10-13）

**状态：`planned_not_executed`。本文件是本地 Codex 的可执行协议，不代表已经运行、已重标注或官方基线已复现。**
研究类型：诊断 / benchmark parity / per-GT 任务失败账本。唯一任务：**准确找出官方正常推理下 Mask75 未成功匹配的 COCO GT，再对其关联的同一最终预测检查 Box/Mask 质量**。不训练、不修改权重、不设计新模块、不重新遍历全部 30828840 raw 计算 Mask。

## 0. 已有证据、唯一待解决命题

- 权重：`yolo26m-seg.pt`，Ultralytics `8.4.100`；历史 `experiments/coco_raw_capability_5000_20260916/runs/RUN_180459232c434a8ebc385f51dec06f63/run.json` 记录的 SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5` 与官方 GitHub Releases v8.4.0 的同名模型文件**完全一致**。
- 官方 v8.4.100 文档宏 `docs/macros/yolo-seg-perf.md`：COCO val2017、640、单模型单尺度、E2E，Box AP50:95 **52.5**、Mask AP50:95 **44.1**（官方只发表一位小数）。
- 历史自定义预测导出/pycocotools Run：Box **51.9799235**、Mask **43.5184591**，未对齐官方发表精度；历史输出 Mask75 成功 **22767/36335**、失败 **13568/36335**，当前只能称为**历史自定义评估下**的逐 GT 标签。
- 历史全量 raw 能力诊断保留：共 30828840 raw；`exists_raw_mask75=24823`、`exists_raw_joint75=23805`，该批统计受**自身历史输入与解码口径**约束，不自动等同新原生验证。
- 历史 `raw_argmax_correct` 五类（含 6459 个 Mask fail）是独立语义条件的诊断，不是正常 one-to-one 输出的任务失败总体。
- **待证伪命题：** 在复现官方原生 O2O COCO 输出后，逐 GT `official_segm75` 定义稳定、可复核；交叉使用 `bbox` 的任务匹配 ID 时，不再把两个不同检测实例的框/掩码误拼为“同一个 raw 的质量”。

上述历史证据只作原始对照，不覆盖历史 `REPORT.md`、`SUMMARY.json`、`gt_joined.jsonl` 或原 Run。

## 1. 对象、规则与冻结参数

### 1.1 必须明确版本：`end2end` 才是 O2O 的推理选择

- 本计划固定 Ultralytics **`v8.4.100`**（项目 `shared/vendor/ultralytics_8_4_100/` 或相同发行版），官方相同 SHA 的权重、COCO 原始 val2017 全部 5000 图。
- 明确 `end2end=True` / `head.end2end=True`，`agnostic_nms=False` / `head.agnostic_nms=False`，`max_det=300`，`conf=0.001`（严格大于）。实际配置回执必须记录并断言。`head` 的该属性在 v8.4.100 原生 `validator.py` 中会接收验证器的 `agnostic_nms` 参数。
- **不要用 `nms=False` 来宣称已选择 one-to-one：** 在**固定版本** `ultralytics/cfg/default.yaml` 中，`nms` 是**导出时是否将 NMS 融合进导出模型**的参数；原生验证通过 `end2end` 控制输出分支。该版本官方 `docs/en/tasks/segment.md` 给出的复现命令是 `yolo val segment data=coco.yaml device=0`，未把 `nms=False` 列为分支选择器。必须依本版本源码、实际张量分支和配置断言，而不是套用其他版本或在线最新说明。
- 固定 `imgsz=640`、`split=val`、`save_json=True`，确认使用的是无 TTA 的同一权重。`batch`、`rect`、`precision`、`fuse` 和逐张输入实际 H×W 必须在正式跑前采用当前 **v8.4.100 原生 val** 的预期设置并冻结、记录；不得在看到 AP 后逐项调参追官网分数。
- **特别预检 COCO 路径**：v8.4.100 `DetectionValidator.init_metrics` 依据 `data[val]` 识别 `is_coco`，要求路径包含 `coco` 并以 `/val2017.txt` 结尾（Windows 用本机分隔符）；要断言 `is_coco=True`、`class_map==coco80_to_coco91_class()`、`save_json=True`，否则不能把普通 YOLO 类别映射或内部 TP 算作 COCO 官方任务匹配。必须实际验证 5000 张图片，不用 COCO8/COCO128。
- GT：原始 `instances_val2017.json`，SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`；`areaRng=all`、`useCats=1`、`maxDets=[1,10,100]`、`IoU=.50:.05:.95`。正常非 crowd/非 ignore GT 预期 36335；对 crowd/ignored 必须保留 COCOeval 本身的 ignore 机制，禁止当作普通失败 GT。
- `candidate_policy=inference_output`；实际推理不读取 GT，GT 只在离线 COCOeval 和诊断 IoU 阶段使用；不把 raw oracle 或 TAL replay 混入主失败定义。
- **本研究唯一主标签：** 对普通 GT `g`，如果在这次固定的 `segm` COCOeval、`IoU=.75`、`area=all`、`maxDet=100` 中，`gtMatches[threshold75,g] != 0`，则 `task_mask75=success`；反之 `failure`。`ignored` 不进入 36335 的分母。这是**从官方 COCOeval 输出导出的逐 GT 诊断状态**，不是官方单独公布的一个 AP 指标。

### 1.2 绝不能混淆“raw、最终检测、GT”

固定源码路径（详见已有 [Top-K 与基线审计](../../project/audits/candidate_eval_20261012/O2O_TOPK_OFFICIAL_COCO_BASELINE_20261013.md)）：Head 首先按每 raw 的最大类别分数选前 K=300 个 raw；随后从入选 raw 的 80 类分数展平选前 K=300 个 `(raw_id,class_id,score)` 组合；NMS-free `end2end` 后处理对组合输出行做 `score>conf` 过滤，才用对应 raw 的同一框和 Mask 系数生成结果。

一个 raw 可以输出多个类别，最终检测身份必须用独立 `output_detection_id`；正常 COCO JSON 通常没有 raw_id，禁止由 COCO 的 `dt_id` 推测 raw_id。

## 2. 阶段、依赖、验收与停机条件

### G0｜环境 / 资源 / 同一数据身份预检，不启动正式全量评估

1. 读取以上规范和历史 `PROTOCOL.json`、`run.json`、`SUMMARY.json`；冻结模型 SHA、`8.4.100` 源码、`COCO GT SHA`、5000 图清单 SHA、类别映射及 run 输出目录。模型文件 SHA 不一致、验证集非 5000 图、GT/类别映射异常：**停机 `insufficient_inputs`**，不得以替代权重或错配数据继续。
2. 采用原生 `model.val` / CLI；加载验证器并检查 `is_coco=True`、实际 `end2end=True`、`agnostic_nms=False`、`max_det=300`，记录所有实际参数及输入形状策略。记录 `validation_backend=COCOeval_faster`。只有确认使用 `COCOeval_faster` 的完整 bbox/segm 结果和原生 `predictions.json`，才可称原生 COCO 验证。
3. 不依赖目录名称自动算作成功：显式 assert 图片 ID、类别映射、`instances_val2017.json`、权重 SHA、输出路径和可写磁盘空间。
4. 原生 CLI 原型（配置预检通过后运行；`LOCAL_COCO_YAML` 必须指向符合上述 `val2017.txt` 的本地 COCO 数据集，而不是普通图片目录）：

   `yolo val segment model=assets/models/coco_clean_20260911/yolo26m-seg.pt data=<LOCAL_COCO_YAML> imgsz=640 split=val device=0 end2end=True agnostic_nms=False conf=0.001 max_det=300 save_json=True`

   将原生默认 `batch/rect/precision/fuse` 冻结写入 `EVAL_CONTRACT.json`。这**不是**强迫所有参数与旧自定义 raw Run 相等的命令；本阶段目标是复现**官方原生验证**，不是通过更改后处理消除差距。

### G1｜最小全链路烟雾测试，最多 16 图 + 合成测试

- 真实图：覆盖普通 GT、多 GT、空预测、crowd/ignore 以及重复 raw–类别可能性；检查原生图片输入尺寸、输出字段 `[image_id,category_id,score,bbox,segmentation]`、原图 RLE 尺寸、与 GT 类别映射。烟雾图的 AP **不得**当作 5000 图官方基线。
- 合成测试 A：一个 raw 两个类别输出共享几何，两个 `output_detection_id` 不同，不按 raw_id 去重。
- 合成测试 B：`d1` 通过 Box75、`d2` 通过 Mask75、同 GT 但不同预测，`bbox_match_id != segm_match_id`；禁止用 `bbox success & segm success` 断言某单条预测 Box/Mask 都正确。
- 合成测试 C：GT crowd/ignore、类别错误、高分候选竞争、同分排序、`IoU=.75` 边界、无预测图，核对 `COCOeval.evalImgs` 逐阈值匹配与输出行 ID 的稳定性。
- 任何身份 join、解码 RLE、有效 GT 计数或匹配不通过：**只修复当前实现，G1 重新通过前不得启动 G2**。不扩大样本、不改变统计目标。

### G2｜只做一次全量原生评估 + 官方发表基线 Parity

- G1 通过后，在全 5000 张 COCO val2017 使用冻结配置跑**一次**原生验证；保存 `predictions.json`、终端原始日志、实际参数、模型/GT/图像清单 SHA、分支证明、原生 Box/Mask 12 项 COCO 指标和 Run 元数据。历史 Run 不覆盖。
- 对照官方**发表到 1 位小数**的 Box 52.5、Mask 44.1：只有两者以 `round(AP_points,1)` 比对均一致，才标记 `published_baseline_parity=passed`；原始浮点值仍保留，不假装知道官方未发表的更多精度。
- 任一不一致则 `published_baseline_parity=failed`：**停止将该 Run 的 GT 分类宣称为“已复现官方公布基线”**；保存该 Run，转向只读对比预测导出/数据加载/分支/输入形状/类别映射，不自动尝试若干 batch、阈值、NMS 参数来追分。可作为 `native_unaligned_diagnostic` 暂存，但不能作为正式最终任务失败主账本发表。
- 注意旧评估连 Box AP 都低 0.520 点，因此先审计“同图预测框+类别+分数记录集是否一致”，再审计 Mask 解码。不能只将差距归结为 Mask 预测。

### G3｜从**同一份**原生预测 JSON 建立逐 GT 官方失败账本

仅当 G2 parity 通过后执行（无需重新前向）。

1. 冻结原生 `predictions.json` SHA，顺序固定；给每条最终检测写唯一 `output_detection_id` 和 `source_pred_json_index`。对同一列表制作 `bbox` 评价投影（bbox/score/cat/image）和 `segm` 评价投影（segmentation/score/cat/image），**顺序不变**；segm 投影不要携带 bbox，以避免 pycocotools `loadRes` 使用 bbox area 代替 mask area 导致尺度统计偏差。
2. 每种投影独立调用 `pycocotools.COCOeval`，冻结 `iouThrs`、`area=all`、`maxDets=[1,10,100]`、`useCats=1`。通过 `evalImgs` 取 `gtIds`、`gtMatches[5]`（.75）、`gtIgnore`、`dtIds`、`dtScores`，必要时 .50 同步取第 0 行。**不得用 Ultralytics validator 内部 `match_predictions` 冒充 COCOeval GT 配对。**
3. 对每个 `(image_id, annotation_id)` 普通 GT 记录：
   - 主：`task_mask75`、`task_mask50`、`task_box75`；
   - 官方独立任务匹配身份：`bbox75_match_output_id`、`segm75_match_output_id`，必须保留二者可能不同；
   - 固定具体 bbox 匹配检测 `d_b` 后：`box_iou_of_bbox_match`、`mask_iou_of_bbox_match`、`bbox_match_score`、`bbox_match_category_id`；若无匹配写 null，不能从其他 d 的 Mask 填值；
   - 诊断辅助（可由同一最终集合求）：`exists_same_output_joint75_for_gt`，含义是有**同一条正确类别最终检测**同时 BoxIoU≥.75、MaskIoU≥.75；这是每 GT 的存在性诊断，**不是唯一匹配后的官方 TP/召回**。如需唯一性，另在图内求限制一对一匹配并显式命名，不能把存在性之和冒称真正检测召回；
   - `correct_class_mask75_present_in_final_candidates`（是否任何正确类别最终输出 MaskIoU≥.75），可用于识别“候选存在却官方未匹配”的竞争问题；存在性不自动等于 GT 成功；
   - `is_ignored`、`gt_area`、`gt_category_id`、`source_prediction_sha`、`branch`、`eval_contract_sha`。
4. 最小互斥读表（仅作为**任务状态及表型**，不宣称网络因果）：正常 GT 按 `mask75=success`；否则分为 `mask75=fail & box75=fail`、`mask75=fail & box75=success & mask_iou_of_bbox_match < .75`、`mask75=fail & box75=success & mask_iou_of_bbox_match >= .75`。最后一类必须检查 GT 竞争/阈值/ID 映射，不能静默定为 Mask 本体坏。所有使用 ≥、< 的地方按实际浮点比较规则和 COCO threshold 固定。
5. 同预测记录的 Box/Mask 质量描述与独立 COCO 任务匹配**始终两套字段**。任何 GT 辅助 raw 几何 Hungarian 最优位置不允许替代 `bbox75_match_output_id`。

**重算完成不变量：** 普通 GT 唯一且总数与官方注释一致；`success+failure=36335`（ignored 单独）；每个成功 GT 的 matched dt ID 可以映射到唯一原生预测行，且图片与类别一致；同一任务同一阈值不重复把一个普通 detection 计为两个普通 GT TP；按原生 JSON 重新算的 bbox/segm AP 与原生 `COCOeval_faster` 精确值差绝对 ≤ **0.02 AP 点**（多阈值主指标），否则记 `coco_backend_parity_failed`、停止 G4，先核对投影 area、类别、ignore、排序与阈值，而不是继续输出失败结论。

### G4｜与既有历史报告差异对账（只读历史）

- 旧 `gt_joined.jsonl` 在历史源 Run 的本地目录（未同步到 GitHub）寻找，不得猜测它已在 GitHub。用 `annotation_id`、`image_id`、`category_id` 校验后，逐 GT 生成 2×2：
  - 旧成功→新成功
  - 旧成功→新失败
  - 旧失败→新成功
  - 旧失败→新失败
  同时列旧/新 `Mask75` 正确数、失败数、基线 Mask AP 和受影响 GT ID。
- 旧 `mask75 成功=22767` 仅用于准确描述原历史协议，**不预期**新结果一定也是 22767；任何差别需解释为输入、分支、输出、解码或 COCO 评分的可验证差异，不能直接定为某种网络错误。
- 复用 `raw_all` / `raw_argmax_correct` 历史结果**只在**模型 SHA、GT ID、`branch`、`input_hw`、letterbox、Mask decoder 质量关系经核验可比时执行逐 raw 身份关联。若原生输入大小/letterbox/native 解码与旧 raw 不同，写 `raw_link_status=not_comparable` 并保留原研究分支的统计；**绝不跨不同前向/不同图像几何使用相同 raw 数字索引硬连接**。
- 如旧 `gt_joined.jsonl` 缺失，仍交付**新官方 GT 主账本**，旧新逐 GT 对照写 `blocked_missing_legacy_rows`，不得因此重算全部 30828840 raw。除非有不可替代的论文关键判断且经过新预注册，不允许打开全 raw 大重跑。

### G5｜完成、失败与唯一科研裁决

- 成功：G0、G1、G2 官方发表精度 parity、G3 双评估器一致性和逐 GT 总数/输出身份不变量全部通过；G4 若旧大表确实缺失，独立记录并不阻断新官方 GT 账本交付。完成 `REPORT.md`，只基于任务失败 GT 与固定检测的真实表现选择下一个**机制研究问题**，不是直接宣称训练 TAL 或 Prototype 导致。
- 未通过：指出**唯一失败 Gate** 和复核所需的最小证据，`status=blocked_at_G#`，禁止提高预算、换类别、换评价阈值、读 val 后重订规则、掩盖旧新差异，禁止宣称官方失败分解已更新。
- 本次不做模型改进、不扩充 raw 层指标、不以 GT 选最佳 raw 当“实际预测成功”，不将相同 raw 不同类别行误合并。

## 3. 预定产物与最小资源上限

新 Study 位于 `experiments/official_gt_failure_rebaseline_20261013/`。开始实际运行时由本地 runner 分配真实唯一 Run ID 于 `runs/<RUN_ID>/`；**不要预创建虚构 Run，也不要覆盖既有历史 Run**。只有完成实际执行才将 `status` 更新。

| 产物 | 内容/用途 |
|---|---|
| `EVAL_CONTRACT.json` | 所有参数、实际版本、模型/标注/图片/预测 SHA、输入形状分布、`candidate_policy=inference_output`、`gt_assisted=False`（推理阶段）、官方基线来源、固定门槛、旧 Run 引用 |
| `run.json`、`smoke_test.json`、日志 | 真实命令、执行环境、运行状态、G0/G1 断言结果及终止原因 |
| `predictions.json` + SHA | 完全由无 GT 原生 O2O 推理产生的输出；每检测含 Box/Mask/类别/score |
| `native_eval_summary.json`、`coco_cross_backend.json` | 原生 `faster_coco_eval` 指标和逐 GT 用 `pycocotools` 独立核验 |
| `PER_GT_FAILURE.jsonl` | 每个有效 GT 一行的主失败账本及同输出 Box/Mask 诊断 |
| `OUTPUT_ID_MAP.jsonl` | 源 JSON 行序→COCO 检测 ID；无法映射 raw 时 `raw_id=null` |
| `OLD_NEW_GT_DELTA.csv`（若可得） | 旧/新标签交叉表、具体变化 GT 及分母 |
| `SUMMARY.json`、`REPORT.md` | 最小汇总，Gate 结果，是否通过“官方基线复现”标准，可行动的下一步 |

**计算预算**：最多一次小样本烟雾推理（≤16 张真实图）+ 合成单元测试 + **一次**原生 5000 图验证；之后只处理保存的预测/GT 和历史缓存，不再运行 5000 图第二遍、不训练、不重新计算全部 raw Mask。硬件不可用、权重或数据缺失时停止并列出本地 Codex 唯一必要输入，不假定本会话能连接本机。

## 4. 当前已核实来源（不替代待执行 Run）

- [官方 v8.4.100 Segmentation 表](https://github.com/ultralytics/ultralytics/blob/v8.4.100/docs/macros/yolo-seg-perf.md)
- [官方 v8.4.100 Segmentation 复现说明](https://github.com/ultralytics/ultralytics/blob/v8.4.100/docs/en/tasks/segment.md)
- [官方 v8.4.100 `validator.py` 分支传播](https://github.com/ultralytics/ultralytics/blob/v8.4.100/ultralytics/engine/validator.py)
- [官方 v8.4.100 `cfg/default.yaml` 中 `end2end` / `nms` 定义](https://github.com/ultralytics/ultralytics/blob/v8.4.100/ultralytics/cfg/default.yaml)
- [官方 v8.4.100 `head.py` 两次 Top-K](https://github.com/ultralytics/ultralytics/blob/v8.4.100/ultralytics/nn/modules/head.py)
- [旧全量 raw 研究](../coco_raw_capability_5000_20260916/REPORT.md)
- [历史 argmax 子集诊断](../coco_raw_argmax_conditioned_20261005/REPORT.md)
- [本仓库候选与评估规范](../../project/科研项目通用规范/候选身份与评估口径.md)

