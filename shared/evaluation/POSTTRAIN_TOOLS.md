# 后训练工具使用与验收边界

执行阶段只看 [ACTIVE_PLAN](../../project/ACTIVE_PLAN.md)。native_output_bridge 的 trace 子命令会在用户指定的本地环境加载冻结 YOLO 并做原生验证；其余工具负责离线重放、审计、优化组件或统计。没有工具在本次交付中启动训练或连接服务器。

## 当前实现和仍需本地完成的部分

| 工具 | 已实现 | 仍需要真实环境验证 |
|---|---|---|
| [build_native_ledger.py](../../experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py) | 原样预测 JSON、bbox/segm 独立 COCOeval、输出 ID、GT 账本、预算内外存在性、原生后端摘要核对 | numpy/pycocotools 联调、原生运行回执/日志、完整5000图和规定 smoke |
| [posttrain_audit.py](posttrain_audit.py) | 固定目标 T、确定性抽样、原生输出不变量、oracle 身份/分母/机会统计、失败阻断 | 真实账本和原生 decoder 已验证的逐实例 oracle 结果 |
| [finite_coefficient_solver.py](../../experiments/mask_readout_mechanism_v2_20261012/scripts/finite_coefficient_solver.py) | 固定 P 的有限系数优化、已知非零最优点控制 | torch 正控制、缓存与真实输出桥接、完整原图评分 |
| [native_output_bridge.py](native_output_bridge.py) | 真实 Top-K 索引观察、P/c0/F 缓存、真实层/几何/hash、原生两阶段 decoder、zero-delta/替换系数全输出 RLE 重放 | 固定版本真实权重的16图 smoke、5000图 trace 与精确独立重放均已通过；见 S0 REPORT |
| [adapter_primitives.py](../../experiments/mask_readout_mechanism_v2_20261012/scripts/adapter_primitives.py) | 无 GT 的预测框采样、P/z0 联合空间置换、监督与保持项独立归一化 | torch 梯度/数值测试、训练 runner 接入 |
| [damage_acceptance.py](damage_acceptance.py) | 无放回 GT 抽样、精确超几何单侧上界、预测/样本身份和分母核验 | 真 checkpoint 冻结及抽样证据、完整方法 COCO 账本 |
| [合成测试](tests/test_posttrain_audit.py) | 身份、分母、缺行、非有限数、budget failure、COCO 合成集与 solver 控制 | 无依赖测试和可选依赖测试分开报告；skip 不是通过 |

原生导出与重放代码已经提供；[新增测试](tests/test_posttrain_completion.py) 验证精确统计和空间/损失/decoder 组件。**完整 adapter 训练与评价 runner 已实现，真实模型桥接已通过笔记本 S0 验收。** [S0 报告](../../experiments/official_gt_failure_rebaseline_20261013/REPORT.md)分别记录原生账本通过和发表 AP 复现失败；[S1 报告](../../experiments/mask_readout_mechanism_v2_20261012/S1_REPORT_20261013.md)记录机会 Gate 通过，[S2 报告](../../experiments/mask_readout_mechanism_v2_20261012/S2_REPORT_20261013.md)及 [soft-IoU 复验报告](../../experiments/mask_readout_soft_iou_pilot_20261014/REPORT.md)记录两版配置均未通过原 S2 科学门槛。工程完成与方法效果分别判定；当前路由仍以总控为准，下述通用 CLI 不授权重复全量前向或追加训练。

## 命令（从仓库根目录；输出路径必须不存在）

先运行：

```text
python -m unittest discover -s shared/evaluation/tests -p "test_posttrain*.py" -v
python project/check_active_plan.py
```

numpy + pycocotools 在本地评价环境安装并锁实际版本；torch 使用模型执行环境的锁定版本。测试会明确跳过缺依赖项目，不能仅看退出码推断 COCO/solver 联调已通过。正式执行前用以下强制检查，缺依赖或测试未通过会非零退出：

```text
python shared/evaluation/tests/test_posttrain_audit.py --require-integration -v
python shared/evaluation/tests/test_posttrain_completion.py --require-integration -v
```

## 原生 trace 与重放

固定 ultralytics==8.4.100，需已安装并冻结 torch、numpy、pycocotools、faster-coco-eval（原生要求≥1.6.7）等运行依赖。[native_source_pins.json](native_source_pins.json) 校验该 tag 的8份关键源码，按 UTF-8/LF 归一化计算 SHA；源码变更不会仅凭版本字符串放行。实际 Torch/CUDA/TF32 设置也记录并在重放时核验。

NATIVE_ARGS.json 必须明确包含 data、imgsz、batch、conf、iou、max_det、rect、device、quantize、workers、end2end、agnostic_nms、single_cls、augment、compile、classes。例如固定原生 O2O、无增强的本地候选设置如下；真实取值仍需在 Run 中冻结，不代表官网 AP 已复现：

~~~json
{
  "data": "<local-coco-data.yaml>",
  "imgsz": 640,
  "batch": 4,
  "conf": 0.001,
  "iou": 0.7,
  "max_det": 300,
  "rect": true,
  "device": "0",
  "quantize": null,
  "workers": 0,
  "end2end": true,
  "agnostic_nms": false,
  "single_cls": false,
  "augment": false,
  "compile": false,
  "classes": null
}
~~~

smoke 的数据 YAML/val 图片清单必须事前限定至最多16图，且保持原生 COCO 识别路径规则及原始 annotations。image_ids 只是核验清单，工具不会偷偷截断5000图 dataloader。scope=full 要求完整5000图。模型与原始 COCO 标注 SHA 均强制核验。

~~~text
python shared/evaluation/native_output_bridge.py trace --weights <LOCAL_YOLO26M_SEG.pt> --annotations <instances_val2017.json> --args <NATIVE_ARGS.json> --image-ids <SMOKE_IDS.json> --scope smoke --max-cache-bytes <FROZEN_BYTE_BUDGET> --out <NEW_TRACE_DIR>
python shared/evaluation/native_output_bridge.py replay --manifest <TRACE_DIR/TRACE_MANIFEST.json> --predictions <TRACE_DIR/native/predictions.json> --out <NEW_ZERO_DELTA.json> --report <NEW_REPLAY_REPORT.json>
~~~

trace 不重算 Top-K tie：临时观察原生 get_topk_index 的实际返回，校验原生输出行再缓存对应 raw_id 的 c0/F。保留同 raw 多类别行；F 来自真实 one2one.feats，记录 head 模块和输入层编号、各层通道/空间尺寸以及串接 offset。8.4.100 AutoBackend 的底层模型不在普通子模块遍历里，工具使用其真实 backend.model 做 head 定位和状态指纹。观测函数在退出（包括异常）时恢复。

每图缓存全部正常输出，保留原 dtype、input_hw、ori_shape、ratio_pad、batch 位置、原图/输入 tensor SHA。预测框不依赖 GT。原生 forward/decoder 的两阶段顺序不改变：process_mask_native 生成输入尺寸的二值 mask，随后 scale_masks 还原原图并转 byte；不能在原图重新阈值化浮点 logits 来近似。

输出包括 TRACE_MANIFEST、OUTPUT_FEATURE_MAP、逐图 .pt、base_before/after 与原生 predictions.json。output_detection_id 仍是实际预测文件 SHA:原始行号；绝不用 COCO detection ID 猜 raw_id。每张图（含无检测图）均登记；缓存超预算、dtype/源码/行/几何/RLE不一致直接报错。

如需要替换系数，准备本地 torch 字典（str(image_id)→完整[N,K] c' tensor，同 dtype），记录其 SHA，再调用：

~~~text
python shared/evaluation/native_output_bridge.py replay --manifest <TRACE_MANIFEST.json> --predictions <NATIVE.json> --coefficients <C_PRIME.pt> --coefficients-sha256 <SHA256> --out <NEW_METHOD.json> --report <NEW_METHOD_REPLAY_REPORT.json>
~~~

工具先核原始 c0 的 exact RLE，然后才解码 c'；所有框/类别/score/行序完全保留，未修改行应显式提供原 c0。替换系数重放不等于 finite oracle 已完成，仍需要 GT 标准栅格、优化和完整原图评分。

本 CLI 面向预登记的 S0 smoke/full 原生验证，使第一次合法输出同时保存桥接缓存。若本地已有 S0 且缺缓存，不能因为这里存在 full 命令便重新全量验证：按执行合同只重放必需图片，沿用原 batch/rect/输出来源并额外核对旧基线身份；当前 CLI 不自动导入任意旧几何清单或替旧 Run 重标 ID。

S0 在完成原生推理后离线运行（full 为全5000图，smoke 最多16图）：

```text
python experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py --annotations <instances.json> --predictions <predictions.json> --image-ids <image_ids.json> --contract <EVAL_CONTRACT.json> --scope full --native-summary <native_eval_summary.json> --out <NEW_LEDGER_DIR>
```

image_ids.json 是图像 ID 的 JSON 列表。EVAL_CONTRACT 必须含 annotations_sha256、predictions_sha256、image_ids_sha256、ultralytics_version、model_sha256、end2end、agnostic_nms、conf、max_det、candidate_policy、gt_assisted_inference、run_id、code_commit、decoder、input_geometry_manifest_sha256；值取真实运行回执。全量还核原始标注 SHA/5000图/36335普通GT。工具核文件，无法从声明反证真实前向配置；native_gt_ledger_validity 保持待人工依据日志/回执裁决。

原生摘要格式：unit=AP_points，backend=COCOeval_faster，三个输入 SHA 与合同一致；bbox/segm 各含 AP、AP50、AP75、APS、APM、APL、AR1、AR10、AR100、ARS、ARM、ARL，未定义指标为 null。摘要必须来自该次原生评价，禁止将本工具的输出复制回去冒充双后端核验。backend parity≤0.02点；smoke 不判断官网基线复现。

S1 在 S0 正式验收后冻结目标：

```text
python shared/evaluation/posttrain_audit.py select --ledger <PER_GT_FAILURE.jsonl> --out <NEW_TARGET_PANEL.json>
```

默认全体 T；如冻结预算要求抽样，在看 oracle 前加 --limit N，按 image_id:annotation_id 的 SHA256 排序取前 N。保存 panel 文件 SHA，求解器必须返回其全部行；缺行或额外行直接报错。

原生桥接完成后调用 solve_coefficients(design, targets, c0)，**返回系数后必须经真实 decoder 评分**；禁止直接以 loss 网格 IoU 填 oracle_mask_iou。每行结果含 image_id、annotation_id、output_detection_id、source_prediction_sha、baseline_mask_iou、baseline_replay_equal、solver_code_sha256、oracle_config_sha256、solver_status、oracle_mask_iou、delta_norm、gradient_inf_norm。保留数值错误行，不从分母删除。

控制报告含 schema_version=1、positive_control_passed、same_solver_config、native_zero_delta_passed、known_coefficient_mask_control_passed、solver_code_sha256、oracle_config_sha256；各布尔值需有真实日志/产物证明，不允许直接填 true 代替测试。数值 positive control 和 native replay 是两项独立要求。

```text
python shared/evaluation/posttrain_audit.py summarize --panel <TARGET_PANEL.json> --results <G1_FIXED_OUTPUT_ORACLE.jsonl> --controls <SOLVER_CONTROLS.json> --out <NEW_G1_SUMMARY.json>
python shared/evaluation/posttrain_audit.py compare-outputs --baseline <native.json> --method <zero_delta.json> --zero-delta --out <NEW_IDENTITY_REPORT.json>
```

compare-outputs 要求顺序/行数/类别/score/bbox 一致；zero-delta 另核序列化 RLE 完全一致。普通方法验收去掉 --zero-delta，此时允许掩码改变。本地 runner 另用 `fingerprint_base_state(base_model)` 保存前后参数/BN buffers 和 train/eval 模式清单，并记录真实 requires_grad 参数列表。下面的检查会比较清单、白名单和 fit/dev 的图像 ID：

```text
python shared/evaluation/posttrain_audit.py check-run-identity --before <base_before.json> --after <base_after.json> --fit-ids <fit.json> --dev-ids <dev.json> --trainable-names <actual_names.json> --allowed-names <frozen_names.json> --out <NEW_STATE_REPORT.json>
```

清单比较不能代替实际模型导出；runner 必须从真实对象生成，不能手填一致的 SHA。

## 空间对照、损失与损伤

adapter_primitives 提供预测框采样、C 的联合空间置换与逐图分组 loss；调用规则和空组处理以 [机制/默认配置](../../experiments/mask_readout_mechanism_v2_20261012/MECHANISM_AND_DEFAULTS.md) 为准。不要在完整 runner 接入前把组件测试称为训练完成。

损伤抽样和验收命令见 [DAMAGE_ACCEPTANCE](DAMAGE_ACCEPTANCE.md)。它与 ΔIoU 的图像簇 bootstrap 是不同设计：512张效果开发图不能冒充随机 GT 损伤样本。有限总体全量时报告精确比例，未见场景推断另做 S4。

## 交付

把命令、版本、输入 SHA、实际退出状态、日志纳入新 Run，输出只写入新的目录/文件。工程测试结果不进入科研结果表；真实 Gate 仍需对应 protocol、合同及 Run 证据。旧10/10 solver 的 GT栅格化/框内并集不被此工具继承，详见 [机制与预设配置](../../experiments/mask_readout_mechanism_v2_20261012/MECHANISM_AND_DEFAULTS.md)。
