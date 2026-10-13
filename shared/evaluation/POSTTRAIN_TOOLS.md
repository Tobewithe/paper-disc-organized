# S0 / S1 工具使用与验收边界

执行阶段只看 [ACTIVE_PLAN](../../project/ACTIVE_PLAN.md)。以下工具不会加载 YOLO、连接服务器或启动训练；有限系数求解组件例外地依赖本地 torch，但不训练基础模型。

## 当前实现和仍需本地完成的部分

| 工具 | 已实现 | 仍需要真实环境验证 |
|---|---|---|
| [build_native_ledger.py](../../experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py) | 原样预测 JSON、bbox/segm 独立 COCOeval、输出 ID、GT 账本、预算内外存在性、原生后端摘要核对 | numpy/pycocotools 联调、原生运行回执/日志、完整5000图和规定 smoke |
| [posttrain_audit.py](posttrain_audit.py) | 固定目标 T、确定性抽样、原生输出不变量、oracle 身份/分母/机会统计、失败阻断 | 真实账本和原生 decoder 已验证的逐实例 oracle 结果 |
| [finite_coefficient_solver.py](../../experiments/mask_readout_mechanism_v2_20261012/scripts/finite_coefficient_solver.py) | 固定 P 的有限系数优化、已知非零最优点控制 | torch 正控制、缓存与真实输出桥接、完整原图评分 |
| [合成测试](tests/test_posttrain_audit.py) | 身份、分母、缺行、非有限数、budget failure、COCO 合成集与 solver 控制 | 无依赖测试和可选依赖测试分开报告；skip 不是通过 |

**尚未实现的完整链路：** 原生模型前向的 P/c0/F 缓存导出与逐行 native decoder 桥接、adapter 训练 runner。不能把上述组件称作已跑通的 S0→S1→S2 一键系统；本地 agent 必须首先完成真实桥接 smoke。

## 命令（从仓库根目录；输出路径必须不存在）

先运行：

```text
python -m unittest discover -s shared/evaluation/tests -p "test_posttrain_audit.py" -v
python project/check_active_plan.py
```

numpy + pycocotools 在本地评价环境安装并锁实际版本；torch 使用模型执行环境的锁定版本。测试会明确跳过缺依赖项目，不能仅看退出码推断 COCO/solver 联调已通过。正式执行前用以下强制检查，缺依赖或测试未通过会非零退出：

```text
python shared/evaluation/tests/test_posttrain_audit.py --require-integration -v
```

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

## 交付

把命令、版本、输入 SHA、实际退出状态、日志纳入新 Run，输出只写入新的目录/文件。工程测试结果不进入科研结果表；真实 Gate 仍需对应 protocol、合同及 Run 证据。旧10/10 solver 的 GT栅格化/框内并集不被此工具继承，详见 [机制与预设配置](../../experiments/mask_readout_mechanism_v2_20261012/MECHANISM_AND_DEFAULTS.md)。
