# S0 原生基线复现与配置核验（2026-10-13）

**采用本次结果作为 `native_config_scoped` 的可复核本地基线；拒绝“官方公布指标已复现”的声明。** 全量原生输出、P/c0/F 身份桥接、精确零残差重放和独立 COCO 账本均通过；官方 52.5/44.1 一位小数门槛未通过。两个状态独立记录，不调参追分、不启动 S1 或训练。

主 Run：[RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170](runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/run.json)。历史核对：[RUN_HISTORY_RECONCILIATION_ea22c6c2ae3c49eda6899b1ea98f40f4](runs/RUN_HISTORY_RECONCILIATION_ea22c6c2ae3c49eda6899b1ea98f40f4/run.json)。执行端为笔记本 `YANGLEI`、RTX 4060 Laptop 8GB；没有在桌面机执行模型基线。

| COCO AP50:95（点） | 官方公开显示值 | 本次原生值 | 本次一位小数 | 判定 |
|---|---:|---:|---:|---|
| Box | 52.5 | 52.3514301541 | 52.4 | 未通过 |
| Mask | 44.1 | 44.0141748477 | 44.0 | 未通过 |

官方来源：[固定 v8.4.100 表](https://github.com/ultralytics/ultralytics/blob/v8.4.100/docs/macros/yolo-seg-perf.md)、[同版本复现说明](https://github.com/ultralytics/ultralytics/blob/v8.4.100/docs/en/tasks/segment.md)。官方未公布的小数位保持未知。本次两个评价后端的 bbox/segm 各 12 项指标**最大绝对差 0.0 点**，达到 ≤0.02 点门槛；完整结果见 [SUMMARY](runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/SUMMARY.json) 和 [诊断回执](runs/RUN_HISTORY_RECONCILIATION_ea22c6c2ae3c49eda6899b1ea98f40f4/NATIVE_LEDGER_DIAGNOSTICS.json)。

统一配置规范已落在[通用规范](../../project/科研项目通用规范/基线复现与配置一致性.md)，本研究采用**官方冻结权重 + 固定版本原生推理/COCO 评价合同 + 单独冻结的新增模块训练合同**。方法和消融的增益相对同协议本地基线计算；官方发表值单列。具体数值和训练记录差异见[配置说明](../../paper/manuscripts/neighbor_sensitive_p3/BASELINE_CONFIGURATION.md)、[profile](../../paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json)；[逐参数审计](runs/RUN_HISTORY_RECONCILIATION_ea22c6c2ae3c49eda6899b1ea98f40f4/CONFIG_PARAMETER_AUDIT.json)保存 111 个默认键、113 个权重训练键、调用覆盖和 111 个实际验证键，以及实际模型/数值环境。新增模块的完整训练合同尚未冻结，本轮没有训练。

本次固定 Ultralytics 8.4.100、原生 fused PyTorch FP32、batch=16、imgsz=640、rect=True、conf=.001、end2end=True、max_det=300、mask_ratio=4、overlap_mask=True、save_json=True。原生矩形验证包含 pad=.5，实际共有 23 种输入 H×W，长边为 672；不能用“640”代替真实几何。权重保存的 mask_ratio=1 属于训练记录，不能覆盖当前 val 合同。内部指标为 Box 51.7060840 / Mask 42.3047163，**不进入上述外部 COCO 主表**。mask_ratio、旧 dfl/semseg 等训练参数的存在，不证明它们在独立 val 中生效。

已经确定的事实：

- 原始权重 SHA、原始 COCO 标注 SHA、5000 图及类别映射通过；342 个框架源码/配置文件与固定官方发行包一致。5000 图在前向前后 SHA 一致，并与[官方 ZIP 目录的大小/CRC 校验](runs/RUN_HISTORY_RECONCILIATION_ea22c6c2ae3c49eda6899b1ea98f40f4/OFFICIAL_IMAGE_ARCHIVE_CHECK.json)全部相符。CRC 核验不冒充官方逐图 SHA 证明。
- 539128 条最终预测均追溯到同次前向的 raw 位置及 P/c0/F；基础状态未变，5000 图精确[零残差重放](runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/ZERO_DELTA.json)通过。原生完整预测 SHA 为 `a8c3e7d0af9db4684653c9b7b412f734d6105dc1e1c98a20c0782cf8d2fd3151`。
- 普通 GT 为 36335，另有 446 个 ignored GT；Mask75 成功 22957、失败 13378。失败中无 Box75 配对 7148，同一 Box75 检测自身 MaskIoU<.75 的目标 T 为 6229；另有 1 个匹配竞争案例 U。
- Box75 与 Mask75 同时成功的 22147 个 GT 中，1257 个匹配到了不同检测。U 案例为 image 43314 / annotation 524440：框匹配行索引 388385 的 MaskIoU=.773954984，但该行在 segm 评价中匹配到 annotation 477579。它属于匹配竞争，不能归为该检测 Mask 本体不达 .75。来源为主 Run 的 `ledger/PER_GT_FAILURE.jsonl`（SHA `8e563c9a3f32e6a3289b6649685c8160f557d74f19b35209b9f712f79c2e5884`）。

尚未证明：官方发布 Run 的全部实际条件与本次相同；AP 差距的具体来源；T 中系数修复的恢复率；新增 adapter 的可学性、AP 增益和成功实例损伤。原生导出/重放已有真实模型证据，空间置换、分组损失、损伤上界仍只有代码和工程测试证据；完整 adapter 训练 runner 尚未实现。权重中记录的 80 epochs、MuSGD 等不能证明完整官方训练过程已复现。

| 阶段 | 实际状态 | 证据位置 | 阻塞项 |
|---|---|---|---|
| S0 | 原生本地基线与 GT 账本通过；官方发表 AP 复现失败 | 主 Run 的 SUMMARY、ZERO_DELTA、EVAL_CONTRACT；历史核对 Run | 官方发布结果对应的完整验证回执尚未取得；具体差异来源待核实 |
| S1 | 未执行；6229 个 T 的身份和同次 P/c0/F 已具备 | 主 Run 的 ledger、trace/cache、OUTPUT_FEATURE_MAP | 尚未冻结本次实际目标/求解配置并运行有限系数 oracle；恢复率未知 |
| S2 | 未执行；组件测试通过，完整训练 runner 未完成 | `shared/evaluation/POSTTRAIN_TOOLS.md`、笔记本工程 Run | S1 机会 Gate、训练数据/合同/runner；不得直接开训练 |
| S3 | 未执行 | 当前 POSTTRAIN_RESCOPE / EXECUTION_CONTRACT | 同一 S2 checkpoint 及成功保护 Gate 尚不存在 |
| S4 | 未执行 | 当前方法协议 | S3 效用 Gate 与独立确认样本尚未满足 |

两份历史账本的逐 GT 对账均已完成；下面是协议/实现变更的对账结果，不是新增方法的修复收益。

| 历史来源 | 旧成功 | 旧成功→新成功 | 旧成功→新失败 | 旧失败→新成功 | 旧失败→新失败 |
|---|---:|---:|---:|---:|---:|
| raw 汇总 `RUN_a7c840bb695d45b4b66bc346715ca1aa` | 22767 | 21960 | 807 | 997 | 12571 |
| task alignment `RUN_1bea3cce89f441ec885b737feaa26989` | 22773 | 21962 | 811 | 995 | 12567 |

第二份原 `official_gt_mask75.json` 和分区图清单本地缺失；从两份原始预测按原顺序离线恢复，严格重现 Mask AP 43.5153060902、成功 22773。没有拿 Box50 槽位 `gt_inventory.csv` 代替任务匹配。具体变化 ID 见历史核对 Run 的两份 `*_OLD_NEW_GT_DELTA.csv`。

旧五臂/20k 草案 `PROTOCOL.md`、`GATES_SPEC.json` 已由 `POSTTRAIN_RESCOPE_20261013.md` 取代；TAL/E0、重预训练、全 raw 重扫不作为本主线依赖。历史 raw、argmax、共享仿射和短训阴性保留原对象与结论；旧 raw 的 batch=1/未融合/自定义几何与本次不同，`raw_link_status=not_comparable`，不得跨前向用 raw 数字索引硬连接。当前 13.19 GiB 的 P/c0/F 缓存、原生预测和 GT 账本可在同一合同下复用。

当前唯一下一步是**取得并核对官方发布 52.5/44.1 所对应的验证回执/配置与原始预测**：先比较同图框、类别、score 记录，再定位 mask 链路。未知阻塞在该发布 Run 的实际运行身份，不归咎于某个未被证实的参数。本轮预算已用完（一次 16 图前向 + 一次 5000 图前向），不依据 AP 结果修改参数重跑。只有新的可核验证据排除差异并通过原定两项一位小数门槛，才推翻“未复现”的结论；否则继续保持本地配置范围，禁止在论文中冒称官方复现。

本次先按用户授权停止旧 PID 22408，现有落盘产物保留，见[停止回执](OLD_TASK_STOP_RECEIPT.json)。笔记本工程 Run `RUN_ENGINEERING_LAPTOP_4996fad760f7412a88d8d16f09598fb9` 通过 24 项严格测试及配置检查。原 smoke 前向与 COCO 已完成，但结果记录钩子失败；原失败 Run 保留，修正后在 `RUN_SMOKE_RECEIPT_RECOVERY_f3d8b73fb3d94a2bb36ceb756db6c34f` 仅用已存输出/缓存通过 G1，没有第二次 smoke 前向。

轻量元数据、日志、配置、回执及历史对账已按原 Run ID 回传并校验 `transfer.json` / `manifest.sha256`；大预测、逐 GT 表和张量保留于 `D:\coco_wire\baseline_native_20261013\experiments\official_gt_failure_rebaseline_20261013\runs\RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170`，详见该 Run 的 ARTIFACT_MANIFEST 和 transfer 的 omitted 清单。仓库 HEAD 为交接提交 `b18802d208ccc235c4f994b68c3a40a98a440f1c`，实际未提交执行源码已冻结进 Run；用户原有改动保留，未提交或推送。
