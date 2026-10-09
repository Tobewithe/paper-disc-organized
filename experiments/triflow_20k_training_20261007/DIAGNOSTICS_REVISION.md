# 诊断精简执行记录

2026-10-08 按用户要求启用 `diagnostics_runtime.py`（SHA256 `c24bc853ff17c7f4a0be3bc847a8b6733283b2aa06bc16388ab1b84fca3380b3`）：普通训练和前七轮 AP 评估保留 5 项必要统计，其余 28 项诊断未计算；每轮首次任务梯度探针和最后一轮诊断使用原完整实现。实际求解、ridge、限幅、失败检查、损失、数据顺序、优化器及固定 8 轮预算保持。

`RUN_TRIFLOW_20K_DIAGNOSTICS_PARITY_S0` 在 4 张真实训练图的 7 个 chunk 上通过 13 项检查：系数、logits、所有损失、逐参数梯度、一次真实 AdamW 更新、优化器和全部 RNG 完全一致；原版自身重复性也一致，精简模式无条件数求谱，必要 Cholesky/solve 保留。`RUN_TRIFLOW_20K_DIAGNOSTICS_OBSERVER_ENGINEERING_VERIFY_S0` 核对真实 32 图、69 次更新、边界回调与原参考的全部科学状态一致，完整诊断 1 次、精简 68 次，最终状态 SHA256 `5f3c4751cd96132a26eeb586be37bca692b6f7c6f72061f8a32f275e1cd4987f`；`RUN_TRIFLOW_20K_DIAGNOSTICS_EVAL_PARITY_S0` 在 4 张真实验证图、194 个候选上核验全部系数、原图 RLE、框类分数及顺序完全一致。以上均为工程验证，不用于 AP 或方法能力结论。

单个真实 chunk 的三次计时中位数：模块前向 40.44 → 36.56 ms，一次训练更新 88.33 → 84.26 ms；这些不包含在线特征提取与完整评估，整轮速度改善待正式运行实测。

第二轮全 5,000 图评估与 284,335 项独立检查完成后，控制 Run `RUN_TRIFLOW_20K_DIAGNOSTICS_REVISION_S0` 保存真实断点：第 3 轮、图像游标 663、chunk4、已提交 93,899 次更新，snapshot SHA256 `53151d6dfaf13209c498af5c4fbdccd8957d33661108e73eb8719294e396d85b`。旧 R2 的 `failed/partial` 是用户授权执行修订的受控停止，历史保留，未判为科学失败。

`RUN_TRIFLOW_20K_TRAIN_S0_R3` 于 2026-10-08 00:20:59（北京时间）从该断点续跑，00:23 已实际更新至 94,437 次、第 3 轮图像游标 900。实际 R3 runtime、适配器、观察器、验证收据及补充协议在 Run 和 epoch capture 中分别锁定 SHA；原四份科学来源只表示基底字节，不冒称完整实际执行未改变。8 个终态 Run 的 559 个源文件/产物已回传并通过清单校验；运行中的 R3 仅回传来源和实时元数据，最终训练和后续逐轮结果仍待完成。

以上为 00:23 的执行快照；23:09 后的真实完成、终态回传及授权通知事实见 `EPOCH_EXECUTION_REVISION.md`，最终结论引用原 `REPORT.md` 与曲线报告 Run。
