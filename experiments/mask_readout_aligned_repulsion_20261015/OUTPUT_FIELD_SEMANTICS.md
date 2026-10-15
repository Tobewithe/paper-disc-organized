# E4导出字段的解释

E4推理只运行原B adapter并对全部输出应用预测残差，不进行邻居选择；邻居仅用于REP训练损失。因此冻结导出脚本为兼容既有逐行证据格式，将所有 `neighbor_rank` 填为−1、`omega_pixels` 填为0。由此累计的 `coverage.no_eligible_neighbor` 不是“这些图不存在符合条件的邻居”的实测数量，不能用于邻域机制或覆盖率判断。实际训练的邻居数量见 `training.jsonl` 的 `matched_with_eligible_neighbor`。

`selected_coefficients`/`applied` 表示应用残差的输出集合，E4为完整185621条；不表示每条系数或二值mask都非零变化。二值mask实际变化数使用 `changed_masks`。科研收益与损伤由完整RLE及原固定配对/GT账本计算，不使用这些邻居占位字段作筛选。

保留冻结脚本和Run原字段，不事后重写日志。该说明只消除兼容字段名造成的解释歧义，不改变计算、分母、Gate或结果。
