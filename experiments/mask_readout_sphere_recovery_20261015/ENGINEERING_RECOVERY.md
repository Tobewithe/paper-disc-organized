# E2恢复账本

科学方法仍是原E2球面方向更新、邻居输入和冲突门控的固定方法包。工程错误不算科学阴性，恢复也不能掩盖物理训练开销。

| 执行 | 实际状态和开销 | 证据 |
|---|---|---|
| 原smoke `RUN_SPHERICAL_SMOKE_be9e1894729a4391b09a03e8cc28b109` | 8fit/8dev原生重放及2步完成 | `../mask_readout_spherical_adapter_20261015/runs/` |
| 原正式 `RUN_SPHERICAL_TRAIN_efde25d2082348c69c765ce70d5d5550` | 315张日志；78个完成更新、另3张反传；第316张image503483无检测，邻居选择空列表索引报错 | 同目录该Run的 `training.jsonl`、`ERROR.txt`、`run.json` |
| 长路径恢复smoke `RUN_SPHERICAL_SMOKE_b9c4681ea9dc45c1b9b22740c4848c54` | 写源码快照时路径262字符失败，优化0步 | `../mask_readout_spherical_adapter_recovery_20261015/runs/` |
| 短目录smoke `RUN_SPHERICAL_SMOKE_aaeed61ffc704253a40dc53fea40f159` | 原8fit/8dev及新增空输出控制通过，2步后丢弃 | 本Study `runs/` |
| 恢复正式 `RUN_SPHERICAL_TRAIN_c1123b621e404af998c310e995db7540` | 原6000图访问/1500步完成，前315条loss和计数精确复现失败前缀 | 本Study `runs/`；`RECOVERY_VERIFICATION_LOCAL.json` |

修复只把原有 `n==0` 返回移到邻居选择调用之前。非空输出的参数、公式和训练目标不变。恢复每25步写工程备份，评价仍只用最后1500步checkpoint；中途备份不参与选择。

正式物理执行总账1578步，另有旧失败3张未更新反传。科学轨迹1500步；两次成功smoke共4步另记。旧run.json仅每25步更新，最后写着75步；实际78步来自完整315张日志与固定4图累积代码，不能把状态更新间隔当训练总数。

由于旧Run没有中间checkpoint，不能直接做丢失tensor状态的逐位比较。恢复采用相同初始化、非空图代码、缓存和图序；在继续之前要求全部315条完整JSON loss/计数精确一致。笔记本生产恢复检查和当前机器对实际新旧日志的独立比较均通过。image503483在新日志的三个epoch均保留，loss为0、no_output_image=true，未删图或改分母。

恢复last.pt SHA：`27fa6e22f6d5771a3f50015f79138b91840e1dd44ee7beaba0e5a8361e52c167`。训练后笔记本及当前机器都独立检查初始共享tensor、172705参数、原IMAGE_ORDERS/TRAINING_JOIN、6000访问、实际Adam状态1500步、loss组合和源码快照，均通过。科学结论须等待该checkpoint的完整原S2评价，训练成功本身不代表Gate通过。

两个失败Study的完整产物均回传并逐SHA核验，包括长路径文件；分别见两个Study的 `TRANSFER_VERIFICATION.json`。旧Run与旧源码不覆盖，新的空输出控制、路径恢复协议和预算说明在启动前冻结。
