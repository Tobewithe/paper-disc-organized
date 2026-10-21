# 原生掩码分支联合适配：服务器自动结案记录

三臂各完成20k图×3轮、15000次更新及完整5000图评价。本记录依据已完成Run自动整理；配置和产物哈希已复核，尚未进行本机独立统计复算。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| 问题 | 旧G已有收益，但固定P的系数修复尚未充分兑现已知机会，需要检验原生分支适配是否提供额外实用收益。 |
| 机制假说 | 原型和系数头共同适配可能改善修复；竞争解释是训练预算或仅原生系数头已足够。参数容量未匹配，不能单独归因为纯协同机制。 |
| 针对性方法 | G_budget、C_native、PC_native共享20k、三轮、监督与优化配方；固定检测及9216面积门控，比较同服务器原生/旧G及两个预算参照，检查全成功GT损伤与成本。 |
| 整体结果 | 主门槛未通过；各方法相对原生与旧G增量、四项PC区间和护栏见下表。 |
| 研究决策 | 本配置未通过冻结主门槛，停止追加训练。优先保留本次低损伤候选中实测AP最高的G_budget，将其收益与可修机会、损伤代价纳入论文证据；唯一下一步是人工复核结案主张及适用边界，不自动开新实验。 |

| 方法 | Mask AP | 相对原生AP点 | 相对旧G AP点 | 原成功GT损伤 |
|---|---:|---:|---:|---:|
| Ns | 44.013836 | +0.000000 | -0.138557 | 0/22958 |
| G_old_s | 44.152393 | +0.138557 | +0.000000 | 90/22958 |
| G_budget | 44.265173 | +0.251337 | +0.112780 | 81/22958 |
| C_native | 44.253024 | +0.239188 | +0.100631 | 120/22958 |
| PC_native | 44.017703 | +0.003867 | -0.134690 | 198/22958 |

| PC必要比较 | AP差值 | 配对95%区间 | 最小增量 | 门槛 |
|---|---:|---|---:|---|
| PC_native_minus_Ns | +0.003867 | [-0.091515, +0.100371] | 0.50 | 未过 |
| PC_native_minus_G_old_s | -0.134690 | [-0.220748, -0.053630] | 0.30 | 未过 |
| PC_native_minus_G_budget | -0.247470 | [-0.328541, -0.175186] | 0.20 | 未过 |
| PC_native_minus_C_native | -0.235321 | [-0.311356, -0.172654] | 0.20 | 未过 |

护栏：{"native_success_damage": true, "all_output_identity": true, "inference_cost": true}。
冻结裁决：primary_joint_configuration_does_not_pass_frozen_investment_gate。
冻结唯一后继：stop_additional_training_for_this_configuration_and_consolidate_best_valid_method_evidence。
旧G参照为同服务器重放，历史笔记本分数不混作本次分母。全部验证图曾参与开发，本轮不是独立外部确认。

证据Run：RUN_JOINT_PREPARE_56fa3ba7fe974a9ab6df42599788ef75, RUN_JOINT_NATIVE_299731c6c64d4cc0acf3b0915401e1e6, RUN_JOINT_SMOKE_8c70c8df51504bdabbc3813a7ac7af9d, RUN_JOINT_TRAIN_G_budget_73cca72b692e4097a974426de2a5b226, RUN_JOINT_TRAIN_C_native_8544ae971c9644a5afbad4ad9bb59ed1, RUN_JOINT_TRAIN_PC_native_27a3311439904036bf09225118f81233, RUN_JOINT_EVAL_G_budget_b079a976da574ad8a65db95132583da8, RUN_JOINT_EVAL_C_native_0cd8bfef737b47f08a7870202b718b8a, RUN_JOINT_EVAL_PC_native_c00834e276ef476681793cfaab9dcdd9, RUN_JOINT_STATISTICS_6bf82147bf394da69fd4d6d4bc243e61.

配置/文件核验详见 CONFIGURATION_AUDIT.json；所有模型、预测和统计文件保留在各Run目录。
