# V18末C30技术与结果图

[48组原始图的实际视觉QA](C_MODEL_VISUAL_QA.json)已经完成。最终结果阴性，见[完整报告](V18_REPORT.md)；局部改善示例不能替代432图全量成绩。

| 内容 | 图与解释 |
|---|---|
| 训练与效用 | [C专用训练/任务汇总](C_TRAINING_SUMMARY/C_training_and_task_summary.png)：全部30轮训练曲线和固定10/20/30轮AP；另有同名SVG。通用技术图`02_training_and_memory_usage.png`含一个C固定面板空轴，原图保留，本补图使用原始日志，无新增模型前向。 |
| 最终任务结果 | [全量比较](runs/RUN_SPM_V17_RESULT_VISUALS_dcb46c54e4d8/01_full_utility_comparison.png)、[修复/损伤与像素错误](runs/RUN_SPM_V17_RESULT_VISUALS_dcb46c54e4d8/02_repairs_damage_and_pixel_errors.png)、[全程AP](runs/RUN_SPM_V17_RESULT_VISUALS_dcb46c54e4d8/04_full_mask_AP_curve.png)。 |
| 来源差异 | [清晰排版的四来源组增量](C_RESULT_FIGURE_REPAIR/source_group_utility.png)：原始长标签重叠图保留；G1–G4全名及原数值见同目录`INPUTS.json`，没有变更分组。 |
| 最终掩码案例 | [固定SHA及明确标出的最大改善/损伤](runs/RUN_SPM_V17_RESULT_VISUALS_dcb46c54e4d8/05_fixed_improved_and_harmed_cases.png)。绿TP、红FP、蓝FN；[上半](C_RESULT_FIGURE_REPAIR/fixed_gain_loss_cases_half_1.png)/[下半](C_RESULT_FIGURE_REPAIR/fixed_gain_loss_cases_half_2.png)仅为读图裁切。 |
| 实际融合链路 | [第一组](C_VISUAL_CONTACTS/fusion_01.png)/[第二组](C_VISUAL_CONTACTS/fusion_02.png)：8个固定案例的本检测原生概率、对象/边界/遮挡检索、重建内容、实际残差与最终Mask。来自348实际冻结模型，64例RLE重放精确一致。 |
| 记忆与条目 | [记忆总览](C_VISUAL_CONTACTS/overview_01.png)、[直接解码与检索](C_VISUAL_CONTACTS/overview_02.png)、[对象/遮挡案例](C_VISUAL_CONTACTS/overview_03.png)。直接解码使用共同局部上下文，不能独立证明128种语义。 |
| 重建、查询与归属 | [重建1](C_VISUAL_CONTACTS/reconstruction_01.png)/[2](C_VISUAL_CONTACTS/reconstruction_02.png)，[检索1](C_VISUAL_CONTACTS/retrieval_01.png)/[2](C_VISUAL_CONTACTS/retrieval_02.png)，[归属1](C_VISUAL_CONTACTS/ownership_01.png)/[2](C_VISUAL_CONTACTS/ownership_02.png)。C结构输出是辅助诊断，不能冒充最终Mask。 |

拼图与原图映射见[C_VISUAL_CONTACTS/INPUTS.json](C_VISUAL_CONTACTS/INPUTS.json)。35张技术原图位于`runs/RUN_SPM_V17_MODEL_VISUALS_C_86036c79c14b/`，5张任务图在`runs/RUN_SPM_V17_RESULT_VISUALS_dcb46c54e4d8/`，8张实际融合图在`runs/RUN_SPM_V17_NATIVE_CONDITION_VISUALS_79dd3f52ff53/`；CPU补图/排版修正单独保存，原始Run未修改。

完整A/B学习能力分别见[A图集](A_VISUAL_GUIDE.md)及[B图集](B_VISUAL_GUIDE.md)。用户追加的真实/Blender成对域诊断见[新图集](../spm_blender_domain_replay_20261024/VISUAL_GUIDE.md)。
