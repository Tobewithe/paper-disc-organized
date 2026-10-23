# V4.1真实技术与结果图集

全部图来自本次实际数据、已训练权重和输出；英文图题对应下列说明。

| 图 | 含义 |
|---|---|
| [00_method_and_supervision](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/00_method_and_supervision.png) | 实际P2/P3→任务投影→32中心→额外32基底→64系数链路 |
| [01_training_and_primary_results](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/01_training_and_primary_results.png) | 四臂loss、所有开发AP与+5点主目标 |
| [02_full_baseline_learning](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/02_full_baseline_learning.png) | 全量120轮基线学习与早晚验证轨迹 |
| [03_prototype_training_phases](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/03_prototype_training_phases.png) | 20000步实例任务原型的三阶段监督/梯度 |
| [04_use_ablation_quality_and_cost](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/04_use_ablation_quality_and_cost.png) | 同checkpoint删除/错配、AP75、Box与推理开销 |
| [05_data_coverage_and_poses](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/05_data_coverage_and_poses.png) | 全量输入、实例尺度与真实生成姿态频数 |
| [06_actual_synthetic_scene_and_label_pairs](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/06_actual_synthetic_scene_and_label_pairs.png) | 真实已渲染场景、可见ID与成对外观 |
| [07_prototype_atlas_foreground](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/07_prototype_atlas_foreground.png) | 前景16中心的真实train exemplar，红十字为stride4查询位置 |
| [07_prototype_atlas_background](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/07_prototype_atlas_background.png) | 背景16中心及其真实来源/层支持 |
| [08_prototype_support](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/08_prototype_support.png) | 32槽利用率与各训练域支持 |
| [09_embedding_PCA](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/09_embedding_PCA.png) | 同一学习嵌入按域和C/B/I/BG显示，仅作描述 |
| [09b_prototype_center_cosine](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/09b_prototype_center_cosine.png) | 32个真实中心的完整128维余弦矩阵，不从二维投影重叠推断塌缩或效用 |
| [10_real_improvement_and_damage_cases](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/10_real_improvement_and_damage_cases.png) | 固定规则选择最好3例和最差3例 |
| [11_contact_or_explicit_boundary_crops](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/11_contact_or_explicit_boundary_crops.png) | 真实接触crop；无接触时显式标注边界fallback |
| [12_actual_prototype_responses](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/12_actual_prototype_responses.png) | 实际前景/背景余弦、Top1及新增基底幅度 |
| [13_actual_mask_logit_contribution](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/13_actual_mask_logit_contribution.png) | 同一个真实最终检测的原32/新32/合计Mask logit，阈值与框裁剪前 |
| [14_all_family_damage_breakdown](runs/RUN_SPM_SCALE_EVALUATE_TASK_64db8fe5e5c0/figures/14_all_family_damage_breakdown.png) | 四个文件名family的完整描述性损伤 |

选图账本和响应数据保存在同图目录；展示score .25、GT最佳IoU排序不参与模型选择或官方AP评价。PCA和prototype热图不独立证明效用。

图1中保留的是全部检查点评价记录，模型文件按合同保留best与last；不表示保存了每个中间检查点权重。案例着色仅用于区分各图中的实例，颜色不表示跨模型/GT身份对应；请比较轮廓与逐对象账本。
