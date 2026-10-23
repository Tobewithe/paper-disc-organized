from common import dump, read, sha

NAMES = {'baseline': 'B*全真实基线', 'continued': 'B同预算续训', 'adapter': 'G普通Adapter',
         'random_memory': 'R随机字典', 'task_spm': 'F任务原型'}


def report(root, run, sources, metrics, costs, family, decision):
    delta = decision['measured_F_minus_continued_pp']; used = decision['operational_prototype_use_drop_exceeds_0_1_pp']
    target = decision['five_pp_metric_target_met']; specific = decision['F_exceeds_G_and_R_and_off_drops']
    f = metrics['task_spm']; b = metrics['continued']; base = metrics['baseline']
    status = '达到开发集+5点数值目标' if target else '尚未达到+5点目标'
    mechanism = 'F优于G/R且删除贡献掉点>.1pp' if specific else '原型特异性效用未共同获得支持'
    comparison = f"F Mask AP {f['mask_ap']:.4f}，相对同预算B {delta:+.4f}pp、G {decision['F_minus_comparators_pp']['adapter']:+.4f}pp、R {decision['F_minus_comparators_pp']['random_memory']:+.4f}pp、B* {f['mask_ap']-base['mask_ap']:+.4f}pp。Mask AP75相对B {f['mask_ap75']-b['mask_ap75']:+.4f}pp、B* {f['mask_ap75']-base['mask_ap75']:+.4f}pp；Box AP相对B {f['box_ap']-b['box_ap']:+.4f}pp。"
    drops = '；'.join(f"{k} {v:+.4f}pp" for k, v in decision['normal_minus_intervention_pp'].items())
    lines = ['# SPM-YOLO V4.1 全量实例任务原型实验', '',
             f"本实验完成1272真实训练图、432真实开发图、2048几何/4096外观、4096 cutout，B* 120轮、原型20000步及四臂各8000步。{status}；{mechanism}。本结果为单seed开发集选优，未使用独立test。", '',
             '| 维度 | 本次实验的整体研究链路 |', '|---|---|',
             '| **问题** | 上轮54图Run的P3门控近零，关原型AP几乎不变；小幅整体收益不能证明原型帮助分割。本实验检验实例监督和直接高分辨率Mask读出能否纠正这一缺口。 |',
             '| **机制假说** | H：fg/bg辅助目标缺乏实例归属监督，间接门控容易被绕开。跨外观实例匹配与直接Mask基底有望把学习原型变为有效输出。竞争解释为原型冗余，或任意额外特征/基底均可达到同效；F−G/R和同checkpoint定向破坏共同区分。 |',
             '| **针对性方法** | 固定同一全量B*的P2+P3与归一化，512→256→128投影按5k真实/cutout、10k混合、5k真实训练；train-only fg16/bg16中心。F增加32原型基底/32零初始系数，保留原32基底及原生O2O/process_mask。B/G/R/F同8k步、batch12=8real+2cut+2syn、相同样本与增强序列，前1k仅Mask路径，后7k最终head，上游/投影/中心冻结。 |',
             '| **整体结果** | '+comparison+'正常减干预：'+drops+f'。{status}；{mechanism}。破坏实验只说明该已训练输出是否依赖分支，不独立证明因果机制。 |',
             '| **研究决策** | 当前选定方向：`'+decision['chosen_research_direction']+'`。唯一下一步：`'+decision['unique_next_action']+'`，本次未启动后续训练。'+('完成本地权重/图像/SHA验收后才允许标记goal完成。' if target and specific else 'goal保持未完成，保留全部阴性/损伤；不得将代码完成或小收益当作5点达成。')+' |', '',
             '| 模型 | Mask AP | AP50 | AP75 | Box AP | 推理+解码中位毫秒 |', '|---|---:|---:|---:|---:|---:|']
    for mode, name in NAMES.items():
        m = metrics[mode]
        lines.append(f"| {name} | {m['mask_ap']:.4f} | {m['mask_ap50']:.4f} | {m['mask_ap75']:.4f} | {m['box_ap']:.4f} | {costs[mode]['median_inference_and_mask_decode_ms']:.2f} |")
    lines += ['', '分母均为同432张真实dev、同640画布；Mask mAP50–95及差值单位为绝对百分点。AP来自全量官方COCO评价，分组与展示IoU不替代AP。检测类别/score路径在同F干预前后逐项一致；原生空Mask过滤可能移除不同输出，已单独记录。', '',
              '| 原型干预（同一F权重） | Mask AP | 正常−干预pp |', '|---|---:|---:|']
    for intervention, difference in decision['normal_minus_intervention_pp'].items():
        lines.append(f"| {intervention} | {metrics['task_spm_'+intervention]['mask_ap']:.4f} | {difference:+.4f} |")
    lines += ['', '| 文件名family | 图数/实例 | F−B pp | F−B* pp |', '|---|---:|---:|---:|']
    for group, entry in family['families'].items():
        m = entry['metrics']; lines.append(f"| {group} | {entry['images']}/{entry['instances']} | {m['task_spm']['mask_ap']-m['continued']['mask_ap']:+.4f} | {m['task_spm']['mask_ap']-m['baseline']['mask_ap']:+.4f} |")
    params = {m: sources[m]['summary'].get('full_head_trainable_parameters') for m in NAMES}
    lines += ['', '文件名family不是已核实物理猪栏；它们是描述性开发分组。F/R有效可训练参数完全一致；G相对F的实际容量差为 '+str(params['adapter']-params['task_spm'])+'，G/F并非严格容量匹配，R用于该控制。G/R共享同投影预训练信息，不能把新增P2信息本身全部归给学习中心。', '',
              '原始数据仅使用原PigLife train的1704图分组切分；原test未参与训练、构库、阈值或模型选择。合成来自同一Quaternius猪资产；姿态、布局、密度、相机和外观随机化不等于真实分布完全覆盖。', '',
              '运行、复核与证据：', '', '- 冻结方法：[TASK_PROTOCOL_V3.md](TASK_PROTOCOL_V3.md)、[评价合同](EVALUATION_PROTOCOL.md)、[全量基线](BASELINE_PROTOCOL.md)。',
              f'- 总评价Run：`{run.name}`；源Run、检查点SHA与选优step见 [SOURCE_RUNS.json](runs/{run.name}/SOURCE_RUNS.json)。',
              f'- [全量指标](runs/{run.name}/METRICS.json)、[裁决](runs/{run.name}/DECISION.json)、[分组损伤](runs/{run.name}/FAMILY_RESULTS.json)、[真实贡献](runs/{run.name}/CONTRIBUTIONS.json)。',
              '- 原生/方法所有训练、工程失败和先前近零门控阴性均保留；GPU评价步骤0，未追加训练/seed/网格。',
              '- 训练和推理加载见 [USAGE.md](USAGE.md)，实际技术图见 [VISUAL_GUIDE.md](VISUAL_GUIDE.md)。', '']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (run/'REPORT.md').write_text('\n'.join(lines).replace('(TASK_PROTOCOL', '(../../TASK_PROTOCOL').replace('(EVALUATION_PROTOCOL', '(../../EVALUATION_PROTOCOL').replace('(BASELINE_PROTOCOL', '(../../BASELINE_PROTOCOL').replace('(runs/', '(../../runs/').replace('(USAGE.md)', '(../../USAGE.md)').replace('(VISUAL_GUIDE.md)', '(../../VISUAL_GUIDE.md)'), encoding='utf-8')
    guide = ['# V4.1真实技术与结果图集', '', '全部图来自本次实际数据、已训练权重和输出；英文图题对应下列说明。', '',
             '| 图 | 含义 |', '|---|---|']
    descriptions = {'00_method_and_supervision': '实际P2/P3→任务投影→32中心→额外32基底→64系数链路',
                    '01_training_and_primary_results': '四臂loss、所有开发AP与+5点主目标', '02_full_baseline_learning': '全量120轮基线学习与早晚验证轨迹',
                    '03_prototype_training_phases': '20000步实例任务原型的三阶段监督/梯度', '04_use_ablation_quality_and_cost': '同checkpoint删除/错配、AP75、Box与推理开销',
                    '05_data_coverage_and_poses': '全量输入、实例尺度与真实生成姿态频数', '06_actual_synthetic_scene_and_label_pairs': '真实已渲染场景、可见ID与成对外观',
                    '07_prototype_atlas_foreground': '前景16中心的真实train exemplar，红十字为stride4查询位置',
                    '07_prototype_atlas_background': '背景16中心及其真实来源/层支持', '08_prototype_support': '32槽利用率与各训练域支持',
                    '09_embedding_PCA': '同一学习嵌入按域和C/B/I/BG显示，仅作描述', '10_real_improvement_and_damage_cases': '固定规则选择最好3例和最差3例',
                    '11_contact_or_explicit_boundary_crops': '真实接触crop；无接触时显式标注边界fallback', '12_actual_prototype_responses': '实际前景/背景余弦、Top1及新增基底幅度',
                    '13_actual_mask_logit_contribution': '同一个真实最终检测的原32/新32/合计Mask logit，阈值与框裁剪前',
                    '14_all_family_damage_breakdown': '四个文件名family的完整描述性损伤'}
    for name, description in descriptions.items():
        guide.append(f'| [{name}](runs/{run.name}/figures/{name}.png) | {description} |')
    guide += ['', '选图账本和响应数据保存在同图目录；展示score .25、GT最佳IoU排序不参与模型选择或官方AP评价。PCA和prototype热图不独立证明效用。', '']
    (root/'VISUAL_GUIDE.md').write_text('\n'.join(guide), encoding='utf-8')
