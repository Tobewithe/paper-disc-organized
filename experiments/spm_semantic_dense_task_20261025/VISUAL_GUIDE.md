# V20 完整结果与重点技术图

10000步训练及验收均已结束，真实收益未通过。原始80张PNG、14份同图PDF全部保留，另补1张读入强度PNG及对应PDF；[校验与实际看图记录](VISUAL_QA.json)、[整体报告](REPORT.md)。

## 先看这三组

| 图 | 回答什么 |
|---|---|
| [完整正常输出成绩](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/task_results.png) | 真实框/掩码下降，合成掩码提高；四个真实来源分别如何变化。 |
| [真实/合成原型读入强度](figures/memory_read_mass_by_domain.png) | 真实部位与整体词表多被reject门拒绝，不能用“出现过多少ID”替代实际调用强度。 |
| [部位与整体的可视联系](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/anatomical_part_whole_links.png) | 绿色源部位在身体中的位置、相同128维词条单独解码的局部形状，以及四种源姿态代码。 |

## 词表对象本体

| 类型 | 代表图 | 容量 / 有训练支持 |
|---|---|---:|
| 部位 | [参考与词向量解码](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_representative.png) | 256 / 238 |
| 整体 | [不同视向与体形](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_representative.png) | 128 / 128 |
| 遮挡 | [身体与隐藏区域](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_occlusion_representative.png) | 64 / 63 |
| 边界 | [局部前景与轮廓](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_boundary_representative.png) | 64 / 64 |

每格上排为固定训练参考，下排为仅由词向量和类型解码的结果。图中保留原代码名称；旧prone实际是屈腿代理，见[源姿态侧视图](formal_preview/SOURCE_POSE_GEOMETRY.png)。UNSUPPORTED条目没有合格训练来源，不能把共享解码器输出的图案当作已学对象。所有512槽位都有逐页图，链接见文末。

## 检索、训练与数据

- [语义绑定成绩](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/semantic_binding.png)：前两项80%必要条件未通过；遮挡/边界是关联成功正样本的条件准确率。
- [整体姿态混淆](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/whole_pose_confusion.png)：保留拒绝和漏检；这是实例加权展示，主Gate仍按几何族归约。
- [各词条调用分布](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/vocabulary_usage.png)：归一化分布应与实际读入强度一起看。
- [类型内向量相似度](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/within_bank_similarity.png)：不把不同类型空间的余弦当作同一标尺。
- [训练曲线](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/training_curves.png)：损失下降与真实效用分别判断。
- [密度、遮挡和部位接触](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/data_coverage.png)、[实际正式图抽检](formal_preview/FORMAL_DATA_QA.jpg)。
- [正常误报/漏检](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_error_profile.png)、[实例计数](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/instance_counts.png)。

## 正常预测与原型调用

以下固定样例在看结果前按输入来源哈希选择，每域8张；GT只在前向结束后展示。三个面板颜色与ID各自独立，不表示已经建立同一预测对应。调用图选择最高分正常查询，黄色框为该查询ROI。灰色表示非拒绝读入不足0.1；真实部位图近乎全灰是实际低调用结果。

两个额外极值图为[最多损伤例](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/example_most_damaged.png)和[最多修复例](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/example_most_repaired.png)，仅作事后解释，不改变完整指标、阈值或权重选择。

| 输入来源 | GT/原生/V20比较 | 实际词条调用 |
|---|---|---|
| dev_0039_v1 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_dev_0039_v1.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_dev_0039_v1.png) |
| dev_0085_v0 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_dev_0085_v0.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_dev_0085_v0.png) |
| dev_0107_v0 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_dev_0107_v0.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_dev_0107_v0.png) |
| dev_0118_v1 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_dev_0118_v1.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_dev_0118_v1.png) |
| dev_0131_v1 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_dev_0131_v1.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_dev_0131_v1.png) |
| dev_0141_v1 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_dev_0141_v1.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_dev_0141_v1.png) |
| dev_0153_v0 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_dev_0153_v0.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_dev_0153_v0.png) |
| dev_0155_v0 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_dev_0155_v0.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_dev_0155_v0.png) |
| real/133 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_real_133.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_real_133.png) |
| real/1484 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_real_1484.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_real_1484.png) |
| real/1485 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_real_1485.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_real_1485.png) |
| real/1561 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_real_1561.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_real_1561.png) |
| real/473 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_real_473.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_real_473.png) |
| real/573 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_real_573.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_real_573.png) |
| real/839 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_real_839.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_real_839.png) |
| real/896 | [正常预测](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/normal_real_896.png) | [检索与局部图](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/reads_real_896.png) |

## 全部512词条逐页图

- part: [00](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_00.png), [01](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_01.png), [02](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_02.png), [03](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_03.png), [04](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_04.png), [05](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_05.png), [06](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_06.png), [07](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_07.png), [08](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_08.png), [09](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_09.png), [10](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_10.png), [11](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_11.png), [12](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_12.png), [13](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_13.png), [14](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_14.png), [15](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_part_all_15.png)
- whole: [00](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_all_00.png), [01](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_all_01.png), [02](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_all_02.png), [03](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_all_03.png), [04](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_all_04.png), [05](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_all_05.png), [06](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_all_06.png), [07](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_whole_all_07.png)
- occlusion: [00](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_occlusion_all_00.png), [01](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_occlusion_all_01.png), [02](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_occlusion_all_02.png), [03](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_occlusion_all_03.png)
- boundary: [00](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_boundary_all_00.png), [01](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_boundary_all_01.png), [02](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_boundary_all_02.png), [03](runs/RUN_SPM_V20_FINAL_VISUALS_9e0483f2648b/atlas_boundary_all_03.png)
