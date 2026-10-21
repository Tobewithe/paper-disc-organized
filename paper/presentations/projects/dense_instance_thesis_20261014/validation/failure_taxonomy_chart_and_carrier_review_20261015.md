# 失败分类补充版：图表与构图核对

本轮新增 P06/P08/P09，原失败统计图移至 P07。新页只有分类定义、阈值与用途，没有新实验结果，也没有将旧 O2M/raw 统计混入当前 O2O 数量。

P06 modules: executor-base, executor-table. 五状态×判定×用途以同列对齐的可编辑文本与水平分隔呈现；支持率为同页独立解释。

P08 modules: executor-base, executor-table. 五种连续错误分量采用相同列结构，以便逐类对照“测什么”和“为什么测”；不使用示意假数据。

P09 modules: executor-base. 左侧两个场景几何量与右侧关系类型分别解释，底部承接分层评价。语义是并行字段，不画因果箭头。

几何签名：P06 选择研究对象 → 状态表与独立支持量；P07 明确当前研究对象 → 已核验数量图；P08 区分修复策略 → 像素组成表；P09 控制场景差异 → 双栏条件说明；P10 评价改进 → 沿用收益与损伤解释。三个新页中两个共享表格结构，是为保持判定和用途逐行可比。

本轮六个数据图的完整对象与上一版已验证源逐字一致，数值源哈希再次核对。无需重新选择比例尺或重画几何。原位置计算器回执见 three_module_chart_and_carrier_review_20261015.md。

verify-charts: 07_失败定位.svg | object=failure-split | mode=direct-calc | scale=0..8000 | verified chart retained exactly

verify-charts: 13_修正空间.svg | object=oracle-area | mode=direct-calc | scale=0..100% | verified chart retained exactly

verify-charts: 15_自动修正探索.svg | object=pilot-effects | mode=decomposable-calc | scale=-4..2 pp | verified chart retained exactly

verify-charts: 16_面积分层.svg | object=area-damage | mode=direct-calc | scale=0..16% | verified chart retained exactly

verify-charts: 17_选择性修正结果.svg | object=selective-effects | mode=decomposable-calc | scale=-0.5..2 pp | verified chart retained exactly

verify-charts: 17_选择性修正结果.svg | object=selective-damage | mode=direct-calc | scale=0..6% | verified chart retained exactly

载体核对：表格用水平规则和齐列维持判定关系；条件解释不需要路径或独立轮廓载体。保留模型图中的矩形节点与带标记直线已能精确表达输入/分支/输出，未增加非矩形预设。分组靠位置与标题，强调靠独立加粗文字；本次没有需要额外括号或批注气泡的对象。顺序页延续编号，箭头本身是可编辑 line marker，并非未画出的关系。

全篇零渐变与滤镜：内容没有需要用光照或深度传达的属性。inline emphasis 为零是因强调落在独立完整句/数值上；没有图标池或图片资源要求。图表和表格为原生可编辑形状与文字，不声称带数据工作簿或原生表格对象。
