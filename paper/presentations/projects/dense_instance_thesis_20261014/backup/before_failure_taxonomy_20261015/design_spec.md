<!-- ppt-master-schema: design-spec/v1 -->
# 通用密集检测分割毕业论文汇报 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | 通用密集检测分割毕业论文汇报 |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 22 |
| Primary Language | zh-CN |
| Target Audience | 导师与毕业论文汇报评审 |
| Communication Intent | 用通俗语言说明应用需求、现有不足、研究方法和系统用途 |
| Desired Audience Outcome | 先理解为什么需要研究，再理解具体做法，最后了解系统能做什么 |
| Core Message / Ask / Action | 改善密集场景中个体识别、轮廓区分和连续记录，形成通用系统，猪只作为应用案例 |
| Delivery Context | 现场口头汇报，正文约 16—20 分钟 |
| Artifact Afterlife | 可编辑 PPTX 与逐页讲稿 |
| Reading Mode | presentation |
| Content Strategy | 四部分，自定义诊断指标、真实探索实验和结论集中于第二部分，其它部分只讲应用和任务 |
| Design Style | 简洁、正式、易懂 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — retained |
| Custom Animations | disabled — retained |
| Narration Audio | disabled — retained |
| Created Date | 2026-10-14 |

- **Template Application**: 延续已安装 Academic Research 样式的字体与配色。按用户最新要求重组为通俗汇报，保留背景和系统章节，用实验数据图重建研究方法。
- **Design spec depth**: brief
- **Revision authority**: 用户确认三个任务模块，并要求详述模块一的模型、实验数据图和重要结论；模块二、三只写设想模型与实验。继续现有项目修订，不重启规划。

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | 0 0 1280 720 |
| Margins | 64 px |
| Content Area | x=64..1216, y=175..640；页脚 y=665..704 |

## III. Visual Theme

- **Mode**: custom
- **Mode References**: instructional, pyramid
- **Mode Behavior**: 背景说明应用需要与现有不足，方法按检测分割、连续身份保持、遮挡恢复组织；模块一展开真实证据，系统用使用流程说明功能。
- **Visual style**: custom
- **Visual Style References**: swiss-minimal
- **Visual Style Behavior**: 大字、开放留白、克制分隔线。避免卡片网格、密集脚注和多级小标题。细节图只在方法章节出现。
- **Theme**: 通俗的问题链；第一部分不出现候选、掩码、关联指标或具体算法名。
- **Tone**: 正式、自然，不使用口号、反问标题和制作说明。

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 画布 |
| Secondary background | #F1F5F8 | 方法图局部底色 |
| Primary | #172B46 | 标题 |
| Accent | #007F82 | 轮廓改进 |
| Secondary accent | #C36D37 | 困难与修正风险 |
| Body text | #25374C | 正文 |
| Secondary text | #596B80 | 注释 |
| Divider | #D4DEE7 | 分隔 |
| Instance selection | #315E9D | 个体选择 |
| Surface | #F7F9FB | 辅助底色 |

## IV. Typography System

- **Title stack**: Microsoft YaHei, Arial
- **Body stack**: Microsoft YaHei, Arial

| Role | Size |
| --- | --- |
| Cover title | 56 |
| Title | 40 |
| Subtitle | 32 |
| Lead | 32 |
| Body | 28 |
| Annotation | 22 |
| Footnote | 16 |
| Display | 72 |

## V. Layout Principles

- **Hierarchy direction**: 每页一个清楚主题。背景和系统页使用少量完整短句，技术细节限于 P05—P19。
- **Composition tendency**: 开放行列、两栏对照、实验数据图和使用步骤。避免全篇同一种三栏卡片。
- **Cross-page continuity**: 仅四个章节标识；蓝色表示对照、青色表示修正机会、橙色表示损伤风险。
- **Spacing posture**: 留白优先，删减文字而非缩小字号。
- **Spacing anchors**: page margin 64; block gap 24; column gutter 48; body leading 44.

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VII. Visualization Reference List

实验图采用既有数据与可编辑形状，标准绘图另存 PNG/SVG/PDF。

| Page | Object | Family | Source |
| --- | --- | --- | --- |
| P09 | failure-split | horizontal_bar_chart | E0—E4，保留原分母和区间 |
| P10 | oracle-area | horizontal_bar_chart | E0—E4，保留原分母和区间 |
| P12 | pilot-effects | scatter_chart | E0—E4，保留原分母和区间 |
| P13 | area-damage | horizontal_bar_chart | E0—E4，保留原分母和区间 |
| P14 | selective-effects | scatter_chart | E3/E5，同目标与总体配对 |
| P14 | selective-damage | column_chart | E3/E5，同1024原成功面板 |

## VIII. Image Resource List

无装饰图片。模型结构图采用可编辑节点、连线和文字；实验图来自真实 Run。

## IX. Content Outline

### Slide 01 - 研究定位
- **Title**: 面向密集场景的检测分割与持续跟踪方法及系统研究
- **Audience move**: 研究背景、研究方法、系统设计、研究安排；检测分割与持续跟踪共同构成研究对象。
- **Core message**: 研究背景、研究方法、系统设计、研究安排；检测分割与持续跟踪共同构成研究对象。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 文字封面
- **Content**: 研究背景、研究方法、系统设计、研究安排；检测分割与持续跟踪共同构成研究对象。

### Slide 02 - 应用需求
- **Title**: 连续个体监测的应用需求
- **Audience move**: 数量、位置、轮廓和连续记录。
- **Core message**: 数量、位置、轮廓和连续记录。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 开放文本行
- **Content**: 数量、位置、轮廓和连续记录。

### Slide 03 - 技术进展
- **Title**: 现有技术已能识别和跟踪目标，但目前在密集场景下仍难保持稳定
- **Audience move**: 保留已有能力与复杂场景不足的转折。
- **Core message**: 保留已有能力与复杂场景不足的转折。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 两栏能力与应用限制
- **Content**: 保留已有能力与复杂场景不足的转折。

### Slide 04 - 主要困难
- **Title**: 密集场景仍然存在的主要困难
- **Audience move**: 密集接触造成漏认与轮廓混合；交叉移动导致换号；遮挡重现后记录接不上。
- **Core message**: 密集接触造成漏认与轮廓混合；交叉移动导致换号；遮挡重现后记录接不上。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 场景与应用后果对照
- **Content**: 密集接触造成漏认与轮廓混合；交叉移动导致换号；遮挡重现后记录接不上。

### Slide 05 - 方法总览
- **Title**: 三个模块衔接单帧识别与连续身份记录
- **Audience move**: 一：检测与实例分割；二：连续身份保持；三：遮挡恢复。后二者未开展。
- **Core message**: 一：检测与实例分割；二：连续身份保持；三：遮挡恢复。后二者未开展。
- **Relationships**: order: 单帧输出支持连续关联，丢失后进入身份恢复。
- **Topology decision**: yes
- **Composition**: 三条横行，标清状态和衔接
- **Content**: 一：检测与实例分割；二：连续身份保持；三：遮挡恢复。后二者未开展。

### Slide 06 - 评价指标
- **Title**: 评价既看轮廓改善，也看原成功实例是否受损
- **Audience move**: 先定义目标T，再定义图宏ΔIoU、任务修复率、成功损伤率。IoU与AP不是自创新指标。
- **Core message**: 先定义目标T，再定义图宏ΔIoU、任务修复率、成功损伤率。IoU与AP不是自创新指标。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 四条指标解释与统一阈值
- **Content**: 先定义目标T，再定义图宏ΔIoU、任务修复率、成功损伤率。IoU与AP不是自创新指标。

### Slide 07 - 基础模型
- **Title**: 基础模型与轮廓生成过程
- **Audience move**: 冻结YOLO26m-seg；特征提取与融合输出框/类别/分数、32维c0和32通道P。原型与系数组合经原生解码得到轮廓。
- **Core message**: 冻结YOLO26m-seg；特征提取与融合输出框/类别/分数、32维c0和32通道P。原型与系数组合经原生解码得到轮廓。
- **Relationships**: link: 图像→特征→检测输出；P与c0共同生成轮廓。
- **Topology decision**: yes
- **Composition**: 原生模型支路图，冻结与新增修正分开
- **Content**: 冻结YOLO26m-seg；特征提取与融合输出框/类别/分数、32维c0和32通道P。原型与系数组合经原生解码得到轮廓。

### Slide 08 - 修正器结构
- **Title**: 轻量系数修正器的实际结构
- **Audience move**: 33×8×8空间输入→两层Conv64+GELU→均值64；F64+c32+box4+score1+class16；融合181→128→32维Δc。167872参数。
- **Core message**: 33×8×8空间输入→两层Conv64+GELU→均值64；F64+c32+box4+score1+class16；融合181→128→32维Δc。167872参数。
- **Relationships**: link: 空间与实例条件共同进入融合层，输出残差后修正c0。
- **Topology decision**: yes
- **Composition**: 双输入支路与融合结构图
- **Content**: 33×8×8空间输入→两层Conv64+GELU→均值64；F64+c32+box4+score1+class16；融合181→128→32维Δc。167872参数。

### Slide 09 - 失败定位
- **Title**: 6,229 个失败实例仍有合格检测框
- **Audience move**: 分割失败13378：无合格框7148、T6229、竞争1；T占46.56%。
- **Core message**: 分割失败13378：无合格框7148、T6229、竞争1；T占46.56%。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 横向数据条形图
- **Content**: 分割失败13378：无合格框7148、T6229、竞争1；T占46.56%。
- **Visualization**: failure-split=chart/horizontal_bar_chart; 310,230,850,500; 0..8000
- **Native-ready**: failure-split=no
- **Data class**: observed

### Slide 10 - 修正空间
- **Title**: 固定检测后，仍存在轮廓修正空间
- **Audience move**: GT辅助有限系数优化：122/256；GT面积分组38/148、44/64、40/44。
- **Core message**: GT辅助有限系数优化：122/256；GT面积分组38/148、44/64、40/44。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 数据图与总比例
- **Content**: GT辅助有限系数优化：122/256；GT面积分组38/148、44/64、40/44。
- **Visualization**: oracle-area=chart/horizontal_bar_chart; 230,226,810,466; 0..100
- **Native-ready**: oracle-area=no
- **Data class**: observed

### Slide 11 - 探索实验设计
- **Title**: 探索实验逐项检验修正条件
- **Audience move**: 空间置乱、正常空间BCE、同ROI soft-IoU、固定面积选择；前三者同2000图3轮1500步，最后零新增训练。
- **Core message**: 空间置乱、正常空间BCE、同ROI soft-IoU、固定面积选择；前三者同2000图3轮1500步，最后零新增训练。
- **Relationships**: contrast: 各实验保持同一基线，逐项区分空间、loss和选择条件。
- **Topology decision**: yes
- **Composition**: 四行实验设计对照
- **Content**: 空间置乱、正常空间BCE、同ROI soft-IoU、固定面积选择；前三者同2000图3轮1500步，最后零新增训练。

### Slide 12 - 自动修正探索
- **Title**: 无条件自动修正尚未兼顾收益与保护
- **Audience move**: 三种训练配置目标Δ−1.930、−0.996、+0.680pp及原95%CI。soft整体−0.262pp，损伤53/1024。
- **Core message**: 三种训练配置目标Δ−1.930、−0.996、+0.680pp及原95%CI。soft整体−0.262pp，损伤53/1024。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 点区间图
- **Content**: 三种训练配置目标Δ−1.930、−0.996、+0.680pp及原95%CI。soft整体−0.262pp，损伤53/1024。
- **Visualization**: pilot-effects=chart/scatter_chart; 420,224,1160,464; -4..2
- **Native-ready**: pilot-effects=no
- **Data class**: observed

### Slide 13 - 面积分层
- **Title**: 小掩码的修正损伤更集中
- **Audience move**: 原生预测面积S/M/L损伤37/270、14/424、2/330；小掩码占26.4%却占损伤69.8%。
- **Core message**: 原生预测面积S/M/L损伤37/270、14/424、2/330；小掩码占26.4%却占损伤69.8%。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 损伤率数据条形图
- **Content**: 原生预测面积S/M/L损伤37/270、14/424、2/330；小掩码占26.4%却占损伤69.8%。
- **Visualization**: area-damage=chart/horizontal_bar_chart; 250,220,1000,490; 0..16
- **Native-ready**: area-damage=no
- **Data class**: observed

### Slide 14 - 选择性修正结果
- **Title**: 选择性修正在开发筛查中兼顾收益与保护
- **Audience move**: 原生面积≥96²使用已有修正。目标+1.002pp，总体+0.089pp，损伤2/1024。原三项数值通过；复用开发面板非独立确认。
- **Core message**: 原生面积≥96²使用已有修正。目标+1.002pp，总体+0.089pp，损伤2/1024。原三项数值通过；复用开发面板非独立确认。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 左侧效应区间图、右侧同面板损伤对比图
- **Content**: 原生面积≥96²使用已有修正。目标+1.002pp，总体+0.089pp，损伤2/1024。原三项数值通过；复用开发面板非独立确认。
- **Visualization**: selective-effects=chart/scatter_chart; 310,240,740,510; -0.5..2.0. selective-damage=chart/column_chart; 865,240,1190,510; 0..6.
- **Native-ready**: selective-effects=no; selective-damage=no
- **Data class**: observed

### Slide 15 - 模块一结论
- **Title**: 模块一的重要结论与验证边界
- **Audience move**: 固定表示有机会；无条件修正有损伤；面积选择通过开发筛查。尚待同配置全5000图AP和全原成功总体损伤。
- **Core message**: 固定表示有机会；无条件修正有损伤；面积选择通过开发筛查。尚待同配置全5000图AP和全原成功总体损伤。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 三项结论与单个下一步
- **Content**: 固定表示有机会；无条件修正有损伤；面积选择通过开发筛查。尚待同配置全5000图AP和全原成功总体损伤。

### Slide 16 - 身份保持模型
- **Title**: 模块二：连续视频中的身份关联与稳定保持
- **Audience move**: 拟采用ByteTrack基线，参照BoT-SORT结合运动和外观。轮廓只在显式进入外观提取后参与身份判断。
- **Core message**: 拟采用ByteTrack基线，参照BoT-SORT结合运动和外观。轮廓只在显式进入外观提取后参与身份判断。
- **Relationships**: link: 当前观测与历史轨迹经运动/外观关联产生持续ID。
- **Topology decision**: yes
- **Composition**: 拟研究框架图及待验证问题
- **Content**: 拟采用ByteTrack基线，参照BoT-SORT结合运动和外观。轮廓只在显式进入外观提取后参与身份判断。

### Slide 17 - 身份保持实验
- **Title**: 模块二的实验设想
- **Audience move**: 同视频同检测，比较ByteTrack、运动外观、原生轮廓外观、修正轮廓外观；IDF1、ID切换、断裂和端到端速度。
- **Core message**: 同视频同检测，比较ByteTrack、运动外观、原生轮廓外观、修正轮廓外观；IDF1、ID切换、断裂和端到端速度。
- **Relationships**: contrast: 固定观测隔离关联收益，再比较原生/修正轮廓。
- **Topology decision**: yes
- **Composition**: 对照设计与统一评价
- **Content**: 同视频同检测，比较ByteTrack、运动外观、原生轮廓外观、修正轮廓外观；IDF1、ID切换、断裂和端到端速度。

### Slide 18 - 遮挡恢复模型
- **Title**: 模块三：遮挡后的身份恢复与轨迹续接
- **Audience move**: 沿用同一跟踪框架；lost记忆→候选筛选→连续观测确认→旧ID续接；不足则保持未确认或终止。
- **Core message**: 沿用同一跟踪框架；lost记忆→候选筛选→连续观测确认→旧ID续接；不足则保持未确认或终止。
- **Relationships**: order: 跟踪中→丢失记忆→重现待确认→恢复旧ID。
- **Topology decision**: yes
- **Composition**: 带分支的拟研究状态流程
- **Content**: 沿用同一跟踪框架；lost记忆→候选筛选→连续观测确认→旧ID续接；不足则保持未确认或终止。

### Slide 19 - 遮挡恢复实验
- **Title**: 模块三的实验设想
- **Audience move**: 默认保留→只延长→同预算加记忆→加多帧确认；所有合格重现事件共同分母，报告正确/误接/未恢复与延迟超时。
- **Core message**: 默认保留→只延长→同预算加记忆→加多帧确认；所有合格重现事件共同分母，报告正确/误接/未恢复与延迟超时。
- **Relationships**: contrast: 等待长度、记忆、确认分开消融。
- **Topology decision**: yes
- **Composition**: 消融对照与事件级指标
- **Content**: 默认保留→只延长→同预算加记忆→加多帧确认；所有合格重现事件共同分母，报告正确/误接/未恢复与延迟超时。

### Slide 20 - 系统功能
- **Title**: 通用系统的使用流程
- **Audience move**: 选图像或视频、查看位置轮廓编号、回看遮挡恢复记录、导出。拟建系统。
- **Core message**: 选图像或视频、查看位置轮廓编号、回看遮挡恢复记录、导出。拟建系统。
- **Relationships**: order: 导入任务→查看分析→保存记录。
- **Topology decision**: yes
- **Composition**: 使用者流程，技术细节不外溢
- **Content**: 选图像或视频、查看位置轮廓编号、回看遮挡恢复记录、导出。拟建系统。

### Slide 21 - 应用案例
- **Title**: 猪只群体监测的应用案例
- **Audience move**: 猪只仅作应用验证；其它类别需分别验证。
- **Core message**: 猪只仅作应用验证；其它类别需分别验证。
- **Relationships**: none
- **Topology decision**: no
- **Composition**: 应用问题与预期记录
- **Content**: 猪只仅作应用验证；其它类别需分别验证。

### Slide 22 - 研究安排
- **Title**: 现有基础与后续研究安排
- **Audience move**: 图像已有实验；视频身份与遮挡恢复尚未开展；后续全量图像验证、视频标注与方法实验、系统集成。
- **Core message**: 图像已有实验；视频身份与遮挡恢复尚未开展；后续全量图像验证、视频标注与方法实验、系统集成。
- **Relationships**: order: 完善图像验证→视频两模块→系统论文。
- **Topology decision**: yes
- **Composition**: 基础、任务与交付
- **Content**: 图像已有实验；视频身份与遮挡恢复尚未开展；后续全量图像验证、视频标注与方法实验、系统集成。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Content**: 每页自然讲稿，技术、方法图和实验指标限于P05—P19。图像模型与实验按真实Run描述；两个视频模块和系统标明拟开展。
- **Total duration**: 约16—20分钟，模块一为主要篇幅。
- **Evidence**: sources/三模块修订依据_20261015.md；E0—E5和原文献来源。
