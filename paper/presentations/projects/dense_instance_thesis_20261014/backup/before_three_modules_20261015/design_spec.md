<!-- ppt-master-schema: design-spec/v1 -->
# 通用密集检测分割毕业论文汇报 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | 通用密集检测分割毕业论文汇报 |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 15 |
| Primary Language | zh-CN |
| Target Audience | 导师与毕业论文汇报评审 |
| Communication Intent | 用通俗语言说明应用需求、现有不足、研究方法和系统用途 |
| Desired Audience Outcome | 先理解为什么需要研究，再理解具体做法，最后了解系统能做什么 |
| Core Message / Ask / Action | 改善密集场景中个体识别、轮廓区分和连续记录，形成通用系统，猪只作为应用案例 |
| Delivery Context | 现场口头汇报，正文约 12—15 分钟 |
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
- **Revision authority**: 用户于2026-10-15要求研究方法分成三个模块，先介绍自定义评价指标，再介绍探索实验和结论，图必须为真实实验数据图。继续现有项目的修订流程。

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
- **Mode Behavior**: 背景说明应用需要与现有不足，方法按评价、探索、风险分析组织真实证据，系统用使用流程说明功能。
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

- **Hierarchy direction**: 每页一个清楚主题。背景和系统页使用少量完整短句，技术细节限于 P05—P12。
- **Composition tendency**: 开放行列、两栏对照、实验数据图和使用步骤。避免全篇同一种三栏卡片。
- **Cross-page continuity**: 仅四个章节标识；蓝色表示对照、青色表示修正机会、橙色表示损伤风险。
- **Spacing posture**: 留白优先，删减文字而非缩小字号。
- **Spacing anchors**: page margin 64; block gap 24; column gutter 48; body leading 44.

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |
| P07 | horizontal_bar_chart | authored | failure-split：13,378个分割失败GT的互斥分类 |
| P08 | horizontal_bar_chart | authored | oracle-area：按GT面积的有限修正成功比例 |
| P09 | scatter_chart | authored | pilot-effects：图宏ΔIoU及原协议95%区间 |
| P10 | horizontal_bar_chart | authored | area-damage：原生预测面积分组的任务损伤率 |
| P11 | scatter_chart | authored | adjusted-effects：控制另一因素后的损伤率差与校正区间 |

五张图的数据来自现有真实Run。PPT中的图由可编辑几何与文字构成；标准绘图工具生成同数据PNG/SVG/PDF，保存在experiment_figures_20261015。全量来源、哈希及CSV随图保留。不将GT辅助优化解释为部署收益。

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

页面使用原生可编辑图形，没有位图图片依赖。所有图表数值来自experiment_figures_20261015/chart_data.json。

## IX. Content Outline

### Part 0: 封面

#### Slide 01 - 研究定位
- **Audience move**: 明确论文题目和汇报的四个部分。
- **Title**: 面向密集场景的目标检测与实例分割方法及系统研究
- **Core message**: 围绕密集个体感知开展方法与系统研究。
- **Relationships**: none
- **Composition**: 深蓝纯文字封面，完整题目分两行，底部列四个章节名称。
- **Content**: 毕业论文研究方案汇报；研究背景、研究方法、系统设计、研究安排；2026.10.15。

### Part 1: 研究背景

#### Slide 02 - 应用需求
- **Audience move**: 理解持续记录个体的应用价值。
- **Title**: 连续个体监测的应用需求
- **Core message**: 动物、人员与车辆监测都需要可用的个体记录。
- **Relationships**: 数量、位置与区域分别对应不同记录需求。
- **Composition**: 一行场景说明与三个开放文本行，不画技术图。
- **Content**: 当前能看到多少个体；分别出现在哪里、停留多久；相邻个体的轮廓能否分开；稳定记录支撑后续统计。

#### Slide 03 - 技术进展
- **Audience move**: 理解现有技术已有识别与跟踪能力，但密集场景中的稳定性仍有不足。
- **Title**: 现有技术已能识别和跟踪目标，但目前在密集场景下仍难保持稳定
- **Core message**: 图像中的位置和轮廓、视频中的移动记录已有技术支持，但复杂条件仍需改善。
- **Relationships**: 图像与视频是两种应用输入；复杂条件构成共同限制。
- **Composition**: 两行标题呈现已有能力与当前不足的转折；图像与视频两栏能力概述，底部说明复杂场景的具体限制。
- **Content**: 图像中找到目标位置并描出轮廓；视频中连续记录个体移动；拥挤、接触和较长遮挡时仍可能出现记录中断或不稳定。

#### Slide 04 - 主要困难
- **Audience move**: 理解密集场景困难及其直接后果。
- **Title**: 密集场景仍然存在的主要困难
- **Core message**: 认不全、分不清与记录接不上分别影响应用。
- **Relationships**: 三种场景条件对应三类应用后果。
- **Composition**: 三个开放横行，场景在左，直接后果在右，不展示方法案例。
- **Content**: 个体拥挤时容易少认或重复认；接触时轮廓难分开；遮挡后可能换编号，原记录难接续；研究目标是提升个体记录可靠性。

### Part 2: 研究方法

#### Slide 05 - 方法总览
- **Audience move**: 从评价、实验到风险解释，形成完整问题链。
- **Title**: 研究方法分为三个模块
- **Core message**: 从评价、实验到风险解释，形成完整问题链。
- **Relationships**: none
- **Composition**: 三个开放横行说明每个模块回答的问题。
- **Content**: 实例级评价与失败定位；轮廓修正探索；修正风险与适用范围。

#### Slide 06 - 评价指标
- **Audience move**: 定义研究使用的诊断指标，区分固定检测质量与任务成败。
- **Title**: 用实例级指标定位错误并衡量修正代价
- **Core message**: 定义研究使用的诊断指标，区分固定检测质量与任务成败。
- **Relationships**: none
- **Composition**: 四个开放文本行，底部解释IoU与阈值。
- **Content**: T目标条件；图宏平均ΔIoU；成功损伤率；不限类不同raw邻框最大IoU。

#### Slide 07 - 失败定位
- **Audience move**: 在真实最终输出中识别可以进入轮廓修正研究的目标。
- **Title**: 6,229 个失败实例仍有合格检测框
- **Core message**: 在真实最终输出中识别可以进入轮廓修正研究的目标。
- **Relationships**: none
- **Composition**: 左侧横向条形图；右侧46.56%与明确分母。
- **Content**: 13,378个Mask75失败GT：无Box75配对7148、T6229、匹配竞争1。
- **Visualization**: failure-split=chart/horizontal_bar_chart; experimental data; x=310..850,y=230..500; counts 0..8000
- **Native-ready**: failure-split=no
- **Fact ID**: E0
- **Data class**: observed

#### Slide 08 - 修正空间
- **Audience move**: 标注辅助有限优化在256个固定目标中找到122个达标结果。
- **Title**: 固定检测后，仍存在轮廓修正空间
- **Core message**: 标注辅助有限优化在256个固定目标中找到122个达标结果。
- **Relationships**: none
- **Composition**: 左侧按GT面积分组横向条形图，右侧总比例与解释。
- **Content**: 小38/148，中44/64，大40/44；合计47.66%；GT辅助、固定P/框，仅改系数。
- **Visualization**: oracle-area=chart/horizontal_bar_chart; experimental data; x=230..810,y=226..466; percent 0..100
- **Native-ready**: oracle-area=no
- **Fact ID**: E1
- **Data class**: observed

#### Slide 09 - 自动修正探索
- **Audience move**: 空间信息及loss替换有相对证据，但当前配置均未通过完整验收。
- **Title**: 自动修正尚未兼顾目标收益与整体保护
- **Core message**: 空间信息及loss替换有相对证据，但当前配置均未通过完整验收。
- **Relationships**: none
- **Composition**: 三行点区间图，底部列soft-IoU总体与损伤护栏。
- **Content**: 同T=655；空间打乱−1.930pp、正常BCE−0.996pp、soft-IoU+0.680pp；95%CI；soft总体−0.262pp、损伤53/1024、上界6.43%。
- **Visualization**: pilot-effects=chart/scatter_chart; experimental data; x=420..1160,y=224..464; signed pp -4..2
- **Native-ready**: pilot-effects=no
- **Fact ID**: E2/E3
- **Data class**: observed

#### Slide 10 - 面积分层
- **Audience move**: 面积能区分当前修正器的风险，需要保留原成功分母。
- **Title**: 小掩码的修正损伤更集中
- **Core message**: 面积能区分当前修正器的风险，需要保留原成功分母。
- **Relationships**: none
- **Composition**: 横向损伤率柱状图，每类标注损伤数及原成功数。
- **Content**: 原生预测面积S/M/L：37/270、14/424、2/330；小掩码占面板26.4%、占损伤69.8%；大目标点估计不能替代1%上界验收。
- **Visualization**: area-damage=chart/horizontal_bar_chart; experimental data; x=250..1000,y=220..490; percent 0..16
- **Native-ready**: area-damage=no
- **Fact ID**: E4
- **Data class**: observed

#### Slide 11 - 风险因素对比
- **Audience move**: 校正后面积差异仍为正，邻框重叠差异区间跨零。
- **Title**: 控制邻框重叠后，面积与损伤仍有关联
- **Core message**: 校正后面积差异仍为正，邻框重叠差异区间跨零。
- **Relationships**: none
- **Composition**: 两行点区间图，标签说明各自控制变量。
- **Content**: 原生面积小−中大+11.765pp，区间[5.665,18.710]；高−低邻框重叠+3.287pp，区间[−2.700,7.168]；8项比较校正。
- **Visualization**: adjusted-effects=chart/scatter_chart; experimental data; x=450..1160,y=225..465; signed pp -5..20
- **Native-ready**: adjusted-effects=no
- **Fact ID**: E4
- **Data class**: observed

#### Slide 12 - 结论与后续验证
- **Audience move**: 从探索结果提出一个可检验的后继方案，尚无方法有效性结论。
- **Title**: 已有证据支持先研究选择性修正
- **Core message**: 从探索结果提出一个可检验的后继方案，尚无方法有效性结论。
- **Relationships**: none
- **Composition**: 三个证据状态行与一条可信范围说明。
- **Content**: 已确认部分目标有修正机会且小掩码风险高；当前自动修正未通过完整验收；拟验证原生掩码面积≥96²时修正、其余保持原结果；COCO开发数据，密集专项及视频仍待验证。

### Part 3: 系统设计

#### Slide 13 - 系统功能
- **Audience move**: 理解用户使用系统时做什么、得到什么。
- **Title**: 通用系统的使用流程
- **Core message**: 选择图像或视频，查看个体分析，保存可回查记录。
- **Relationships**: 选择任务、查看分析、保存记录构成使用顺序。
- **Composition**: 三个开放步骤，只有用户动作和结果，不画模型或接口架构。
- **Content**: 选择文件与识别对象；查看个体位置、轮廓及编号；回看数量变化与个体记录；保存结果与报表；拟建系统。

#### Slide 14 - 应用案例
- **Audience move**: 理解猪只只是通用系统的一个应用场景。
- **Title**: 猪只群体监测的应用案例
- **Core message**: 以猪只场景检验系统能否提供可用的个体记录。
- **Relationships**: 场景问题对应用户可查看的记录。
- **Composition**: 关心的问题与预期记录两栏，不展示算法或实验细节。
- **Content**: 当前可见数量、个体位置与停留、遮挡后的记录衔接；数量变化、位置与区域记录；面向其它类别的应用需分别验证。

### Part 4: 研究安排

#### Slide 15 - 研究安排
- **Audience move**: 了解现有基础、尚需工作和最终交付。
- **Title**: 现有基础与后续研究安排
- **Core message**: 复用图像与处理流程，补充视频、完成方法验证并形成可演示系统。
- **Relationships**: 现有基础支持后续三阶段工作，最终汇聚为论文与系统成果。
- **Composition**: 上方简述基础，中部三个顺序任务，底部预期交付，不放数值或具体技术方法。
- **Content**: 已有动物图像、推理导出流程与前期实验经验可复用；整理视频、完成方法验证、系统联调与论文整理；形成方法验证结果和可演示通用系统。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: 讲稿与画面同一难度。背景、系统和安排只讲应用与任务，算法、案例分析和指标限于 P05—P09。方法图均为示意，方法与系统均按拟研究表述。
- **Total duration**: 约 10—12 分钟，重点分配给研究方法。
- **Notes style**: 自然连贯的正式口头讲稿，不朗读内部字段和制作流程。
- **Evidence**: 已核验文献和工程资料沿用 sources/应用与视频强化版_引用与证据.md；本轮不新增科学结果，不改旧实验或研究路线。
