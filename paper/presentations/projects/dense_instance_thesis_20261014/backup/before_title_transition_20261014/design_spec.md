<!-- ppt-master-schema: design-spec/v1 -->
# 通用密集检测分割毕业论文汇报 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | 通用密集检测分割毕业论文汇报 |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 12 |
| Primary Language | zh-CN |
| Target Audience | 导师与毕业论文汇报评审 |
| Communication Intent | 用通俗语言说明应用需求、现有不足、研究方法和系统用途 |
| Desired Audience Outcome | 先理解为什么需要研究，再理解具体做法，最后了解系统能做什么 |
| Core Message / Ask / Action | 改善密集场景中个体识别、轮廓区分和连续记录，形成通用系统，猪只作为应用案例 |
| Delivery Context | 现场口头汇报，正文约 10—12 分钟 |
| Artifact Afterlife | 可编辑 PPTX 与逐页讲稿 |
| Reading Mode | presentation |
| Content Strategy | 四部分，技术名称、方法案例与指标集中于第二部分，其它部分只讲应用和任务 |
| Design Style | 简洁、正式、易懂 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — retained |
| Custom Animations | disabled — retained |
| Narration Audio | disabled — retained |
| Created Date | 2026-10-14 |

- **Template Application**: 延续已安装 Academic Research 样式的字体与配色。按用户最新要求重组为通俗汇报，删除正文文献时间线、数值大表和多层技术架构。
- **Design spec depth**: brief
- **Revision authority**: 用户要求通俗易懂、减少技术章节，具体方法与案例只在研究方法章节展开。继续现有项目的修订流程。

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
- **Mode Behavior**: 背景说明应用需要与现有不足，方法用少量示意讲清处理思路，系统用使用流程说明功能。
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

- **Hierarchy direction**: 每页一个清楚主题。背景和系统页使用少量完整短句，技术细节限于 P05—P09。
- **Composition tendency**: 开放行列、两栏对照、方法案例示意和使用步骤。避免全篇同一种三栏卡片。
- **Cross-page continuity**: 仅四个章节标识；方法一用蓝色、方法二用青色。
- **Spacing posture**: 留白优先，删减文字而非缩小字号。
- **Spacing anchors**: page margin 64; block gap 24; column gutter 48; body leading 44.

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |

无数值图表。方法示意为可编辑几何图，不表示本项目实测结果。

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

无新增图片。方法页保留可编辑案例示意，其它章节不展开实例分析。

## IX. Content Outline

### Part 0: 封面

#### Slide 01 - 研究定位
- **Audience move**: 明确论文题目和汇报的四个部分。
- **Title**: 面向密集场景的目标检测与实例分割方法及系统研究
- **Core message**: 围绕密集个体感知开展方法与系统研究。
- **Relationships**: none
- **Composition**: 深蓝纯文字封面，完整题目分两行，底部列四个章节名称。
- **Content**: 毕业论文研究方案汇报；研究背景、研究方法、系统设计、研究安排；2026.10.14。

### Part 1: 研究背景

#### Slide 02 - 应用需求
- **Audience move**: 理解持续记录个体的应用价值。
- **Title**: 连续个体监测的应用需求
- **Core message**: 动物、人员与车辆监测都需要可用的个体记录。
- **Relationships**: 数量、位置与区域分别对应不同记录需求。
- **Composition**: 一行场景说明与三个开放文本行，不画技术图。
- **Content**: 当前能看到多少个体；分别出现在哪里、停留多久；相邻个体的轮廓能否分开；稳定记录支撑后续统计。

#### Slide 03 - 技术进展
- **Audience move**: 理解现有技术已经可以识别目标和追踪移动。
- **Title**: 现有技术已经能够识别和跟踪目标
- **Core message**: 图像中的位置和轮廓、视频中的移动记录已有技术支持，但复杂条件仍需改善。
- **Relationships**: 图像与视频是两种应用输入；复杂条件构成共同限制。
- **Composition**: 图像与视频两栏能力概述，底部一句应用不足，不放模型名称和指标。
- **Content**: 图像中找到目标位置并描出轮廓；视频中连续记录个体移动；拥挤、接触和较长遮挡时仍可能出现记录中断或不稳定。

#### Slide 04 - 主要困难
- **Audience move**: 理解密集场景困难及其直接后果。
- **Title**: 密集场景仍然存在的主要困难
- **Core message**: 认不全、分不清与记录接不上分别影响应用。
- **Relationships**: 三种场景条件对应三类应用后果。
- **Composition**: 三个开放横行，场景在左，直接后果在右，不展示方法案例。
- **Content**: 个体拥挤时容易少认或重复认；接触时轮廓难分开；遮挡后可能换编号，原记录难接续；研究目标是提升个体记录可靠性。

### Part 2: 研究方法

#### Slide 05 - 研究思路
- **Audience move**: 理解两项方法研究与视频集成的关系。
- **Title**: 研究方法的总体安排
- **Core message**: 研究个体选择与轮廓修正，再在视频中检验连续记录。
- **Relationships**: 两项方法分别验证，视频集成承接其输出与应用评价。
- **Composition**: 两项方法并列，上下分出视频集成与验证，避免把跟踪包装成第三项新算法。
- **Content**: 方法一减少漏检与重复；方法二改善个体轮廓；接入已有跟踪方法，检查遮挡前后记录；先独立验证再观察完整视频。

#### Slide 06 - 个体识别方法
- **Audience move**: 通过两个候选案例理解需要判断个体对应关系。
- **Title**: 方法一：减少漏检和重复识别
- **Core message**: 结合预测完整性与邻近关系，判断哪些预测对应不同个体。
- **Relationships**: 两个个体高重叠应保留，同一个体多个预测需要去重，形成相反选择需求。
- **Composition**: 左右案例示意，下方给出具体研究思路和验证对象。
- **Content**: 关系辅助的候选判别与冗余控制；完整性、相邻关系；兼顾遗漏与重复；图示非实测。

#### Slide 07 - 轮廓改进方法
- **Audience move**: 理解修正有时有益、有时会损坏正确区域。
- **Title**: 方法二：改善相邻个体的轮廓
- **Core message**: 先生成候选修正，再预测是否值得采用。
- **Relationships**: 相同局部修改可能去除错分区域，也可能丢失正确细节；采用决策需区分两者。
- **Composition**: 两组修正案例，下方明确提出修改、判断收益、选择采用的思路。
- **Content**: 邻接关系辅助的掩码修正收益预测；原有状态、变化及邻居信息；评价修复与退化；图示非实测。

#### Slide 08 - 视频身份恢复
- **Audience move**: 理解目标重新出现后还要接回原身份，以及本课题的跟踪工作范围。
- **Title**: 视频中的个体跟踪与遮挡恢复
- **Core message**: 接入既有跟踪方法，单独检验能否正确续接原身份。
- **Relationships**: 遮挡前、不可见期间、重新可见构成连续过程。
- **Composition**: 三帧机制示意，底部说明接入方式与失败类型。
- **Content**: 保留历史记录，等待重现并尝试续接；结合现有运动关联方法；记录接错、重新编号、未找回；预测状态不等于可见观测。

#### Slide 09 - 效果验证
- **Audience move**: 理解如何判定方法和完整系统是否有效。
- **Title**: 研究方法与应用效果的验证
- **Core message**: 分别验证识别、轮廓、身份连续性和持续运行，并保持对照条件一致。
- **Relationships**: 每类研究目标对应评价内容；单项对照与组合对照有序开展。
- **Composition**: 四个开放行，左侧通俗检查内容，右侧必要指标，底部对照原则。
- **Content**: 遗漏与重复及 Box AP；区域误差与 Mask AP；HOTA、IDF1 和遮挡后续接；时延、掉帧和连续运行；标准 ByteTrack 不消费掩码，仅改掩码不推断身份收益。

### Part 3: 系统设计

#### Slide 10 - 系统功能
- **Audience move**: 理解用户使用系统时做什么、得到什么。
- **Title**: 通用系统的使用流程
- **Core message**: 选择图像或视频，查看个体分析，保存可回查记录。
- **Relationships**: 选择任务、查看分析、保存记录构成使用顺序。
- **Composition**: 三个开放步骤，只有用户动作和结果，不画模型或接口架构。
- **Content**: 选择文件与识别对象；查看个体位置、轮廓及编号；回看数量变化与个体记录；保存结果与报表；拟建系统。

#### Slide 11 - 应用案例
- **Audience move**: 理解猪只只是通用系统的一个应用场景。
- **Title**: 猪只群体监测的应用案例
- **Core message**: 以猪只场景检验系统能否提供可用的个体记录。
- **Relationships**: 场景问题对应用户可查看的记录。
- **Composition**: 关心的问题与预期记录两栏，不展示算法或实验细节。
- **Content**: 当前可见数量、个体位置与停留、遮挡后的记录衔接；数量变化、位置与区域记录；面向其它类别的应用需分别验证。

### Part 4: 研究安排

#### Slide 12 - 研究安排
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
