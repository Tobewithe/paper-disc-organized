<!-- ppt-master-schema: design-spec/v1 -->
# 通用密集检测分割毕业论文汇报 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | 通用密集检测分割毕业论文汇报 |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 16 |
| Primary Language | zh-CN |
| Target Audience | 导师与毕业论文汇报评审 |
| Communication Intent | 解释应用需求、业界进展、尚存矛盾和两个攻克方向，说明如何形成通用系统；实验与成果只简要交代 |
| Desired Audience Outcome | 理解实例选择与区域修正的独立难点，认可可复用基础与可检验的实施路径 |
| Core Message / Ask / Action | 密集感知需要同时做好实例对应与可见区域；两个方法方向接入通用系统，猪只作为应用案例 |
| Delivery Context | 明天现场口头汇报，正文约 12—15 分钟；备用页用于问答 |
| Artifact Afterlife | 可编辑 PPTX、逐页讲稿、后续研究汇报复用 |
| Reading Mode | presentation |
| Content Strategy | 按现有 13 页问题链提炼，保留事实与限定词，补充三页备用材料 |
| Design Style | 问题链驱动的学术极简 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — workflow default |
| Custom Animations | disabled — workflow default |
| Narration Audio | disabled — workflow default |
| Created Date | 2026-10-14 |

- **Template Application**: 使用已安装的 Academic Research 样式模板，继承其问题—已有进展—方法—证据边界的叙述方式、清晰图解和限制条件可见的要求；正文按用户指定的问题链组织，自主设计版式。模板只提供方法与风格，不带机构标识。
- **Design spec depth**: brief
- **Delegation**: 用户明确要求自行安装、选择模板并制作成稿；本文件记录代理决策，不伪造确认 UI 回执。
- **Direction comparison**: 方案 1 为问题链图解：同一处理链逐页聚焦，单一无衬线字体，白底与两种语义色；方案 2 为论文评述：文献证据占较大比重，以宋体标题、微软雅黑正文形成对比，灰白栏式版面，细线平面示意；方案 3 为系统演示：以用户操作顺序切入，深蓝场景页和明亮方法页交替，微软雅黑一致字体，平面线条示意。三者均采用 custom 叙事、custom 视觉及备选 custom 平面图像语言；本次不采用 AI 图像。选方案 1（零基索引 0），因为用户最重视问题链且已有明确 13 页讲稿。三种方案均承认方法待验证，不把示意当结果。

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 64 px |
| Content Area | 正文通常在 x=64..1216，y=160..624；页脚 y=660..704 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode References**: instructional, pyramid
- **Mode Behavior**: 先解释应用问题和现有能力，再用具体矛盾推导研究问题与方法；每页标题给出该页可核对的判断，主张与证据边界同时可见。
- **Visual style**: custom
- **Visual Style References**: swiss-minimal
- **Visual Style Behavior**: 白色画布、明确对齐和开放留白，深蓝标题；实例选择用蓝色、区域修正用青色，橙色只标冲突和风险。用可编辑的关系图、成对对照和少量表格组织信息，示意与实测明确标注。
- **Theme**: 从重叠个体到可核对输出；第 08—11 页保持“候选—实例—区域—统计”的共同认知结构。
- **Tone**: 正式、准确、克制；不用口号式标题、反问句、制作提示或内部研究管理术语。

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 主体画布 |
| Secondary background | #F1F5F8 | 分组与方法区域 |
| Primary | #172B46 | 标题、核心框架 |
| Accent | #007F82 | 区域修正与采用分支 |
| Secondary accent | #C36D37 | 冲突、风险、待验证 |
| Body text | #25374C | 正文 |
| Secondary text | #596B80 | 注释、来源 |
| Divider | #D4DEE7 | 分隔、细边 |
| Instance selection | #315E9D | 候选与实例对应 |
| Surface | #F7F9FB | 对照页底色 |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | 清晰无衬线、粗体 | Microsoft YaHei | Arial | none |
| Body | 清晰无衬线 | Microsoft YaHei | Arial | none |

- **Title stack**: Microsoft YaHei, Arial
- **Body stack**: Microsoft YaHei, Arial

### Font Size Hierarchy

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

### Deck-wide Direction

- **Hierarchy direction**: 统一采用学术章节标识、技术主题标题和机制说明，正文呈现研究内容，页脚仅保留文献来源或图示名称。
- **Composition tendency**: 大型成对示意、横向演进、分支决策、系统分层按页面任务变化。
- **Cross-page continuity**: 选择为蓝色、修正为青色；第 08—11 页复用处理链语义。
- **Spacing posture**: 正文适合投影，备用页略密，页脚仅放可追溯来源。
- **Spacing anchors**: page margin 64; block gap 24; column gutter 40; corner radius 0; body leading 44.

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |
| P14 | table | comparison_matrix | 已有实验的复用内容和新增验证义务 |
| P15 | table | metric_table | RCMC 历史同协议对照 |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

本次使用可编辑关系图说明方法机制。应用页以案例配置、方法对照和系统评价构成正式验证方案。图示仅表示机制与处理关系，历史数值集中于附录。

## IX. Content Outline

### Part 1: 研究背景

#### Slide 01 - 研究定位
- **Audience move**: 明确论文研究对象及方法与系统的关系。
- **Title**: 面向密集场景的目标检测与实例分割方法及系统研究
- **Core message**: 以密集实例判别与区域优化为研究主线，开展通用系统设计及动物群体场景验证。
- **Relationships**: 实例对应、区域表达与系统应用构成研究对象的三个层次。
- **Composition**: 深蓝封面，完整论文题目与实例关系示意并列，副标题说明研究范围。
- **Content**: 毕业论文研究方案汇报；研究背景、关键问题与技术路线；2026.10.14。

#### Slide 02 - 密集场景的个体级感知需求
- **Audience move**: 理解计数、区域与占用分析对实例级输出的共同要求。
- **Title**: 密集场景的个体级感知需求
- **Core message**: 准确统计与区域分析共同依赖实例区分和像素归属。
- **Relationships**: 计数对应实例集合，个体区域对应掩码，前景占用对应掩码并集。
- **Composition**: 左侧密集实例示意，右侧数量、区域、占用三项需求。
- **Content**: 动物群体、人员与车辆监测的共同需求；遗漏、重复、粘连和区域缺失影响统计。

#### Slide 03 - 目标检测与实例分割的技术进展
- **Audience move**: 了解通用模型的现有能力及密集场景的研究重点。
- **Title**: 目标检测与实例分割的技术进展
- **Core message**: 统一检测分割、端到端预测与概念感知提供研究起点，进一步聚焦密集实例区分和接触边界质量。
- **Relationships**: 代表模型指标与三项现有能力并列，衔接密集场景研究重点。
- **Composition**: 官方指标与技术能力分区展示。
- **Content**: YOLO26m-seg 官方 COCO val2017、640 输入、Box AP 52.5 / Mask AP 44.1；一对一免 NMS 预测；SAM 3（2025）概念驱动多实例感知。

### Part 2: 研究现状与关键问题

#### Slide 04 - 密集目标检测中的候选选择研究
- **Audience move**: 认识候选选择的已有技术基础与具体研究切入点。
- **Title**: 密集目标检测中的候选选择研究
- **Core message**: 候选选择已覆盖密度适应、集合预测与查询去冗余，研究切入点应落实到复杂候选集合中的实例对应歧义。
- **Relationships**: 三类代表方法按发表时间排列，展示不同技术策略，非性能排名。
- **Composition**: 时间序列连接三项代表工作，下方归纳进展与研究切入点。
- **Content**: Adaptive NMS 2019；CrowdDet 2020；DDQ 2023；完整候选、局部片段与邻接个体共存。

#### Slide 05 - 难点一：实例对应歧义与检出完整性
- **Audience move**: 理解相邻实例与同一实例片段带来的选择冲突。
- **Title**: 难点一：实例对应歧义与检出完整性
- **Core message**: 候选重叠程度难以充分区分实例身份，需要兼顾相邻实例保留与重复候选抑制。
- **Relationships**: 不同实例高重叠与同一实例片段低重叠形成相反选择要求。
- **Composition**: 两个并排的实例对应机制图。
- **Content**: 高重叠的邻接实例可能被误抑制；低重叠的局部片段可能重复保留；引出可见完整性和相互关系。

#### Slide 06 - 掩码质量评估与局部精细化研究
- **Audience move**: 区分当前质量估计、局部细化与操作收益预测。
- **Title**: 掩码质量评估与局部精细化研究
- **Core message**: 当前质量评分与具体修正的收益估计具有不同预测目标。
- **Relationships**: 质量估计、局部细化、易错区域处理构成已有研究基础，衔接操作收益预测。
- **Composition**: 左列三项代表研究，右侧强调具体操作的收益估计。
- **Content**: Mask Scoring R-CNN（2019）；Boundary Patch Refinement（2021）；Mask Transfiner（2022）；结合操作响应与邻接信息进行实例级选择。

#### Slide 07 - 难点二：接触边界归属与修正风险
- **Audience move**: 理解修正收益与正确区域退化并存的技术挑战。
- **Title**: 难点二：接触边界归属与修正风险
- **Core message**: 缺少真实标注时，具体修正可能去除错误区域，也可能损伤正确区域。
- **Relationships**: 同类收紧操作在两种像素归属条件下产生相反效果。
- **Composition**: 修复与退化的双案例机制图，下方归纳共同挑战。
- **Content**: 去除邻接实例误分像素与丢失自身真实细节；邻接预测含误差；兼顾净收益和正确区域保护。

### Part 3: 研究目标与技术路线

#### Slide 08 - 密集实例选择与区域优化的总体框架
- **Audience move**: 建立两项研究内容与系统应用之间的整体关系。
- **Title**: 密集实例选择与区域优化的总体框架
- **Core message**: 实例集合优化与掩码区域优化分别形成研究内容，接入统一的统计与展示流程。
- **Relationships**: 基础候选、实例选择、掩码修正与系统统计依次连接。
- **Composition**: 四阶段处理图，下方并列两个研究目标和共同评价基础。
- **Content**: 减少筛选造成的遗漏与重复；提高区域精度并降低退化；失败诊断与分层评价为基础；独立消融和组合对照。

#### Slide 09 - 关系辅助的候选判别与冗余控制
- **Audience move**: 明确候选判别方法的输入、决策、输出与验证目标。
- **Title**: 关系辅助的候选判别与冗余控制
- **Core message**: 拟在限定候选预算内利用质量和关系信息兼顾实例召回与冗余控制。
- **Relationships**: 限定候选提供质量和关系表征，实例判别及集合选择产生保留集合。
- **Composition**: 输入、特征、决策与输出构成横向方法图。
- **Content**: 可见区域完整性；重叠、包含、互补和相邻候选关联；质量特征、关系特征与联合策略对照。

#### Slide 10 - 邻接关系辅助的掩码修正收益预测
- **Audience move**: 明确修正收益预测的信息来源、监督方式与采用分支。
- **Title**: 邻接关系辅助的掩码修正收益预测
- **Core message**: 拟预测具体修正带来的质量变化，并选择采用修正或保留原结果。
- **Relationships**: 原掩码产生候选修正，结合操作响应及邻接信息预测收益，分支输出最终区域。
- **Composition**: 左到右的修正处理链及采用、保留两个结果分支。
- **Content**: 原掩码状态、操作响应、邻接实例关系；标注监督质量变化；固定检测集合开展特征消融；评价精度、修复与退化。

### Part 4: 通用系统设计与应用

#### Slide 11 - 实例感知结果与场景统计的关联
- **Audience move**: 理解不同算法输出如何影响不同场景统计量。
- **Title**: 实例感知结果与场景统计的关联
- **Core message**: 将实例集合、实例掩码与前景并集分别映射到可见数量、个体区域与区域占用。
- **Relationships**: 三种输出分别对应三项统计，个体归属变化不必导致前景占用变化。
- **Composition**: 三列指标定义图，下方说明指标联系和统计对象。
- **Content**: 可见实例计数；个体可见区域；ROI 内掩码并集占比；视频按帧统计。

#### Slide 12 - 通用检测分割系统的总体架构
- **Audience move**: 了解系统模块、接口配置和预期输出。
- **Title**: 通用检测分割系统的总体架构
- **Core message**: 以统一模型接口、可配置类别和统计区域承载方法与场景分析。
- **Relationships**: 数据输入、模型适配、方法处理、展示保存依次组成系统。
- **Composition**: 四模块架构图，下方说明参数、结果追溯与预期输出。
- **Content**: 单图、批量与视频；模型适配；独立模块与组合对比；可视化、逐帧统计、处理配置和报表；系统为设计方案。

#### Slide 13 - 通用系统的应用验证方案
- **Audience move**: 明确猪只案例在通用系统验证中的作用和预期成果。
- **Title**: 通用系统的应用验证方案
- **Core message**: 以猪只群体监测完成应用验证，预期形成通用方法模块与可配置系统。
- **Relationships**: 案例配置支持方法对照，再开展效果与系统评价。
- **Composition**: 上方陈述案例特点，下方三栏组织验证方案，末尾归纳预期成果。
- **Content**: 密集聚集、相似外观与遮挡；已有动物图像数据；基线、单模块与组合；检测分割、统计误差、处理时延和稳定性；跨场景适用性独立评价。

### Part 5: 附录

#### Slide 14 - 现有实验基础与后续研究衔接
- **Audience move**: 区分可复用研究基础与新增方法和系统工作。
- **Title**: 现有实验基础与后续研究衔接
- **Core message**: 复用数据、诊断流程和修正原型，将新增工作集中于方法评价与系统集成。
- **Relationships**: 各项现有工作对应可复用内容及后续研究任务。
- **Composition**: 三列研究基础表及两行现有结论说明。
- **Content**: 实例诊断实验；RCMC 修正原型；动物数据；推理导出组件；已有候选排序方案未达到预设评价要求。
- **Visualization**: reuse-matrix 为研究基础与后续任务对照表。
- **Native-ready**: reuse-matrix=yes

#### Slide 15 - 选择性掩码修正的前期探索
- **Audience move**: 了解修正原型的开发性结果及其证据范围。
- **Title**: 选择性掩码修正的前期探索
- **Core message**: 选择性门控原型在开发性评价中呈现正向结果，同时保留后续独立验证要求。
- **Relationships**: 同一 4500 图协议下三种处理比较；固定平滑与选择性门控的修复和退化数量对应展示。
- **Composition**: 两模型三种处理的指标表，下方实例变化与评价范围。
- **Content**: m 43.38/43.83/43.94，s 39.42/39.80/39.90；32831 个非 crowd 标注；单次拟合，评价集曾参与开发，AP 未估计置信区间；密集子集及独立应用效果需验证。
- **Visualization**: rcmc-metrics 为模型与处理方案的 Mask AP 表。
- **Native-ready**: rcmc-metrics=yes

#### Slide 16 - 研究实施与分阶段验证安排
- **Audience move**: 了解阶段任务和近期工作顺序。
- **Title**: 研究实施与分阶段验证安排
- **Core message**: 从评价方案、独立方法验证、联合评价推进到系统验证。
- **Relationships**: 四个研究阶段有序推进。
- **Composition**: 四行编号阶段，各行给出具体任务，底部陈述近期工作。
- **Content**: 确定评价条件；候选关系与修正收益特征消融；基线和模块组合比较；应用、时延和稳定性评价。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: 与正式修订页面一致的口头讲稿；只讲研究问题、已有进展、拟研究方法和验证范围。引用文献与历史实验身份单独存放在 sources/正式修订版_引用与证据.md。
- **Total duration**: 正文约 12—15 分钟，附录用于问答。
- **Notes style**: 正式、连贯、可直接讲述，不含制作过程或内部管理提示。
- **Presentation purpose**: 说明背景、现状、两项关键问题及研究内容，最终落到通用系统与应用验证。
