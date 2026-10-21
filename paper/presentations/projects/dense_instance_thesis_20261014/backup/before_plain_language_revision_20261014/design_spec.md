<!-- ppt-master-schema: design-spec/v1 -->
# 通用密集检测分割毕业论文汇报 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | 通用密集检测分割毕业论文汇报 |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 18 |
| Primary Language | zh-CN |
| Target Audience | 导师与毕业论文汇报评审 |
| Communication Intent | 解释应用需求、业界进展、尚存矛盾和两个攻克方向，说明如何形成通用系统；实验与成果只简要交代 |
| Desired Audience Outcome | 理解帧内错误、遮挡身份恢复和部署约束如何影响应用，并区分两项方法研究与跟踪系统集成 |
| Core Message / Ask / Action | 从可靠帧内观测走向连续个体分析；实例选择与区域修正独立验证，跟踪及系统集成承接应用需求，猪只作为案例 |
| Delivery Context | 明天现场口头汇报，正文约 15—18 分钟；备用页用于问答 |
| Artifact Afterlife | 可编辑 PPTX、逐页讲稿、后续研究汇报复用 |
| Reading Mode | presentation |
| Content Strategy | 按15 页正文问题链提炼，保留事实与限定词，补充三页备用材料 |
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
- **Direction comparison**: 方案 1 为问题链图解：同一处理链逐页聚焦，单一无衬线字体，白底与两种语义色；方案 2 为论文评述：文献证据占较大比重，以宋体标题、微软雅黑正文形成对比，灰白栏式版面，细线平面示意；方案 3 为系统演示：以用户操作顺序切入，深蓝场景页和明亮方法页交替，微软雅黑一致字体，平面线条示意。三者均采用 custom 叙事、custom 视觉及备选 custom 平面图像语言；本次不采用 AI 图像。选方案 1（零基索引 0），因为用户最重视问题链且已有明确的问题链要求。三种方案均承认方法待验证，不把示意当结果。

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
- **Theme**: 从重叠个体到可核对输出；第 10—14 页保持“候选—实例—区域—统计”的共同认知结构。
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
- **Cross-page continuity**: 选择为蓝色、修正为青色；第 10—14 页复用处理链语义。
- **Spacing posture**: 正文适合投影，备用页略密，页脚仅放可追溯来源。
- **Spacing anchors**: page margin 64; block gap 24; column gutter 40; corner radius 0; body leading 44.

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |
| P16 | table | comparison_matrix | 已有实验的复用内容和新增验证义务 |
| P17 | table | metric_table | RCMC 历史同协议对照 |

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

#### Slide 02 - 密集场景的输出失效与应用后果
- **Audience move**: 从具体场景理解计数偏差、区域失真和身份中断的不同成因。
- **Title**: 密集场景的输出失效与应用后果
- **Core message**: 局部遮挡、个体接触和相似外观交叉分别引发检出、区域与身份问题。
- **Relationships**: 场景条件经由输出错误影响应用统计；每条因果链有不同验证对象。
- **Composition**: 三行场景示意、输出错误和应用后果，末行归纳连续个体分析的要求。
- **Content**: 局部遮挡造成间歇漏检，影响可见计数和轨迹；个体接触造成边界错归，影响个体区域；相似目标交叉或重现造成错配，影响身份与停留记录。机制示意，不是本项目实测。

#### Slide 03 - 现有技术能力与连续身份评价
- **Audience move**: 认识当前检测分割与跟踪能力，并理解单帧 AP 不评价跨帧身份。
- **Title**: 现有技术已支持检测、分割与视频跟踪
- **Core message**: 帧内检测分割、低分关联与运动纠偏、概念驱动视频跟踪均有现成方法；连续身份必须独立评价。
- **Relationships**: 官方帧内指标与三个技术层次并列，底部区分定位分割评价与身份关联评价。
- **Composition**: 左侧官方指标，右侧现有能力，底部直接指出应用评价缺口。
- **Content**: YOLO26m-seg 官方 COCO val2017、640，Box AP 52.5 / Mask AP 44.1；ByteTrack / OC-SORT；SAM 3；AP 不定义“达到跟踪级别”的统一门槛。

#### Slide 04 - 遮挡后的重新检出与身份恢复
- **Audience move**: 区分重新发现目标与恢复原身份，理解不可见期间的信息缺失。
- **Title**: 遮挡后重新检出，仍需恢复原有身份
- **Core message**: 目标重现仅提供当前观测；长时无观测、复杂运动与相似外观共同增加跨帧关联歧义。
- **Relationships**: 遮挡前观测、不可见状态预测、重现后的历史身份重关联构成三个阶段。
- **Composition**: 三帧连续机制图，保留同色实例，遮挡期间不伪造可见检测，下方说明恢复条件。
- **Content**: 既有低分框关联、运动纠偏与视频记忆方法已改善部分困难；再次检出和原身份续接需分别评价；机制示意非实验结果。

#### Slide 05 - 连续视频部署中的工程约束
- **Audience move**: 认识论文精度与部署可用性之间的验证差距。
- **Title**: 连续视频部署需要兼顾稳定性与响应时延
- **Core message**: 输出确认、轨迹保留与多路处理具有收益和成本，需要在完整处理链上验证。
- **Relationships**: 三类工程决策分别关联质量收益、等待或资源成本，以及对应测量量。
- **Composition**: 三条横向因果链展示输出确认、轨迹保留、处理规模，底部限定统计对象。
- **Content**: 确认等待减少短暂虚假轨迹但增加响应时延；延长保留有利于重关联但增加资源开销；吞吐、端到端时延及掉帧分别测量；可见计数与预测存量分开。

### Part 2: 研究现状与关键问题

#### Slide 06 - 密集目标检测中的候选选择研究
- **Audience move**: 认识候选选择的已有技术基础与具体研究切入点。
- **Title**: 密集目标检测中的候选选择研究
- **Core message**: 候选选择已覆盖密度适应、集合预测与查询去冗余，研究切入点应落实到复杂候选集合中的实例对应歧义。
- **Relationships**: 三类代表方法按发表时间排列，展示不同技术策略，非性能排名。
- **Composition**: 时间序列连接三项代表工作，下方归纳进展与研究切入点。
- **Content**: Adaptive NMS 2019；CrowdDet 2020；DDQ 2023；完整候选、局部片段与邻接个体共存。

#### Slide 07 - 难点一：实例对应歧义与检出完整性
- **Audience move**: 理解相邻实例与同一实例片段带来的选择冲突。
- **Title**: 难点一：实例对应歧义与检出完整性
- **Core message**: 候选重叠程度难以充分区分实例身份，需要兼顾相邻实例保留与重复候选抑制。
- **Relationships**: 不同实例高重叠与同一实例片段低重叠形成相反选择要求。
- **Composition**: 两个并排的实例对应机制图。
- **Content**: 高重叠的邻接实例可能被误抑制；低重叠的局部片段可能重复保留；引出可见完整性和相互关系。

#### Slide 08 - 掩码质量评估与局部精细化研究
- **Audience move**: 区分当前质量估计、局部细化与操作收益预测。
- **Title**: 掩码质量评估与局部精细化研究
- **Core message**: 当前质量评分与具体修正的收益估计具有不同预测目标。
- **Relationships**: 质量估计、局部细化、易错区域处理构成已有研究基础，衔接操作收益预测。
- **Composition**: 左列三项代表研究，右侧强调具体操作的收益估计。
- **Content**: Mask Scoring R-CNN（2019）；Boundary Patch Refinement（2021）；Mask Transfiner（2022）；结合操作响应与邻接信息进行实例级选择。

#### Slide 09 - 难点二：接触边界归属与修正风险
- **Audience move**: 理解修正收益与正确区域退化并存的技术挑战。
- **Title**: 难点二：接触边界归属与修正风险
- **Core message**: 缺少真实标注时，具体修正可能去除错误区域，也可能损伤正确区域。
- **Relationships**: 同类收紧操作在两种像素归属条件下产生相反效果。
- **Composition**: 修复与退化的双案例机制图，下方归纳共同挑战。
- **Content**: 去除邻接实例误分像素与丢失自身真实细节；邻接预测含误差；兼顾净收益和正确区域保护。

### Part 3: 研究目标与技术路线

#### Slide 10 - 密集实例选择与区域优化的总体框架
- **Audience move**: 建立两项研究内容与系统应用之间的整体关系。
- **Title**: 密集实例选择与区域优化的总体框架
- **Core message**: 两项空间方法分别验证，系统接入跟踪及重关联，开展连续身份与应用统计评价。
- **Relationships**: 基础候选、实例选择、掩码修正与系统关联分析衔接；具体输入依接口分别消费。
- **Composition**: 四阶段处理图，下方并列两个研究目标和共同评价基础。
- **Content**: 减少筛选造成的遗漏与重复；提高区域精度并降低退化；失败诊断与分层评价为基础；跟踪集成不作为已验证的新算法贡献。

#### Slide 11 - 关系辅助的候选判别与冗余控制
- **Audience move**: 明确候选判别方法的输入、决策、输出与验证目标。
- **Title**: 关系辅助的候选判别与冗余控制
- **Core message**: 拟在限定候选预算内利用质量和关系信息兼顾实例召回与冗余控制。
- **Relationships**: 限定候选提供质量和关系表征，实例判别及集合选择产生保留集合。
- **Composition**: 输入、特征、决策与输出构成横向方法图。
- **Content**: 可见区域完整性；重叠、包含、互补和相邻候选关联；质量特征、关系特征与联合策略对照。

#### Slide 12 - 邻接关系辅助的掩码修正收益预测
- **Audience move**: 明确修正收益预测的信息来源、监督方式与采用分支。
- **Title**: 邻接关系辅助的掩码修正收益预测
- **Core message**: 拟预测具体修正带来的质量变化，并选择采用修正或保留原结果。
- **Relationships**: 原掩码产生候选修正，结合操作响应及邻接信息预测收益，分支输出最终区域。
- **Composition**: 左到右的修正处理链及采用、保留两个结果分支。
- **Content**: 原掩码状态、操作响应、邻接实例关系；标注监督质量变化；固定检测集合开展特征消融；评价精度、修复与退化。

### Part 4: 通用系统设计与应用

#### Slide 13 - 方法输出与连续应用的验证关系
- **Audience move**: 识别两项方法对跟踪和区域统计的不同作用路径。
- **Title**: 方法改进需要沿实际接口验证应用收益
- **Core message**: 候选集合经固定跟踪器评价身份，掩码经区域统计评价像素效用；两条路径分别验证。
- **Relationships**: 框与分数驱动标准 ByteTrack，掩码进入区域统计；跟踪指标与掩码指标不互相替代。
- **Composition**: 上下两条处理链，底部列帧内、跨帧及遮挡事件评价。
- **Content**: Box/Mask AP；HOTA/AssA、IDF1、ID 切换；遮挡后正确续接、误连及未恢复；固定框和分数的掩码修正不能直接改变标准 ByteTrack 的关联。

#### Slide 14 - 通用检测分割与跟踪分析系统
- **Audience move**: 了解系统模块、接口配置和预期输出。
- **Title**: 通用检测分割与跟踪分析系统
- **Core message**: 统一输入与模型接口，接入两项空间方法及既有跟踪器，输出有明确状态的连续分析结果。
- **Relationships**: 输入适配、检测分割、空间优化、身份关联、输出分析形成系统；图像任务可绕过关联。
- **Composition**: 五模块系统架构，下方分开当前观测、短时丢失和重关联状态。
- **Content**: 图像与视频、时间戳和场景配置；框分数与掩码统一格式；跟踪及重关联；轨迹、区域与统计导出；可见计数与预测存量分开。为拟建系统。

#### Slide 15 - 通用系统的应用验证方案
- **Audience move**: 明确猪只案例在通用系统验证中的作用和预期成果。
- **Title**: 通用系统的应用验证方案
- **Core message**: 猪只群体作为相似外观与遮挡案例；复用静态基础，新增连续身份标注与遮挡事件评价。
- **Relationships**: 案例配置支持方法对照，再开展效果与系统评价。
- **Composition**: 上方陈述案例特点，下方三栏组织验证方案，末尾归纳预期成果。
- **Content**: 已有图像可复用，视频序列、跨帧 ID 和遮挡重现事件需新增标注；按视频分组；固定跟踪器比较基线与空间方法；帧内、身份、恢复事件和部署评价；跨场景适用性需独立验证。

### Part 5: 附录

#### Slide 16 - 现有实验基础与后续研究衔接
- **Audience move**: 区分可复用研究基础与新增方法和系统工作。
- **Title**: 现有实验基础与后续研究衔接
- **Core message**: 复用数据、诊断流程和修正原型，将新增工作集中于方法评价与系统集成。
- **Relationships**: 各项现有工作对应可复用内容及后续研究任务。
- **Composition**: 三列研究基础表及两行现有结论说明。
- **Content**: 实例诊断实验；RCMC 修正原型；动物静态数据；推理导出组件；新增视频 ID 与遮挡事件标注、跟踪接入和系统测试；旧排序方案未达到预设评价要求。
- **Visualization**: reuse-matrix 为研究基础与后续任务对照表。
- **Native-ready**: reuse-matrix=yes

#### Slide 17 - 选择性掩码修正的前期探索
- **Audience move**: 了解修正原型的开发性结果及其证据范围。
- **Title**: 选择性掩码修正的前期探索
- **Core message**: 选择性门控原型在开发性评价中呈现正向结果，同时保留后续独立验证要求。
- **Relationships**: 同一 4500 图协议下三种处理比较；固定平滑与选择性门控的修复和退化数量对应展示。
- **Composition**: 两模型三种处理的指标表，下方实例变化与评价范围。
- **Content**: m 43.38/43.83/43.94，s 39.42/39.80/39.90；32831 个非 crowd 标注；单次拟合，评价集曾参与开发，AP 未估计置信区间；密集子集及独立应用效果需验证。
- **Visualization**: rcmc-metrics 为模型与处理方案的 Mask AP 表。
- **Native-ready**: rcmc-metrics=yes

#### Slide 18 - 研究实施与分阶段验证安排
- **Audience move**: 了解阶段任务和近期工作顺序。
- **Title**: 研究实施与分阶段验证安排
- **Core message**: 从评价方案、独立方法验证、联合评价推进到系统验证。
- **Relationships**: 四个研究阶段有序推进。
- **Composition**: 四行编号阶段，各行给出具体任务，底部陈述近期工作。
- **Content**: 明确序列身份与遮挡事件标注及评价条件；两项空间方法独立消融；固定跟踪器开展组合及身份评价；应用、时延、掉帧与连续稳定性评价。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: 与应用与视频强化页面一致的正式口头讲稿；说明研究进展、具体失效、两项空间方法及跟踪系统评价边界。新增证据见 sources/dense_video_reliability_research_20261014.facts.json 与应用与视频强化版_引用与证据.md；历史实验沿用正式修订版证据。
- **Total duration**: 正文约 15—18 分钟，附录用于问答。
- **Notes style**: 正式、连贯、可直接讲述，不含制作过程或内部管理提示。
- **Presentation purpose**: 说明背景、现状、两项关键问题及研究内容，最终落到通用系统与应用验证。
