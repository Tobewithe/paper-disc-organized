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
- **Tone**: 准确、克制、适合毕业论文方案汇报。

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

- **Hierarchy direction**: 先读论点，再读示意或证据，再读作用范围。
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

本次页面使用关系示意和可编辑图形表达机制，无必须依靠照片确认的对象身份。猪只案例页展示拟实现的交互流程，明确标注设计示意，未采用合成实验图片或假结果。历史真实案例图保留在原报告中供追问。

## IX. Content Outline

### Part 1: 需求与现状

#### Slide 01 - 密集场景中的每个个体都应被看清

- **Audience move**: 从猪只专用课题印象 → 通用密集检测分割问题。
- **Relationships**: 实例对应与区域表达共同支撑场景统计。
- **Composition**: 完整题目配大型重叠区域示意。
- **Title**: 面向密集场景的目标检测与实例分割方法及系统研究
- **Core message**: 研究从实例对应到区域质量再到系统应用的完整链路。
- **Content**: 通用方法；两类输出决策；猪只作为应用案例；毕业论文研究方案汇报。
- **Cover impact**: 看清每个个体，才能用好场景数据；大标题与重叠实例的矛盾并列。

#### Slide 02 - 应用需要数量准确，也需要区域可信

- **Audience move**: 从识别类别 → 理解个体级输出需求。
- **Relationships**: 计数依赖实例对应，个体区域依赖像素归属，占用依赖掩码并集。
- **Title**: 应用需要数量准确，也需要区域可信
- **Core message**: 不同类别共享可见数量、个体区域和前景占用三种输出需求。
- **Content**: 动物群体、人员、车辆作为需求背景；漏检少计、重复多计、像素错归影响区域。

#### Slide 03 - 基础模型已具备强能力，密集输出仍需逐实例核对

- **Audience move**: 从泛泛困难 → 知道当前起点与未满足要求。
- **Relationships**: 类别位置区域、免 NMS 输出、概念感知是已有能力；密集实例对应与边界可靠性是进一步要求。
- **Title**: 基础模型已具备强能力，密集输出仍需逐实例核对
- **Core message**: 在成熟基础能力之上研究输出可靠性。
- **Content**: YOLO26m-seg 官方 COCO 640 Box AP 52.5 / Mask AP 44.1；已有端到端免 NMS 分支；SAM 3（2025）概念驱动多实例检测分割与视频跟踪。官方值非本地复现。

#### Slide 04 - 实例选择已从固定抑制走向集合建模

- **Audience move**: 从“改一下去重” → 理解候选竞争已有充分研究。
- **Relationships**: 密度适应、集合预测和区分查询展示不同研究层次；不能把时间演进当通用性能排名。
- **Title**: 实例选择已从固定抑制走向集合建模
- **Core message**: 研究切口必须具体到完整候选、局部片段与邻接个体的歧义。
- **Content**: Adaptive NMS 2019；CrowdDet 2020；DDQ 2023；TQS 2026 预印本固定候选关系选择，跨域稳定性有限。

### Part 2: 两类难点

#### Slide 05 - 高重叠不等于重复，低重叠也不等于不同个体

- **Audience move**: 从阈值问题 → 实例身份歧义问题。
- **Relationships**: 相邻个体高重叠与同一目标片段低重叠形成相反判断；候选质量与相互关系共同决定保留。
- **Title**: 重叠程度无法直接回答“是不是同一个体”
- **Core message**: 保留邻居与删除重复需要同时兼顾。
- **Content**: 两种问题示意；限定已有候选池和计算预算；仅选择不能生成缺失预测。

#### Slide 06 - 质量估计与边界细化已有基础，修正收益仍要单独判断

- **Audience move**: 从笼统优化边界 → 区分当前质量、错误位置和操作收益。
- **Relationships**: 质量评分、局部细化和易错节点处理提供信息，但不直接保证某次操作产生正收益。
- **Title**: “当前有多好”与“这次改得值不值”是两种判断
- **Core message**: 已有精细修正能力之上，需要判断具体操作的收益。
- **Content**: Mask Scoring R-CNN 2019；BPR 2021；Mask Transfiner 2022；不声称修正收益此前无人研究。

#### Slide 07 - 同一次修正可能去掉错误，也可能切掉正确区域

- **Audience move**: 从平均提升 → 理解修复与误伤并存。
- **Relationships**: 删除邻居误报与删除自身正确细节对同一收紧操作产生相反作用。
- **Title**: 同一次修正，可能修好，也可能误伤
- **Core message**: 接触位置像素归属不确定，需要按实例判断采用与否。
- **Content**: 修复与误伤双案例为机制示意，不编造 IoU；推理无 GT；邻居预测也可能错误；同时检验总体质量和成功保护。

### Part 3: 攻克方向

#### Slide 08 - 两个层次，两种决策，共同支撑应用

- **Audience move**: 从两类孤立问题 → 理解统一研究框架。
- **Relationships**: 基础候选先被选择成实例集合，再进行选择性区域修正，最终用于统计。
- **Title**: 先判断保留谁，再判断怎样修正
- **Core message**: 方向一处理实例集合，方向二处理已选实例的区域。
- **Content**: 候选集合 → 实例选择 → 掩码修正 → 可核对的统计；诊断与分层评价作为基础。

#### Slide 09 - 攻克方向一：密集实例的候选判别与冗余控制

- **Audience move**: 从研究问题 → 明确方向一输入、决策和输出。
- **Relationships**: 候选质量、可见完整性和候选关系提供信息，联合选择产生保留集合。
- **Title**: 方向一：结合候选质量与关系，选择不同实例
- **Core message**: 在固定候选预算内降低筛选丢失与冗余。
- **Content**: 限定候选 → 重叠/包含/互补与完整性 → 质量和关系联合判断 → 保留集合；身份分离，新增方法待验证。

#### Slide 10 - 攻克方向二：邻接关系辅助的掩码修正收益预测

- **Audience move**: 从统一修正 → 明确按操作收益选择。
- **Relationships**: 原掩码生成候选修正，响应和邻居信息预测收益，分支采用修正或保留原结果。
- **Title**: 方向二：先试修正，再判断是否采用
- **Core message**: 预测具体操作收益，同时控制原成功实例退化。
- **Content**: 原掩码、修正响应、邻居预测；训练可用 GT 监督，推理仅图像与预测；单独评价固定框类别分数检测集合；邻居信息新增增益待验证。

### Part 4: 系统落点

#### Slide 11 - 输出变化必须对应到具体应用信息

- **Audience move**: 从模型指标 → 认识数量与区域的不同来源。
- **Relationships**: 实例选择影响可见数量；区域修正影响个体区域；前景并集变化才影响占用。
- **Title**: 把算法输出转化为可核对的数量与区域
- **Core message**: 每项统计都能追溯到对应实例或像素。
- **Content**: 可见计数；个体可见区域；ROI 掩码并集占比；像素仅在邻居间重新归属时占比可不变；已有诊断工具、RCMC 原型、动物数据可用。

#### Slide 12 - 通用检测分割分析平台

- **Audience move**: 从离线方法 → 形成明确系统实现范围。
- **Relationships**: 输入、模型适配、方法处理、展示统计和结果保存构成处理链；类别模型和区域均可配置。
- **Title**: 系统把模型、方法与统计串成统一流程
- **Core message**: 通用性首先落实为模型接口、类别与区域配置。
- **Content**: 单图/批量/视频；适配模型；基线/选择/修正/组合；框掩码叠加；数量与占用；导出图片视频配置报表。拟实现，不等于已完成。

#### Slide 13 - 猪只作为应用案例，验证完整使用流程

- **Audience move**: 从专用物种项目 → 理解可配置系统的一个实例。
- **Relationships**: 选择模型、输入数据、对比结果、设置 ROI、导出统计构成案例流程。
- **Title**: 猪只群体监测是通用系统的一种配置
- **Core message**: 最终形成可检查、可比较、可导出的密集感知系统。
- **Content**: 界面设计示意无假实测；检验功能、统计误差、端到端延迟与连续任务稳定性；数量为可见实例，曲线为逐帧统计，不默认跨帧身份保持。
- **Closing impact**: 方法回答“保留谁、怎样修”，系统回答“怎样用”；界面示意旁收束全文。

### Part 5: 备用材料

#### Slide 14 - 实验复用边界

- **Audience move**: 从担忧从零开始 → 明确复用与新增任务。
- **Relationships**: 每项已有基础对应复用内容与新增验证，旧结论不迁移成新贡献。
- **Title**: 已有基础可以复用，新方法结论需要重新验证
- **Core message**: 复用工具、缓存与原型，保留各自实验身份。
- **Content**: 密集诊断与候选追踪；RCMC 原型；动物案例数据；推理导出组件；各自标明需补内容；旧候选排序 Gate 未通过。
- **Visualization**: reuse-matrix 为已有基础 × 复用/新增的文字对照表。
- **Native-ready**: reuse-matrix=yes

#### Slide 15 - RCMC 历史证据

- **Audience move**: 从方法设想 → 知道原型已有有限范围证据。
- **Relationships**: 同一 4500 图旧协议下三种处理比较；历史结果与新增邻居关系方案分开。
- **Title**: 选择性修正已有原型，新关系信息的增益仍待验证
- **Core message**: RCMC 可作工程起点，历史证据不能替代新方法评价。
- **Content**: m 43.38/43.83/43.94；s 39.42/39.80/39.90；4500 图 32831 普通 GT；一次门控拟合，旧验证参与探索，AP 未给置信区间；修复和误伤均下降。
- **Visualization**: rcmc-metrics 为模型 × 三种处理的 Mask AP 表。
- **Native-ready**: rcmc-metrics=yes

#### Slide 16 - 实施与验证顺序

- **Audience move**: 从研究范围 → 掌握可执行的下一步。
- **Relationships**: 固定评价条件、分别检验两个方向、组合检验和系统应用顺序推进。
- **Title**: 从可复用原型出发，逐层验证到系统
- **Core message**: 先固定比较口径并接入基线，再独立检验两项方法。
- **Content**: 固定样本分组和预算；关系信息对照；收益模型邻居特征消融；基线/两个单模块/组合；猪只应用确认与端到端系统测试。与独立期刊系数纠错线分开，不启动训练。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: 以原汇报稿各页口播为基础，按实际页面改写；补齐来源链接及限定词。所有文献和历史实验来源来自已核验的原汇报稿。备用页保留 Run 身份，不把示意当实测。
- **Total duration**: 正文 12—15 分钟，备用页不计入
- **Notes style**: 自然、准确、可直接口播
- **Presentation purpose**: 解释问题链与技术切口，说明通用系统落点和可复用基础。
