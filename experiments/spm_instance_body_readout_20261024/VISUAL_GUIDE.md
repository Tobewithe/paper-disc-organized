# V16 结果与关键技术图

共11组主要图，另保留6组原始全画布技术图；每组都有PNG和同名SVG。图来自真实已完成Run，局部放大和规范XYZ着色只改善阅读，不改变模型、评价或样例选择。主要11张PNG均已本地查看，标题、图例和面板可读。

| 主要图 | 内容及阅读边界 |
|---|---|
| [总体成绩](runs/RUN_SPM_V16_ACCEPTANCE_453b1b1c388b/figures/00_results.png) | 相同正常输出上的Box/Mask/Mask75；框精确保留，Mask退化。 |
| [训练轨迹](runs/RUN_SPM_V16_ACCEPTANCE_453b1b1c388b/figures/01_training.png) | 训练loss与1000/2000/4000全量AP；正式候选预选末4000步，曲线回升不代表最终超过原生。 |
| [修复与损伤](runs/RUN_SPM_V16_ACCEPTANCE_453b1b1c388b/figures/02_repair_damage.png) | 左为官方Mask75 GT迁移32/187；右为2388个固定Box75检测自身IoU差。二者分母不同。 |
| [结构与冻结位置](runs/RUN_SPM_V16_ACCEPTANCE_453b1b1c388b/figures/03_architecture.png) | 冻结原生/身体，实例独立表面聚合、10817参数读出、原生logit残差及真实Mask监督。 |
| [FP/FN权衡](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/figures/04_pixel_tradeoff.png) | 2388配对实例求和的FP−42.7%、FN+97.1%；不是全图去重像素。 |
| [改善例1：GT2059](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/figures/case_00_zoom.png) | 同一检测IoU0.7197→0.8883，FP201→81、FN169→53，两个方向均改善。 |
| [改善例2：GT1662](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/figures/case_01_zoom.png) | IoU0.5328→0.6705，FP269→26但FN9→90，局部收益仍伴随真前景流失。 |
| [损伤例1：GT2459](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/figures/case_02_zoom.png) | IoU0.5982→0，FP1→0、FN87→218，小猪Mask被完全删除。 |
| [损伤例2：GT168](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/figures/case_03_zoom.png) | IoU0.6443→0.1017，FP134→27、FN30→291，删除多余区域未抵偿猪体丢失。 |
| [固定顺序例1：GT1](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/figures/case_04_zoom.png) | IoU0.9200→0.8910，FP204→102、FN194→429，原本较好的轮廓被收缩。 |
| [固定顺序例2：GT2](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/figures/case_05_zoom.png) | IoU0.9392→0.9261，FP283→182、FN80→252。 |

6例均从同一Box75检测账本选择：自身IoU增量最大的前两例、损伤最大的前两例、按GT编号最早且不重复的两例。因此它们展示现象，不是随机样本或总体收益估计。完整选择及身份见[VISUAL_CASES.json](runs/RUN_SPM_V16_RGB_ACCEPTANCE_423e63c0e598/VISUAL_CASES.json)。

每例八面板依次是完整RGB与原生框、预测规范身体XYZ着色、原生Mask放大、修正Mask放大、带符号logit残差、局部F与身体聚合T的差异范数、表面支持量、最终TP绿色/FP红色/FN蓝色。残差红为正、蓝为负；FP/FN面板颜色含义另列。各例残差与差异范数色条独立，不能横图按颜色直接比较幅度。表面支持按同一0–1范围着色。

身体颜色把所预测顶点的规范XYZ映到RGB，显示的是模型输出，真实图没有规范身体GT。相邻颜色连贯、支持高或F/T接近均不自动代表对应正确；白色区域为查询窗口之外。160网格的块状外观是实际空间分辨率，未绘制虚构高分辨率身体对应。放大仅用于显示；全量AP与IoU仍使用原640画布。

原始全图保存在[原验收图目录](runs/RUN_SPM_V16_ACCEPTANCE_453b1b1c388b/figures)的`case_00.png`至`case_05.png`及SVG；原图使用顶点ID色图，修订展示采用规范XYZ颜色。底层6份NPZ在[RGB验收目录](runs/RUN_SPM_V16_RGB_ACCEPTANCE_423e63c0e598)，包含真实RGB、GT/原生/修正Mask、身体顶点、残差、F/T、置信及支持；原始数组未改。

绘图追溯：[独立统计与原图Run](runs/RUN_SPM_V16_ACCEPTANCE_453b1b1c388b/run.json)、[展示Run](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/run.json)、[展示来源SHA](runs/RUN_SPM_V16_PRESENTATION_e57a80103a51/SUMMARY.json)。展示作业无训练、无模型前向、无GPU。
