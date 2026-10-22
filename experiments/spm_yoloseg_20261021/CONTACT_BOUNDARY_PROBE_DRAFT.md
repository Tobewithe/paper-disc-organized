# SPM-YOLO：冻结 P3/P4 的同类实例接触边界可读性 Probe（草案，未冻结）

**日期 2026-10-22 / 研究线 \`spm_yoloseg\` / 仅设计 / 0 新增训练步与 0 新 Run。**  
**路线：** [V3 优先级与风险树](../../paper/manuscripts/spm_yoloseg/PRIORITIES_RISKS_AND_DECISION_TREE_V3.md)  
**用途：** 回答「在真实同类猪体接触处，原生 YOLO26-Seg 的中间表征是否存在可被轻量探针读取的**区分两个实例边界 vs 猪-背景边界**的信息？」；据此判断优先研究外观记忆、接触几何信息，还是更高分辨率/多邻域表征。**该实验不检验完整 SPM-YOLO 真实 AP，也不自动确认原型机制成立。**

## 1. 科学命题

**H-REP**：冻结的 P3（及可对照的 P4）局部特征在相同真实视频来源/尺寸/照明约束下，包含超过低成本像素梯度/普通边缘特征的**同类实例接触**判别信息，且该信息可跨保留视频/猪栏稳定读取。

**H-alt-1 外观线索不足**：猪-猪接触内部类似同一头猪的皮肤，P3 及 P4 接触边界特征不可稳定判别。  
**H-alt-2 位置/评价伪捷径**：probe 仅利用背景比例、图像固定位置或实例规模，而不是区分同类接触的线索。  
**H-alt-3 特征容量/尺度混杂**：线性不可读不等于信息不存在；允许小 3×3 邻域非线性 probe 与 P4/可用 P2 的受限对照；只检验当前受限模型可读性。  
**H-alt-4 模型原生后处理/Mask 解码瓶颈**：即便 P3 可读 C 边界，最终 Mask AP 未必改善，尚需独立 B*/G/F 训练与评价。

## 2. 数据合同 / 不能混淆的 GT 类型

1. **train/dev/test 严格按 \`farm_id, pen_id, video_id, camera_id, time segment\` 分组，先分组后抽帧**。训练 probe 和门槛选择只使用真实 train、真实 dev；最终独立 test 在这轮探索严格封存。明确 \`dataset_manifest_sha\`、可见 instance-id mask 校验、来源与授权。无可信可见 Mask 的帧只用于额外未标注可视化，不训练/评估监督 probe。
2. 从 GT **可见实例 ID map** 创建「局部不同实例边界」标签。避免用仅有 bbox、伪实例与可见区域直接估计 amodal occlusion ratio。每张图在原分辨率建立：  
   - \`C\` pig↔pig: 像素邻域存在**不同两头可见猪的 mask ID**边界；主任务正例，排除与背景三叉接触的模糊位置或另设 ignore。  
   - \`B\` pig↔background: 可见猪与背景接壤，但邻域没有两个猪 ID 的边界；主任务困难负例。  
   - \`I\` pig interior: 与任何可见猪/背景实例边界相隔预冻结 r 像素；辅助负例。  
   - \`X\` ignore: 有 GT 疑似漏标、极小/完全遮住目标、画面裁剪边缘、栏杆严重遮挡、三叉接点、分类不确定或重标注冲突。
3. 将原生视频图像通过**同一固定输入预处理/letterbox**映射到 YOLO 640 输入像素；可见 Mask 使用最近邻几何变换。按 **相同输入像素半径**（初始例 4/8/12 像素，实际训练前只冻结一套）关联 P3/P4 网格 token，而不是各尺度用不同物理视野。标记输出位置可能跨多个不同实例感受野；这个标签是**局部几何规则**，不是声称 Neck 自身感受野只看一个网格。
4. 至少记录 \`gt_mask_revision, cluster_key, frame_sha, pig_ids, boundary_type, feature_level, feature_xy, fg_fraction, native_pixel_scale, region_radius, ignore_reason\`。以 train 中多个视频/栏为独立样本支持；同视频不能在 probe train 和 dev 上同时计入。

## 3. 具体探针矩阵：不改原生 YOLO

固定**唯一明确的 Ultralytics commit + checkpoint sha + inference preprocessing**，eval mode 下提取原生 P3/P4 feature，\`requires_grad=False\`；不要把 base 模型梯度、原生 Head/Proto26 或 TEST GT 混入探针。

| 探针 | 输入 | 判定问题 |
|---|---|---|
| \`E0\` 简单边缘控制 | 输入亮度 gradient/Sobel/局部颜色对比、坐标/尺度的简单受限特征 | 单靠像素边界/位置是否足够？ |
| \`E1\` P3-1×1 linear | frozen P3 单元向量 | 接触信号是否线性可读？ |
| \`E2\` P3-3×3 小 CNN/MLP | frozen P3 邻域 | 有局部几何上下文后是否可读？ |
| \`E3\` P4-1×1（及匹配邻域） | frozen P4；对齐标签的空间半径 | 分辨率/尺度对照 |
| \`E4\` 额外 P2 或更早层（有合法接口时可选） | frozen 对应层 | 若 P3 太粗，是否有细节上游？ |
| \`E5\` 标签置乱/跨视频错误配对（按需） | 同输入、打乱训练标签 | 防止采样或基准路径泄漏 |

- 共同输出：\`binary(C vs B)\` 为**关键任务**，\`C/B/I\` 多分类仅作为支持；需要 B 与 C 在相机、区域/实例尺度、局部 fg_fraction 和图像位置 bins 充分匹配，避免 \`pig vs background\` 过于容易而制造虚假的「接触识别」。
- **匹配条件**：不同尺度比较要尽可能保持等量 train 视频、边界 token 数、相同局部输入像素窗口定义和有效训练 step；非完全匹配的输入通道/感受野/参数量需单独披露。
- **训练限制**：只有轻量 probe 参与优化；较强 probe 有效可说明可读的非线性信息，不能直接说明现有 Segment26 会调用它。原模型保持完全冻结，日志检查全部模型参数 SHA。

## 4. 正常评价与结果类型

**关键读数：** 在 held-out 真实 dev 视频/猪栏上的 \`C vs B\` Average Precision (AUPRC)；阳性比例、预测置信度校准和混淆矩阵必须同时报告。相同标签/位置的简单边缘控制 E0 与 P3 E1/E2 的差异是判别核心，不仅报告 \`pig vs bg\` accuracy。  
**辅项：** \`C/B/I\` 宏平均 F1，按密集度/实例面积/照明/压缩质量/视频分组的差异；可选择阈值的 F1 **仅在 dev** 调整，最终 test 不参与；时间/显存/训练 token 量。  
**统计：** 同一视频/猪栏内 token 高度相关，按独立视频或猪栏**簇重抽样**，报告簇数、例数、预定区间方法；如果簇不足，结果 \`INCONCLUSIVE\`，不以大量像素 token 冒充独立样本。对训练/开发中摄像头重叠与跨相机泛化分别标识。

**关键异常 QA**：局部不同实例 ID 直接相接可能对应无遮挡边，也可能只是 GT 主观切割；随机复查人工标注一批 C、B 位置（尤其接触无明显强梯度），一旦大量边界标签不可信需修 GT/规则再跑。不以阴影、栏杆或过强 Sobel 差异充当新机制收益。

## 5. 三种实验结果如何改变原型路线（阈值和置信边界需后续冻结）

| 结果 | 支持什么 / 不支持什么 | 唯一下一个科学重点 |
|---|---|---|
| **E1/E2 对 C vs B 在跨视频稳定高于 E0** | frozen P3 有受限 probe 可读、超出简单边缘的同类接触线索；不能证明 memory 或 AP 将受益 | 定向建立真实 train 接触区域 token 的 prototype coverage，并在合成条件中加强 geometry/contact sampling |
| **E1 弱但 E2 明显较强，或 P4/E4 更强** | 存在非线性/更合适尺度的有限可读线索；线性 Top-k 原型可能难以捕捉 | 优先检验局部上下文 query 或更合适的尺度，暂缓扩大 K |
| **E0/E1/E2/E3 均无可靠增益，且数据质量与预算成立** | 当前受限表示/受限 probe 尚未提供可用的 C 边界证据；**不推断数学上不存在信息** | 优先局部几何 cue / 高分辨率 / 区域关系的零训练诊断，暂停大规模纯外观原型 |

**不管 probe 结果如何，都不能直接宣布 SPM-YOLO 的任务 AP 效益。** 若数据/标注分组不合法、有效视频簇太少、模型/特征版本混乱，先修 P0 再谈结果，状态为 \`INVALID\`/\`INCONCLUSIVE\`。

## 6. 最小实现接口（拟议脚本名，未创建代码）

    scripts/spm_yolo/00_audit_real.py
    scripts/spm_yolo/00b_make_contact_boundary_ledger.py
    scripts/spm_yolo/00c_extract_frozen_neck_features.py
    scripts/spm_yolo/00d_train_boundary_probe.py
    scripts/spm_yolo/00e_eval_boundary_probe.py

产物：\`contact_boundary_ledger.jsonl\`、\`boundary_gt_qa.html\`、\`feature_manifest.json\`（model/code/input hashes）、\`probe_config.yml\`、\`probe_scores_by_cluster.csv\`、\`probe_report.md\`、\`artifact_manifest.json\`。执行需要独立冻结预算与真实 run.json；在本次提交中，这些文件**不存在**。

## 7. 当前下一步与研究范围

**现在只做 P0**：验证独立真实数据簇、可见 Mask 合法性、原生模型 P3/P4 taps 与 checkpoint，形成可审计边界 C/B/I 标签集合。**随后按授权与冻结协议进行 E0–E3 小规模 probe**；此阶段不需要先有 Blender 3000 张/完整原型库，也不修改另一个研究线的 Active Plan。

在每一张图评价最终实例分割时继续沿用原研究线规范：COCOeval Box/Mask 各自匹配；单条预测诊断必须固定同一真实输出身份，不能将不同实例的 Box 和 Mask 成功拼接。
