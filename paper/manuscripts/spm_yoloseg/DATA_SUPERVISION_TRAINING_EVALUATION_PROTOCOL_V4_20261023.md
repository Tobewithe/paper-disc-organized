# SPM-YOLO V4：数据准备、监督目标、分阶段融合与评价协议

**日期：** 2026-10-23  
**研究线：** spm_yoloseg  
**状态：** research_draft / hypothesis_only / not_started；**仅为候选实验协议，没有执行数据处理、渲染、训练或评测，也未授权任何 GPU Run。**  
**遵循：** [主线 README](README.md)、[研究规则](RESEARCH_RULES.md)、[V3 科学决策树](PRIORITIES_RISKS_AND_DECISION_TREE_V3.md)、[V2 实施蓝图](IMPLEMENTATION_BLUEPRINT_V2.md)、[V2 消融方案](../../../experiments/spm_yoloseg_20261021/PILOT_AND_ABLATION_V2.md)。本文件只补充可操作的数据/监督/训练/评价合同，不覆盖 V2/V3 或已冻结协议。

## 0. 一句话决策与研究边界

**路线：** P0 只读审计 → 冻结 YOLO P3/P4 同类猪接触边界 Probe（训练须另获授权）→ 数据与标签 smoke → 合成/真实数据效益 B* → 独立原型学习及真实域适配 → 冻结原型库 → P3 零初始化残差融合 → 等容量对照与机制干预 → 真实独立测试。

第一版只研究**可见实例分割**：SPM 原型是跨图像持久记忆，不是 YOLO26-Seg 原生的逐图 Proto26 mask basis；保留原生 mask 系数维度、解码、loss 和后处理。SPM 不承担个体身份 ReID 或 amodal 分割。**K=32、d=128 是固定 K 的小规模 pilot 参数，不撤销 V2/V3 中的自适应 K / 合并剔除方案；模型/预算需要根据 P0 和 Pilot 冻结。**

**首要科学风险：** 猪-猪接触两侧属于同类，单纯学习猪体前景/背景原型不足以区分实例。必须优先判定当前 P3/P4 是否含有可泛化的接触边界线索；probe 若失败，暂停大规模合成、记忆库和融合训练，改查更高分辨率或显式几何表示。

## 1. P0 数据、模型与评价审计（当前唯一可立即执行的只读工作）

1. 逐帧登记：farm_id、pen_id、camera_id、video_id、timestamp/frame_id、image_sha256、RGB 分辨率、codec、标注修订号、annotation_source、license、split。按猪栏/视频/连续采集事件**先分簇再抽帧**，不得让邻近帧跨 train/dev/test。
2. 记录可见实例 mask 是否完整、漏标、接触模糊、边缘腐蚀、重叠标注及 ignore 区域；目标为 visible mask，不能从 visible mask 推断真实遮挡率或 amodal 轮廓。
3. 锁定 YOLO26-Seg 的模型规格、Ultralytics 源码 commit、checkpoint SHA、input 640、P3/P4 tap、one-to-one/one-to-many 行为、原生 Proto26/Segment26、Mask decode、NMS 与 COCOeval 口径。
4. 形成真实错误账本：小目标、重遮挡、同类接触、两猪合并、同猪分裂、漏检、遮挡/模糊等；样本分母和视频簇公开记录。
5. 验证原有评估规则：Box 与 Mask AP 分开匹配；诊断同一条预测时必须保持预测实例身份（box、class、score、mask）一致；raw 候选在评分过滤和 NMS 前固化。不能拿不同预测拼成“同一成功实例”。

**交付：** split_manifest.json、real_domain_profile.json、data_audit.md、error_ledger.csv、model_eval_manifest.json 和相应 SHA。P0 未通过，不进入正式训练。

## 2. 真实数据与 crop 准备

### 2.1 数据集合

- **Real-train：** 用于原型训练、texture/patch donor、合成分布拟合、YOLO 训练。
- **Real-dev：** 用于算法选择、温度/阈值/原型容量调整；不得作为 memory bank 或纹理 donor。
- **Real-test：** 独立视频/猪栏/相机簇，只进行预冻结的最终评价；所有训练和数据统计不得使用其 RGB、mask、频谱或 embedding。
- **Cross-domain-test（可得时）：** 另一猪栏/相机/日期，报告确切泛化范围；单猪场数据不能宣称跨猪场泛化。
- **Public-train：** 有合法可用实例标注的公开猪数据，来源/许可/领域分别记录；没有 Mask GT 的数据只能贡献无监督特征或 Box 任务，不能作为 Mask 真值。

建议首个 smoke 使用 400–800 张**真实独立抽样帧**、约 3,000–8,000 个实例 crop、至少约 1,000 个接触/遮挡 crop；仅为规模量级提案，P0 清点后重订。必须同时报告独立帧、实例、合成图、增强视图的数量，禁止把增强视图当作独立标注样本。

### 2.2 每个实例生成三种视图

| 样本 | 构造 | 目标 |
|---|---|---|
| tight crop | 按 visible mask 裁剪、少量 padding、背景弱化 | 猪体纹理、外观与局部形态 |
| context crop | GT box 外扩约 1.5–2.0 倍，保留真实邻域 | 邻猪、接触、地面/栏杆与成像上下文 |
| contact pair / multi-instance crop | 至少两实例接近或投影接触，保留每个可见实例 mask | 同类接触、遮挡、边界归属；作为重点 pilot 样本 |

每个裁剪保存 crop↔原图坐标变换、真实 mask、背景有效区、source_id、split 和变换 seed；颜色、JPEG、模糊、尺度、轻微平移、有限遮挡等多视图增强须保存可逆几何，**仅在重叠有效区域计算一致性**。禁止通过 crop 的 GT mask 在推理输入中泄漏实例位置。

### 2.3 前景、接触边界与 ignore 的明确标签

在 full-resolution 可见 GT 上定义：
- **I（interior）：** 与该实例边缘有安全距离的猪体内侧。
- **B（pig-background）：** 猪体与可见背景接壤的边界带。
- **C（pig-pig contact）：** 两个**不同实例 ID** 可见边界在预设像素距离内相邻的区域。它是基于 mask 的**接触邻近代理标签**，不自动等于真实物理接触或前后遮挡真值。
- **D（ignore）：** 缺失标注、极模糊边界、三叉边界、人工不可裁决接触、无法可靠区分的孔洞。

正负样本比例与边界带宽必须按原图分辨率或 resize 后像素冻结；不同相机/分辨率需记录缩放系数。下采样到 P3 后同时保存 soft coverage 与 ignore 权重，不能把 1–2 px 接触线直接 nearest-neighbor 下采样后消失。类别 I/B/C 只用于可读性 probe、辅助监督及评价，不是预设三类 prototype 的语义名称。

## 3. 第一项训练判别：冻结 P3/P4 接触边界 Probe

**需在 P0 通过并单独获训练授权后执行；现在尚未运行。**

- 固定原生 YOLO checkpoint，提取 real-train P3/P4（源、预处理、精度及 SHA 固定）。
- 对 I/B/C 有效 token 训练线性 probe 与小 MLP probe；D 区域 ignore。另设**RGB/灰度梯度、普通边缘、前景/背景二分类**对照，排除“只是找物体边缘”的解释。
- dev 按 held-out video / pen 分簇报告 C-vs-B、C-vs-I macro F1、PR-AUC、召回/精度、置信区间以及小目标/模糊/低对比/相机分层。
- 禁止依赖 GT mask 实例编号作为 probe 推理输入；仅以 GT 构造离线监督与评价标签。
- 若 C 线索不可稳定读取或对照同效，先检查更高空间分辨率、局部几何线索、标签质量；不直接扩大原型 K 或 Blender 数量。

## 4. 合成数据：先真实 copy-paste smoke，Blender 按 Gate 扩展

### 4.1 真实 cutout + 真实背景（低成本数据 pilot）

只从 Real-train 中提取真实猪 visible cutout、背景、噪声与光照模板。按 real_domain_profile 的尺度/密度/相机/位置统计安排多猪采样并控制实例前后覆盖顺序，得到 RGB、visible instance masks、boxes、scene_id 和合成谱系。

建议场景配额（可重叠，须按数据审计修订）：稀疏 30%、双猪接近/接触 25%、多猪密集 25%、远小目标 10%、弱光/模糊 10%。**真实 cutout 叠放只是快速可行性数据，不能冒充物理正确的 3D 接触。** 遮挡后 visible mask 必须按层级从后到前重新计算；过度羽化/重影、视觉穿透、边缘 Halo 的图要拒绝。

### 4.2 Blender（真实场景分布约束后扩展）

V3 固定的优先级仍为：**64–128 张几何/标签 smoke**（合法站立、接触、不穿模、可见/完整 mask、深度与相机投影 QA）→ 100–200 个同几何多外观视图检查 → 约 600 个不同 geometry scene 的短程数据对比（经预算授权）→ 正面结果再考虑 3,000+。

Blender 输出 RGB、visible_mask、amodal_mask（仅辅助诊断）、depth、scene/camera/pose/lighting/seed 和质量审核；YOLO 主任务只训练可见 mask。合成几何分布不能用 test 拟合。

### 4.3 真实高频/相机链：必须配对验证

同一个 geometry_id + visible GT 生成：
- V0：基础 RGB。
- V2：普通高频/PASTA 风格扰动。
- V3：来自 **Real-train** 的前景/背景分区真实高频统计与相机噪声/压缩适配。
- 条件满足后，V3a HF-only、V3b camera-only、V3c both 分离因果归因。

同几何配对时 mask hash、数量、训练步数、scene_id 保持一致。频谱或高频处理不得更改几何轮廓、实例标签；不要把真实整图 FFT 相位混入合成图导致语义重绘。若 V3 相对 V2 有提升但未拆 camera-only/HF-only，只能称“复合外观适配有用”，不能宣称高频信息独立有效。

**建议的第一轮数据量**：5,000–10,000 张 cutout 合成图是容量上限候选；不能代替前述小规模 smoke 和同几何消融，也不是已冻结配额。

## 5. 原型监督目标：先可读性，再独立训练

### 5.1 同一特征空间，不预定义部位语义

固定 B* 的 Backbone/Neck 和输入预处理，真实/合成样本用同一 P3（P4 可选）的 feature extractor 与共享 128D projector：
\[
z_{xy}=\operatorname{normalize}(W F_{\mathrm{P3}}(x,y)).
\]
训练 projector 后与原型库共同保存版本和 SHA；最终查询必须使用**同一规范化、维度、尺度与几何约定**。首轮固定 K=32 做可复现 baseline；V2/V3 的 DP-means 风格自适应新建/合并/剔除 K 是**后续机制对照**，不因 V4 示例配置被删除。

原型由前景内部、猪-背景边缘、接触区域以及必要的背景负 token 的平衡 reservoir 中自动发现；每簇事后记录 top exemplar、真实/合成占比、相机/场景/尺度、C/B/I 支持率。**不强制每头猪使用互不相同的 prototype，不把 prototype 预先命名成头/腿/背部。** 可选实例级 pooled 主 assignment 用于辅助稳定性，不作为像素级实例标签，也不替换原生 mask head。

### 5.2 候选独立预训练损失

\[
L_{\mathrm{SPM}}=
\lambda_{\mathrm{fg}}L_{\mathrm{fg}}
+\lambda_{\mathrm{contact}}L_{\mathrm{contact}}
+\lambda_{\mathrm{cons}}L_{\mathrm{cons}}
+\lambda_{\mathrm{sep}}L_{\mathrm{sep}}
+\lambda_{\mathrm{code}}L_{\mathrm{code}}.
\]

| 损失 | 监督来源及实现 | 要排除的失败 |
|---|---|---|
| 前景 \(L_{\mathrm{fg}}\) | 所有可见实例 mask 并集的 P3 soft coverage；BCE/Dice 与 ignore | 原型只响应背景 |
| 接触 \(L_{\mathrm{contact}}\) | C 边界带 soft target，对 B/C 分开计量；class-balanced BCE/Focal | 只能识别 pig/background，不能读同类边界 |
| 一致性 \(L_{\mathrm{cons}}\) | 同一源 crop 经颜色/压缩/几何增强，反变换后在有效可见重叠区域比对 embedding/assignment | 轻微扰动引发模式翻转 |
| 邻近分离 \(L_{\mathrm{sep}}\) | GT mask masked-pooling 后只对同图接触实例对作 margin/对比约束，不要求不同猪独占原型编号 | 邻猪表示混为一体 |
| 用率与多样性 \(L_{\mathrm{code}}\) | 簇使用率、死槽率、原型相似度与方差，防止退化；权重不可强制均匀数据分布 | 所有样本使用一个原型 |

仅作**候选初值**：0.35 / 0.25 / 0.20 / 0.15 / 0.05；实际各损失量纲需用 train/dev 校准，不可把手写权重当已验证最优值。首个 pilot 可先从 \(L_{\mathrm{fg}}+L_{\mathrm{contact}}+L_{\mathrm{cons}}\) 开始，再逐项加入 sep/code 消融。辅助损失作用到投影器/辅助头，不因 “prototype 是 memory” 就默认其梯度必然有效；训练时必须记录 gradient norms 和非塌缩。

### 5.3 原型冻结前的验收

记录：每簇使用率与 dead slots、exemplar 近邻、真实/合成来源比例、C/B/I token 覆盖、跨视频稳定性、foreground activation energy、同实例扰动一致性、邻猪接触可读性。任何单张漂亮热图都不构成正面结论。冻结产物至少包含 prototypes、projector/normalization、B* upstream checkpoint SHA、train source/sample manifest、K 与温度/top-k、构库时间/seed、exemplar 和完整来源统计。

## 6. 是否融合训练：三段式，防止特征空间漂移

**数据融合**与**模型参数联合训练**是两个不同决策。

### Stage B：先选无 memory 的数据基线 B*

Real-only 与 Real + 同几何合成 V0/V2/V3 使用相同起始 checkpoint、真实样本预算、总优化步数、输入尺度、优化器、评估；数据比例（如 real:synthetic=2:1）只作为可比候选。选择 B* 时不能用最终 test，且 B* 可为最简单的数据配置；不能预设真实高频胜出。

### Stage C：独立学习、真实域适配并冻结 memory

在 B* 的**冻结 Backbone/Neck**特征空间中训练 projector/aux heads、真实+合成同图增强一致性与 prototype reservoir。可考虑 synthetic warm-up → mixed real/syn → real-only adaptation 的对照顺序，但**必须与直接混合训练共享样本信息及更新步数**，才能判断“分阶段”是否必要。训练后冻结 bank 与 projector；共享 query 特征定义。

### Stage D：P3 注入与冻结库微调

\[
q_{xy}=W_qF_3(x,y),\quad a_{xy}=\operatorname{softmax}(\mathrm{sim}(q_{xy},P)/\tau),
\]
\[
R_{xy}=W_o\!\left(\sum_k a_{xy,k}V_k\right),\qquad F'_3=F_3+\alpha R.
\]

若 Top-k 检索，K、k、检索效率与计算成本记录清楚。**\(\alpha=0\) 初始化须复现 B* 数值输出；梯度 smoke 要确保 \(\alpha\) 或后续门控层能够从零起步，原型 buffer 在训练全过程不变。** 先冻结原生 Backbone/Neck、bank 和与构库一致的 projector，训练融合参数与允许的 YOLO 原生 head；如冻结 Neck 不足，进入另立试验的小学习率 Neck 微调，但必须监控 query-bank 特征漂移，必要时重建/重冻结 bank 并重做对照。**不要以“库冻结”自动认为 query 空间不会漂移。**

端到端解冻 prototype bank 只作为后续独立消融，不是主结果；若需同步使用 SPM 辅助 loss，必须让等容量无记忆 G 组获得公平的额外 supervision 或明确报告差异。第一版不改 P4/P5、Proto26、mask coefficients、NMS、mask loss 权重。

## 7. 消融矩阵（与 V2 B*/C/D/E/F/G 命名一致）

| 臂 | 内容 | 主要判别 |
|---|---|---|
| A / D-R | 原生 real-only YOLO26-Seg | 基础性能 |
| D-U / D-C0 / D-C2 / D-C3 | 相同预算的无 memory 合成条件/频率比较 | 数据几何与外观的独立贡献 |
| B* continued | 选定最佳合法数据基线再训练相同步数 | 排除额外训练量 |
| G Capacity-NoMem | 与 F 匹配特征、投影器预训练信息、训练数据、可训练容量、步数的普通 Adapter | 排除额外参数或辅助监督 |
| C Real-Mem | 只使用真实训练域的原型 | 真实记忆价值 |
| D Syn-Mem | 只使用合成原型 | 合成记忆价值 |
| E Joint-Mem uniform | 同 real+syn memory，但无条件或简化检索 | 条件 Top-k 的贡献 |
| F SPM-Gated | 联合原型库 + 条件检索 + P3 gate，主要候选 | 相对 B*/G 的独立增益 |
| K Fixed-K | 固定 K=32 或实际 K 对齐的 k-means 库 | 自动 K 是否真正必要 |

优先完成最小对照 A、B* continued、G、C、F；待通过机制 Gate 后再扩 D/E、不同 K、P4、真实 HF 与 Blender 规模。**公平比较至少对齐真实/合成图片、额外 projector 预训练信息、监督目标、训练 steps、seed 和模型选择准则**；无法完全对齐参数与 FLOPs 时必须报告不能归因的部分。

同一 F checkpoint 的必要干预：
- \(\alpha=0\)，去除 memory residual。
- 固定模型其余部分，更换错误的 key–value 配对、保留近似统计的 memory replacement，观察定向损伤；**只对整组 key/value 同步排列无效，因为检索对原型顺序不敏感**。
- 破坏空间查询与 retrieved residual 的对应、比较 real-only/syn-only 替换；控制残差范数与 OOD 干扰。
- 与**重新训练**的 G 比较，不能仅靠破坏测试认定因果机制。

## 8. 评价合同：四层证据，最终以独立 real test 为准

### 8.1 冻结原型自身

前景激活能量、activation-mask overlap、C/B/I 近邻召回、跨相机与增强 top-k assignment 一致性、dead slots、来源偏移/域私有簇；报告每类有效分母和分布，不把原型热图当检测增益证明。

### 8.2 原生分割主指标

**唯一最终主指标：真实独立测试全量 Mask mAP50–95**（与 V2 保持一致）。辅指标：Mask AP75、Box AP75、small AP/AR、稠密/接触/低对比/严重遮挡分层 Mask AP75、Boundary F1 和 Contact Boundary F1。Box/Mask 匹配各自按冻结规则独立执行；对预测实例诊断时保持身份不变。

### 8.3 定义清楚 Merge/Split/Miss

冻结 score/NMS/IoU 设定后，对每张图建 GT–prediction 重叠图并按**固定预定义阈值**分类：多个 GT 的显著可见区域被同一预测 mask 覆盖为 merge 候选；一个 GT 被多个有效预测片段覆盖为 split 候选；无有效匹配为 miss。明确一对多与多对一如何去重、是否与 miss 共计、接触区域低标注质量如何 ignore，报告数量和实例/视频级分母。不能直接拿 AP 变化替代 merge/split 改善。

### 8.4 预测稳定性与部署代价

对同一 real-test 输入做亮度、JPEG、轻模糊、轻平移和有限遮挡的**预定义扰动**，把预测 mask 逆几何变换对齐，记录：实例检测翻转、预测数变化、matched-mask IoU、score rank 变化、merge/split 变化及 raw 候选稳定性。统一 GPU、warm-up、batch、精度报告 FPS/latency、显存、FLOPs/参数量及检索开销。可以测鲁棒性，但调参只能在 dev，不能在 test 上选择扰动强度。

### 8.5 统计单位

至少计划 3 个训练 seed，结果按独立视频/猪栏簇 bootstrap 或分组 CI；列出每组视频/猪栏数量。单场景小样本的“提升 3 个点”不是统计意义证明。对 AP75 dense、merge、boundary 和全量 AP 同时报告绝对差、分层差、CI 与计算开销。

## 9. 候选成功/停止准则（Run 前必须预冻结）

下列数字仅是**预注册门槛建议**，不是已发生结果，也不能拿来事后挑优：
- 真实密集子集 Mask AP75 **绝对 +3 pp** 或预注册的等效效应门槛，且不只集中于一个独立视频簇。
- 全量 real-test Mask mAP50–95 损伤不超过预注册容忍（初建议 −0.5 pp）；小目标召回、接触边界与 merge 率至少有预定义的一项独立正向证据。
- F 相对 B* continued 与 G 均具有可重复的有价值差异，且同 checkpoint 关闭/定向破坏 memory 后存在符合预测的变化。
- 至少 3 seed、多视频/猪栏分层的效应方向与 CI 可解释，推理开销在部署约束内。

**停止或转向：**
1. P0 数据/Mask/模型不可信 → 只修 QA，不进行科研训练。
2. frozen C 边界 probe 对照同效 → 不扩大前景外观记忆，先查几何/分辨率。
3. 合成/高频只改善 synthetic → 不把域迁移作为已证实贡献。
4. F≈G 或记忆扰动无影响 → 不主张“跨域冻结原型”独立机制；可退回简单 Adapter。
5. P3 已有效才考虑 P4；若整体 AP 增但 merge 变差，按多指标披露，不称密集实例分离成功。
6. 数据簇过少、CI 太宽 → evidence_inconclusive，不能报机制阳性或阴性。

## 10. 建议工作包与产出格式

| 工作包 | 交付 | Gate |
|---|---|---|
| W0 只读审计 | split_manifest、model/eval SHA、标注 QA、C/B/I/D 接触账本 | 可追溯、无泄漏 |
| W1 frozen probe（需授权） | linear/MLP + RGB edge baseline、视频簇 CI | 同类接触线索超出普通边缘 |
| W2 合成素材 QA | real cutout/background manifest、64–128 Blender geometry smoke、visible mask label roundtrip | 无标签污染/明显物理错误 |
| W3 无记忆数据消融（需授权） | real vs paired V0/V2/V3、B* 选择记录 | 跨真实 dev 有可重复效用 |
| W4 原型 pilot（需授权） | fixed K=32 vs dynamic K 后续候选、投影器/bank SHA、来源画廊 | 非坍缩、跨域可查、C 线索可用 |
| W5 P3 融合（需授权） | alpha=0 parity、梯度 smoke、B* continued/G/C/F | 训练公平与机制干预通过 |
| W6 最终验证 | held-out real test、密集/边界/稳定性、3 seed/簇 CI、FPS/显存 | 主指标、损伤与成本完整 |

**运行登记：** 正式训练前另行冻结 optimizer steps、split/scene/checkpoint/code/prototype SHA、loss 配方与阈值、seed、仪器与预算；将完整命令、结果和失败状态写入本线 run.json。计划/代码/文档提交不等于实验启动，且不得更改其他独立研究线的 Run 状态。

## 11. V4 相对前序的具体补充

- 给出真实 tight/context/contact crop 与 I/B/C/D 接触标签的数据契约；
- 将“原型学什么”拆成前景、同类接触、增强一致、邻近分离及防坍缩辅助监督；
- 将“是否融合训练”明确为无记忆 B*、离线训练+冻结 memory、P3 zero-alpha 融合、可选低学习率联合微调；
- K=32 固定 pilot 与 V2/V3 自适应 K 保持并存；训练后利用 exemplar、来源、使用率解释原型；
- 细化真实 HF + 相机链配对消融、等容量 G 与真实测试 merge/split、扰动稳定性和视频簇统计；
- **不改变当前唯一前置工作 P0，所有规模和阈值仍是待冻结候选值。**
