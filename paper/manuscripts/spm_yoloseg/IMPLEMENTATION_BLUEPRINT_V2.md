# SPM-YOLO V2：从真实猪场到 Blender、跨域原型记忆与 YOLO26-Seg 的落地蓝图

**日期：** 2026-10-22　**状态：** design_draft / hypothesis_only / not_started  
**上位约束：** RESEARCH_RULES.md、AGENTS.md；本文件细化而不撤销 RESEARCH_CHARTER.md 和旧实验。未冻结正式预算、未开始渲染或训练。  
**原则：** 真实训练域统计 → 条件化合成数据 → 真实成像适配 → 共同特征空间与动态原型 → 冻结检索注入 → 真实独立测试。先证明合成改进有用，再判断原型的独立增量。

**2026-10-22 基础实现细化：** 用户要求先完成必须的基础方案。当前实现以[基础方案V0.1](BASE_IMPLEMENTATION_SPEC.md)和[配置草案](../../../experiments/spm_yoloseg_20261021/BASE_CONFIG_DRAFT.json)为入口：单猪可见任务、共用RLE加载器、只用P3、冻结构库/查询坐标、保留原生两训练分支。本文中的P4、真实HF和多臂比较保留为后续扩展，不构成基础链路必须先取得效果的Gate；任何已冻结Run均不改变。

## 0. 完整系统的输入、输出和核心预测

### 输入
1. **真实训练数据**：RGB 原始视频/抽帧、可见实例 Mask（有标签部分）、拍摄视频/猪栏/猪场/相机/时间身份。没有 Mask 的图只参与允许的统计或无标签表征训练；实际用途必须记录。
2. **资产**：现有 pig.dae（24 骨骼，约 1.625 s 动画）、可用且授权的猪体纹理/材料、猪栏和地面等几何场景、实际摄像机参数。若当前 Blender 无法直接导入 .dae，先转换 glTF/FBX 并通过骨骼/蒙皮/坐标系回归测试；不能以复制一个动作循环代替动作多样性。
3. **基线**：精确锁定权重、源码与评价口径的原生 YOLO26-Seg，独立于旧 prototype→coefficient Run。

### 输出
- 全程可追溯的训练真实域统计表 real_domain_profile.json；
- 带可见/完整掩码、深度与场景真值的 Blender RGB，及场景清单；
- 从**真实训练图像**建立、带分区与来源标签的 real_texture_bank；
- 同几何同标签的 raw / low-frequency / high-frequency / combined 四类合成视图；
- 由真实+合成训练 token 学得的可冻结原型记忆和检索解释卡；
- 与原生输出口径兼容、只在 P3 起步的 SPM-YOLO；
- 预冻结消融报告和真实独立测试结果。

### 核心可证伪预测
**H-data：** 真实域约束的合成场景比任意随机场景在固定真实测试上的密集/遮挡 Mask 效果更好。  
**H-frequency：** 在同几何同数据量条件下，依据真实训练域构建的有类别与尺度约束的高频适配，优于纯渲染或不使用真实统计的高频扰动；也可能只改善相机鲁棒性而不改善接触边界。  
**H-memory：** 同图、同监督、同训练步数、近似同容量时，跨域原型检索优于无记忆分支，关闭/破坏记忆应按预期损失增量。三者必须**分别成立**才可分别作为论文贡献。

## 1. D0：先从真实数据建立「猪场场景分布」，再决定合成什么

### 1.1 最小数据清点与防泄漏
每帧注册：image_sha256、source、farm_id、pen_id、video_id、camera_id、timestamp/frame_id、resolution、fps、codec、license、mask_annotation_source、mask_quality、split、annotation_revision。先按猪场→猪栏→视频/相机/连续时间段分簇划分 train/dev/test，再抽帧。相邻帧及同拍摄事件不能跨 split；最终 test 的 RGB 和统计量也不用于 texture_bank、场景分布拟合、原型、投影器、调参或挑例。若只有一个场地，应明确结果只支持跨视频/跨相机，不能称跨猪场泛化。

### 1.2 分层抽样，而不是随便截几百帧
从训练簇采样稀疏、中密集、紧密接触、严重遮挡、逆光/阴影、低照度、运动模糊、不同机位的帧。抽帧间隔以动作相关性/场景变化定，不把连续 30 帧当 30 个独立样本。用于场景统计的训练帧可以无实例 GT，但接触、遮挡真值需要高质量分割或人工抽检；初始建议人工校验 150–300 张、至少覆盖全部已知训练相机/时段，实际样本量由真实数据确定，非冻结硬指标。

### 1.3 输出必须测量的分布及拟合方法
| 统计量 | 训练域的具体计算 | 合成器如何使用 |
|---|---|---|
| 机位/视场 | 相机分辨率、FOV/K、畸变、俯仰、相机高度、猪栏可见区域；若无标定，记录估计误差 | 固定真实机位族，周围有界抖动；另设未知机位压力集 |
| 可见实例密度 | 每帧标注可见猪数量 N；难例按视频/栏归属 | 条件采样 N 而非独立均匀取 1–20 |
| 实例尺度 | bbox 短边、可见 Mask 面积 / 图像面积、长宽比、到边缘距离 | 匹配投影尺度和小猪/远猪出现概率 |
| 位置/方向 | 实例中心热图、主轴方向；可用关键点时记录姿态分布 | 约束猪走向、食槽附近聚集、活动范围 |
| 接触/遮挡 | 膨胀后接近的不同实例边界长度、邻猪间距；遮挡需人工或合成 amodal 辅助，不凭普通可见 Mask 猜完整比例 | 单独分配接触与可见比例的样本预算 |
| 光照/画质 | 按相机/时段分亮度、白平衡、曝光、局部动态范围、噪声/模糊/压缩、图像功率谱 | 场景材质与后处理遵守条件统计 |
| 背景 | 地面、墙、栏杆、食槽、阴影、污渍占比和纹理特征 | 生成真实可见的结构与材质，不将无关背景随机覆盖 |
| 外观 | 猪体体型、肤色、斑点/污垢、姿态/反光/皮肤纹理范围 | Blender 形态和 PBR 材质拟合分布 |

**统计不是复制分布：** 训练域拟合覆盖常见 70% 左右质量区间，额外有界压力采样覆盖稀有真实但合理的 30% 场景（建议初始比例，待 dev 冻结）。不能用 test 域拟合得更像 test。

### 1.4 实际应完成的产物
- real_domain_profile.json：按 camera_id × lighting_bin × density_bin 保存分位数、有效样本量、来源哈希；
- 训练簇标签和高/低频统计入库的 cutout/patch provenance；
- 一个可视化报告：实际帧 vs 所测尺寸、接触、曝光、频谱直方图；
- D0 Gate：分簇可信；Mask 定义可核；真实高频来源限定于 train；无法稳定估计某统计时使用较宽物理合理范围并标注 UNKNOWN。

## 2. D1：Blender 场景合成器（先几何真实性，再照片真实性）

### 2.1 场景/物体资产
- 建立地面、围栏、饮水器/食槽、光源、摄像机、阴影投射物体的可版本化资产；栏高和相机坐标采用米、统一世界坐标。
- 猪资产整理为统一骨架/原点/前向/体长单位；猪体形态用骨架缩放+有限形变（长度、腹围、腿长、肩宽），不能靠各轴任意缩放制造畸形。
- 从真实训练图提取材质模板（肤色、污点密度、粗糙度、皮肤法线、毛发/细纹尺度）；随机化纹理相位、位置、颜色、强度。真实纹理只从 train 且得到许可的 cutout 得来。
- 现有 gait clip 的步态相位随机只是第一步；增加站立、行走速度、低头、转身、进食、尾/头运动以及动画混合；脚步锁地避免漂浮、滑行。高遮挡场景必须有真实几何接触，不用两张透明猪贴图伪造接触。

### 2.2 两层场景采样
**层一：真实约束条件采样**：先选 camera profile、light profile、N/density、occlusion/contact bin、pose activity bin、body-scale bin。以真实训练条件统计为候选目标，按 scene_id 记录概率和调度。  
**层二：布局求解**：将每猪中心放入合法地面多边形，用碰撞体拒绝不合理躯干互穿；在指定密集度下通过有目的的相邻中心/视线遮挡采样实现接触。检查蹄子落地、肢体不穿墙、姿态合法；不满足则重采样，记录 reject reason。物理合法的接触前后仍可有真实相互遮挡。

初始建议以最终视图可见猪数近似划为稀疏/中密/高密各一档，**真实阈值先由 D0 分位数确定**；每档保持多样 camera、pose、light 与遮挡比例，不用固定「5 猪=稀疏」套所有视角。针对困难样本可对紧贴/局部遮挡加权过采样，但每个困难场景还须有容易场景匹配的几何对照。

### 2.3 渲染和相机链
- 使用 Blender Cycles/Eevee 的确定版本，优先 BlenderProc 脚本生成带种子的场景；可用其 render_segmap(map_by=["instance","class","name"]) 和 COCO writer。别在原生场景改完 RGB 后忘记重同步标签。
- 设置真实训练机位的内外参，附有限扰动、镜头畸变；必要时在**高分辨率**渲染后应用 PSF/模糊、失真并下采样到真实最终像素网格。
- 模拟棚内 LED、天窗或辅助灯、混合色温、相机自动曝光/白平衡变化、阴影与高光；各参数不应独立极端随机，而是组合成有物理意义的 lighting profile。
- **分离渲染与相机成像：** 渲染输出理想 linear RGB；后处理包含色彩映射、镜头 PSF、图像噪声、压缩，且版本化。

### 2.4 两个实例掩码的严格定义
**Visible mask**：每个 RGB 像素最终实际最前方猪物体对应的实例 ID（前景栏杆、其他猪都会遮挡），可见轮廓可能破碎/有孔；训练真实可见实例任务只用它。  
**Amodal mask**：在相同相机和同一猪姿态下单独重渲该猪，暂时隐藏会遮挡它的其他猪及非地面前景物体；仅用于 occlusion_ratio / 辅助分析，仍受画面裁剪约束，需明确是否包含自遮挡。  
occlusion_ratio=1-area(visible)/max(area(amodal_on_screen),1)；不要把部分出画天然当成被其他猪遮挡。对被完全遮住、可见像素过少的猪，保存 3D 真值并按真实标签政策决定是否在可见任务中忽略，不能强行标成可见。

逐图同时输出 RGB、visible instance-id PNG/RLE、amodal masks、bbox、depth、normals、scene graph、camera K/T、materials、poses、random seeds、Blender/asset/script hashes。COCO 的可见 mask 建议以 RLE 为权威真值；如果训练器需要 YOLO polygon txt，必须测试破碎 Mask/孔洞的表达损失，不能静默把多个不相连可见部分当一个简单 polygon。

### 2.5 合成 QA 与首次预算
**Smoke 64–128 帧（设计建议）**：稀疏、中密、接触、遮挡×两类机位/光照；对每个实例重叠检查 ID、bbox、visible 和 amodal，抽检 raw 图与 mask 叠加。  
- 同一像素可见实例 ID 唯一；接触但无互穿；所有实例的 RLE 编解码一致；图像变换后 mask 同坐标；
- 计算可见 mask→目标训练格式→mask 的 roundtrip IoU，建议试验合同设 >=0.995（RLE 无损应为 1.0），若断裂 mask 不能表达则先修加载器；
- 对齐 D0 的 density/size/contact/occlusion/edge-width 直方图；保存偏差，不以「看着逼真」取代可测条件；
- 先做 600 张的 paired pilot（约每密集度档 200 个**不同几何场景**），同一几何产生多个外观版本；P1 再决定是否扩到原建议的 3000 张，不能直接以几万个渲染掩盖错误。

## 3. D2：真实数据如何提取可用「纹理与高频」，而非整张图频谱乱贴

**核心：按语义/成像来源/像素尺度拆成三类。**
1. **外观真实纹理库**：经核准的猪体内部皮肤 patch、环境的地面/墙面/栏杆 patch，附实际像素尺度、相机、亮度、颜色/频谱。猪皮只注入猪体内部；地面只注入地面。可用 UV/PBR 材质贴图合成粗糙度/法线细节，或在渲染后做高频残差适配。
2. **真实图像高频统计**：每类训练 patch 的 Laplacian 多尺度能量、二维频谱振幅/径向功率谱、方向性、跨通道相关，避免直接搬目标图像的相位/边缘。真实 Mask **先腐蚀内缘再裁 patch**，不把真实猪的外轮廓作为合成猪的假肢。
3. **真实相机退化分布**：按 camera_id/lighting_bin 估计模糊核/边缘扩散、噪声幅度随亮度变化、锐化光晕、H.264/HEVC/JPEG 块效应和采样/下采样。若只有压缩视频，估计的是**部署最终视频流的联合退化**，不是原始 RAW 传感器噪声。

**配对视图要求：** 对同一 scene+mask，保存 raw、物理相机退化、仅频谱增强、按真实条件频谱+皮肤残差等 paired 变体，不改变几何、相机投影、监督信息和训练图数量。统计提取/混合细则详见 [REAL_FREQUENCY_TRANSFER_SPEC.md](REAL_FREQUENCY_TRANSFER_SPEC.md)。

## 4. D3：从真实图像提取与自动发现视觉原型

### 4.1 训练输入与可见对象
- **基础版空间是B的冻结YOLO26-Seg Neck P3，P4留作扩展**，不是直接在RGB做K-means；保证构库和推理用同一坐标。第一版是单猪可见实例任务（nc=1），保留原生Mask decoder及结构，不是保持COCO80类训练任务不变。
- 真实原型只允许训练簇 RGB 和训练真实可见 Mask。未标注训练帧可在明确的伪标质量协议下用于无标签增强一致性，但严禁把未审核的伪实例当作可靠的实例内部 token。
- 每帧先执行完全一致的 letterbox/resize/color preprocessing，记录从原图→网络输入的仿射，Mask 同几何映射；采样前保证 P3/P4 网格对齐到 Mask。

### 4.2 Token 采样：实例、边界、背景分开统计
对每个 P3/P4 位置将其对应采样区域投影至 Mask，计算：
- fg_ratio、dominant_instance_ratio（多个可见猪是否混在一个位置）、distance_to_same/different_instance_boundary、visible area bin、camera/domain；
- **纯实例内部候选**：fg_ratio>=0.8 且 dominant_instance_ratio>=0.8（仅候选阈值，注意真实卷积感受野远大于一个 cell）；
- **邻猪接触候选**：在两个不同猪可见边界的指定像素邻域，可保留位置但标记 mixed，不要求它归属某个「猪头/猪腿」预设语义类；
- **猪-背景边界与纯背景候选**：独立记录，降低背景比前景多导致的聚类占据。
每视频/每猪/每场景有采样上限；按 real/syn、尺度、小目标、遮挡/接触、不同摄像头**分层 reservoir**，不能让一个 20 头猪的连续视频把所有中心占满。以图像/视频为独立支持单位记录 source_count。

### 4.3 训练「共同空间」而不是一上来直接聚类
固定 B Backbone/Neck，P3/P4 用两个 1×1 投影归一化到 d=128。训练投影器使用两类**可靠正样本**：
- 真实图：同一训练帧两次轻量 photometric/geometry augmentation，通过**已知几何变换**索引同一可见区域 token；
- 合成图：同一场景/姿态/相机/可见 Mask 的 raw、真实频谱迁移和随机材质版本，几何位置精确对应；
不得把**不同猪个体**的所有 token 生硬定义成负例，也不得把未知部位的 real/syn token 硬配为正样本。

候选目标：同区域一致性（VICReg/方差-协方差防塌缩思路）+ Pig/Background 辅助二分类 + 各来源采样均衡；所有系数在 dev 冻结。初版不建议叠加额外 ViT/DINO 教师和域判别器，避免引入新的预训练知识和算力混杂。检查投影输出方差、真实/合成同条件检索一致性、前景与背景 separability，出现常量塌缩就停止构库。

实现细节：方差/协方差正则作用于投影后的未归一化h，归一化q只用于检索/聚类；同图几何增强先做坐标逆映射和有效区过滤。球面归一化中心的动态算法标作DP-means-inspired，不沿用原欧式DP-means目标的收敛保证。

### 4.4 数据自适应发现 K（不预定义猪头/猪腿等部位）
对 L2-normalized z 建立 foreground、background 两个可查询分区，接触/尺度/域作为支持元数据；可选后续增加边界专用子库，但第一版不增加第三类训练语义。
- 动态中心发现：最小平方距离大于 λ 时新建中心，否则更新；对中心定期重算，合并近重复，移除未达最低真实图片/来源支持数的簇。
- λ 在训练/开发检索稳定性上选；记录中心规模曲线、触及容量上限的频率、real/syn 支持占比。K_max 是算力保险而非预定语义数量，首版可尝试 128/256 容量保险，若触限须报告被截断而非声称「自动发现 K」。
- 每个中心附：feature、support_count、独立训练视频数、各域占比、距离分布、最近 real/syn 训练 patch 索引、尺度/遮挡/相机频数、聚类版本 SHA。构库后冻结。
- 可解释性：为每个中心展示各域 Top-8 检索 patch、位置、可见 Mask 与失效例，由人工**事后**命名近似概念（背部纹理、腿缘、地面、接触边等），且记录混合/不可解释状态；这不等于模型天然有部位标签。
- 若原型只在 synthetic 命中、real 近邻距离大，应标记 domain-private，不强迫它进入共享主库；真正跨域原型需独立验证数据的命中率、纯度和跨条件复现支持。

### 4.5 冻结一致性合同
memory.pt 含 centroids、W3/W4 projection weights、normalization、cluster config、base-neck checkpoint SHA、data split/hash、real/syn 支持与 exemplar mapping、K；仅推理读取它。若后续解冻作为 query 来源的 Neck，其特征已不再是构库空间，第一版**禁止解冻 query tap 上游**；允许训练 tap 下游的注入层与 Segment26 head。若有充分理由解冻 Neck，需要另立「固定教师 query 支路的额外时延」或「漂移约束」实验，不能悄悄沿用冻结库。

## 5. D4：原型如何注入 YOLO26-Seg

### 5.1 最小版本严格只注入 P3
YOLO26-Seg 选择 640 输入时需以锁定源码审计 P3/P4 真实形状；典型 P3=80×80。对每个 P3 位置 x_i：
- q_i=norm(W3 F_i)，用 F_i 的固定 tap 特征查询；
- p_j 为冻结的前景/背景中心，s_ij=dot(q_i,p_j)/τ；
- 各库 Top-k（例如 4，待 dev 冻结），得到 r_i^fg、r_i^bg、maxsim 与 sim_gap；
- u_i=W_o([r_i^fg,r_i^bg,maxsim_fg,maxsim_bg,sim_gap]) 映射回 P3 channel；
- g_i=sigmoid(MLP([LN(F_i),sim_gap,maxsim_fg,maxsim_bg]))；
- F'_i=F_i+α·g_i·u_i，其他 P4/P5 首版不变；原生 Segment26、Proto26、mask coefficient、后处理接口保持不变。

**zero-residual 实现重点：** α=0、W_o 小随机非零，避免 α 和 W_o 双零导致永久无梯度；先验算 eval 与训练辅助分支的零残差等价，再检 α 首次更新非零、gate/W_o 随后真正有梯度。若一个训练 step 中 α 更新为零，检查 loss 连接和 optimizer 白名单。

### 5.2 查询与训练负载
- 先冻结 P0→P3 query tap 的整个 Backbone/Neck，只训练必要的融合与原生 head（包括 one-to-one/one-to-many 合法路径）；B-continued 与无记忆等容量 G 同样冻结/同样训练步数；
- 小库容量先测 64、128、256（依 P2 实际 K 与内存上限决策）；P3 6400 queries × K × 128 维的矩阵乘法负载需量化。K=128 时仅相似度内积约 105M MAC/图（不含 query/out projection），绝非「免费」；必须报告实际延迟与 FLOPs；
- 读取库时原型和投影器为 frozen buffer/weight，不进入 optimizer，不在推理按实时输入更新；追加 P4 前需要证明 P3 的有效增量和检索开销可接受；
- 真实部署只有 RGB 输入与模型内部 memory，没有 Blender 图、真实 Mask GT 或按测试图重构 memory。

## 6. D5：训练顺序不能把几种收益混在一起

**第一层：数据。** R(real only) → S(raw Blender mixed) → SF(FDA/PASTA 类频谱扰动) → SR(real-conditioned high-frequency mixed)。保持每次训练的真实数据、几何 scene IDs、训练图总数、初始化、更新步数、采样比例一致。高频变量应针对同一 scene 的不同外观，不通过多一倍样本混淆。先定最简单哪种合成训练组合真正能在 real dev 上收益，再锁定用它构库。

**第二层：记忆。** 固定上一层选中的 B* 的 Backbone/Neck 和训练数据，构 real-only / synth-only / combined 三类 memory（来源平衡、K 和查询量报告）；相同 B* checkpoint 起步微调 FSPM 和 G-no-memory，尽量匹配可训练参数、head 训练步数、投影器离线训练信息及预算。FSPM 应优于 G 才有记忆独立价值。

**第三层：机制（后续）。** 同checkpoint关闭memory / 错误key↔value对应 / 真实-合成替换 / top-k限制 / fg-only / bg-only等；共同重排keys和values的表行不改变Top-k检索，不能作为有效破坏。简单关闭分支会有分布移位，需结合等幅控制与可重训无记忆G。基础链路阶段先实现明确的key/value存储与可重放接口。

## 7. D6：评价指标、错误类型、统计和停止规则

- **唯一主任务**：真实独立测试 Mask mAP50–95；Box AP 和 Mask AP 独立匹配，不能跨候选拼结果；正常推理口径与官方实现保持相同。
- **预登记关键子集**：猪猪接触（边界接近的不同实例对）、严重遮挡（只在有可靠 amodal/人工规则时计算）、小可见 Mask、低对比、不同相机/猪栏；原 GT 数量、簇数量和比例必须报告。
- **机制定位**：漏检/粘连/分裂/边界 IoU，按固定输出实例记录对同一 GT 的失败（不能只以跨实例 Box75/Mask75 的各自最优拼故事）。
- **域迁移诊断**：S vs SR 差异、matched scene 的 spectrum PSD 距离、真实边界宽度、相似度分布；**PSD/FID 好看不自动说明 Mask AP 会好**。
- **统计单位**：按独立猪栏/视频/摄像机簇 bootstrap，并在开发前冻结 seed/CI 方法；最终盲测一次评价，若样本簇少就明确统计精度不足。
- **工程指标**：FPS/latency、显存、参数、真实投影维度与 K、数据渲染/构库耗时，记录 input resolution、batch 和硬件。
- **Gate：** D0 split 不合法→停；D1 mask 不可信→停；合成 SR 无效但 S 有效→回退 S 并停止无意义 HF 工程；所有合成负迁移→优先修域差/场景生成；P2 real 检索失败→不启动大规模记忆训练；FSPM 不优于 G 或置乱无影响→停止声称记忆机制独立增量。

## 8. 具体代码分层与产物接口（文件路径为**拟建**，不是已存在的代码）

建议实施时新建如下小模块，避免 Blender 逻辑、图像增强、原型训练、模型头缠绕在同一脚本：
- scripts/spm_yolo/00_audit_real.py → real_domain_profile.json、manifest、split；
- scripts/spm_yolo/01_build_real_texture_bank.py → patches、patch_manifest.jsonl、spectral_stats.npz；
- scripts/spm_yolo/02_render_blender.py → scene_manifest.jsonl、raw RGB、visible RLE、amodal；
- scripts/spm_yolo/03_apply_camera_frequency.py → 同 scene 的风格版本，复用 RLE；
- scripts/spm_yolo/04_audit_synthetic.py → mask roundtrip、分布诊断、可视化审计；
- scripts/spm_yolo/05_train_projector.py → W3/W4 + 中间训练状态；
- scripts/spm_yolo/06_discover_prototypes.py → memory.pt + exemplar/cluster card；
- scripts/spm_yolo/07_injection_smoke.py → 基线等价、梯度、checkpoint、导出和耗时；
- scripts/spm_yolo/08_run_ablation.py → 各臂完整记录/同预算配置；
- scripts/spm_yolo/09_eval_real.py → 正常 COCOeval、分层、聚类置信区间、时延；
- 每脚本固定 config.yml、seed、输入/输出 SHA、code commit、日志；大图像/权重仅留授权执行端，由 artifact_manifest 定位。

### 最小数据合同
- real_manifest.jsonl：source_id、farm/pen/video/camera、timestamp、frame_sha、resolution、split、mask_origin、qa_status、license；
- real_texture_bank/patch_manifest.jsonl：patch_sha、frame_sha、**train-only**、region_type、mask_erosion_px、camera/illumination_bin、native_px_scale、patch_shape、validity；
- synth/scene_manifest.jsonl：scene_id、seed、asset_hashes、geometry_hash、pig_positions_and_poses、camera_profile、lighting_profile、label_policy、visible/amodal hashes；
- synth/view_manifest.jsonl：scene_id、view_style(raw/fda/pasta/real_hf/hybrid)、donor_patch_ids/stats_hash、camera_pipeline、RGB_sha、**same_visible_mask_sha**；
- memory/manifest.json：feature_extractor_sha、projection_sha、split_sha、training_domains、K_fg/K_bg、lambda、support_hist、query_contract、memory_sha；
- evaluation/contract.json：weights、code、split_sha、model_outputs_policy、segm_decoder、normal inference config、cluster bootstrap unit。

## 9. 后续候选判别：同一 Blender 场景的高频适配增益

**只控制数据渲染，不增加训练模型复杂度。** 用真实训练域 profile 建 100–200 个 QA 场景，直接生成成对 raw / random high-frequency / real-conditioned HF / real-conditioned HF+camera degradation。先核照片质量、可见 GT 一致性、real-vs-syn PSD 和与真实尺寸/密度分布的差异。之后 600 matched geometry 的小样本训练 B_raw、B_randomHF、B_realHF（真实数据和 step 完全相同），在 real dev 看 Mask AP、遮挡/接触子集和负迁移；通过之后再投资更大 Blender 和 memory。条件已具备时再按 [PILOT_AND_ABLATION_V2.md](../../../experiments/spm_yoloseg_20261021/PILOT_AND_ABLATION_V2.md) 冻结。

以上保留为扩展候选，不是当前下一动作，也不是原型基础实现的前置成功条件。当前先按基础方案清点输入、接通共用标注/特征/梯度/存取接口；是否展开高频效用实验留待之后讨论。

## 10. 预定论文新颖性边界与参考起点

Blender/BlenderProc、领域随机化、FDA 低频风格迁移、PASTA 高频频谱扰动、聚类原型记忆本身**都已有相关文献**。本方案潜在可证实的增量是：**真实场景条件+类别局部真实频率统计的可见实例合成、相同几何的跨风格原型一致性训练、可冻结有来源解释的记忆条件检索**在密集猪群接触边界上的独立作用，而不是宣称创造了新的傅里叶变换/聚类。

参考：
- Yang & Soatto, FDA, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html
- Chattopadhyay et al., PASTA, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Chattopadhyay_PASTA_Proportional_Amplitude_Spectrum_Training_Augmentation_for_Syn-to-Real_Domain_Generalization_ICCV_2023_paper.html
- BlenderProc COCO/Instance Writer: https://dlr-rm.github.io/BlenderProc/examples/advanced/coco_annotations/README.html
- BlenderProc Camera/Lens Distortion: https://dlr-rm.github.io/BlenderProc/examples/advanced/lens_distortion/README.html
- Kim & Byun, Learning Texture Invariant Representation, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Learning_Texture_Invariant_Representation_for_Domain_Adaptation_of_Semantic_Segmentation_CVPR_2020_paper.html

**登记说明：** 上述数字、阈值和目录结构均为可执行设计建议，不是实测值或已冻结标准；真正训练前需要独立 protocol、真实 Run 和授权。
