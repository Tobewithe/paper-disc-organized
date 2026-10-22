# SPM-YOLO V2：高频域适配与原型记忆分阶段试验协议（候选、未冻结）

**日期：** 2026-10-22；**Study:** STUDY_SPM_YOLOSEG_20261021  
**状态：** proposal_only / not_started / formal_optimizer_steps=0；不创建 Run、不自动渲染或训练。

> **V3 执行优先级补充（2026-10-22）：** 完成 P0 的真实数据/Mask/模型审计后，优先进行[冻结 P3/P4 接触边界 Probe](CONTACT_BOUNDARY_PROBE_DRAFT.md) 的小规模科学判别（需另行冻结与训练授权），以辨明同类实例边界线索是否在特征中可稳定读取，再决定外观/几何原型和合成规模。V2 原设计的同几何 V0/V2/V3 高频实验保留、但**不是 P0 后默认立即开训**。[V3 决策树](../../paper/manuscripts/spm_yoloseg/PRIORITIES_RISKS_AND_DECISION_TREE_V3.md)。  
**前置：** [V2 实施蓝图](../../paper/manuscripts/spm_yoloseg/IMPLEMENTATION_BLUEPRINT_V2.md)、[频率迁移规范](../../paper/manuscripts/spm_yoloseg/REAL_FREQUENCY_TRANSFER_SPEC.md)、[RESEARCH_RULES](../../paper/manuscripts/spm_yoloseg/RESEARCH_RULES.md)。本文件补充原 [PROTOCOL_DRAFT](PROTOCOL_DRAFT.md)，**不覆盖或替代任何已冻结 Protocol**。

**当前暂不展开本矩阵：** 用户本轮要求先完善[基础方案](../../paper/manuscripts/spm_yoloseg/BASE_IMPLEMENTATION_SPEC.md)，之后再讨论实验效果。下列Stage A–E作为后续研究候选保留，HF对比阳性不是构建基础原型链路的强制前提；当前只做输入清点与接口设计，未授权远程运行或新训练。

## 1. 研究命题、竞争解释和主比较

| 主张 | 关键干预 | 合法主比较 | 竞争解释 |
|---|---|---|---|
| H-GEO 场景约束有用 | 真实训练统计条件采样 vs 无条件物理合理 Blender randomization | CONSTRAINED_RAW − RANDOM_RAW | 只是场景多样性/有效密集样本变多 |
| H-HF 真实频谱条件迁移有用 | 同一批 geometry/GT，real-conditioned 高频 vs 不依赖真实的强高频扰动 | CONSTRAINED_REALHF − CONSTRAINED_PASTA | 仅普通频率随机化或传感器模糊效果 |
| H-MEM 原型检索独立有用 | 相同 training examples、projector、训练步数及容量，冻结真实/合成 prototype memory 选择性检索 | MEMORY_GATED − CAPACITY_NOMEM，辅以干预 | 额外训练参数、投影训练和普通残差足够 |
| H-SIM2REAL 双源互补 | real-only、syn-only、joint memory | JOINT_MEM 对两个单源 + source coverage | 仅 train-real 数量更多或合成记忆噪声 |

**真实 independent test Mask mAP50–95** 是最终唯一主评价。普通 COCOeval segm/bbox 分别匹配，所有方法同 input size 和 normal inference；接触、遮挡及小目标为提前固定的辅助分组。

## 2. 完整分组矩阵（合理分段实施，禁止同时大规模运行）

### Stage A：素材、几何、频域（不训练检测模型）
- A0 数据审计：real source、许可、visible Mask、视频/猪栏 split、相机/频谱 profile，只读产物，Gate 已在原 P0。
- A1 资产烟雾：现有 pig.dae 导入/转换、材质、地面接触、机位和 depth/visible/amodal 多遍渲染；64–128 帧，修坏标注优先。
- A2 600 个**不同几何 scene_id**（候选 200 × 三类密度，真实分位数决定密度阈值）；每个 scene 产生 V0 raw、V2 high-PASTA、V3 real-HF 以及可选的 V1 low-FDA、V4 hybrid。A2 所有视图 Mask hash 相同，并有 patch bank 来源 hash。
- A3 按 camera/density/region 报告 scale、boundary visibility、真实/合成的谱 band energy、纹理伪影、几何穿模、标注 RLE 等 QA；PSD/FID 等不作为效用证明。

**A 阶段出站**：若训练/测试分组无泄漏、真实 texture 来源可核、>=0.995 的格式往返 IoU 候选门槛通过、随机 QA 中缺陷在预冻结限制内，才申请 B pilot。

### Stage B：数据效益短程比较（真实数据+合成，NO memory）
全臂同官方预训练权重 SHA、相同原生 YOLO26-Seg 版本、batch 有效大小、总优化步数、真实训练 frames、学习率/增强/评价阈值、相同 600 几何 scene_id 或预定义重采样表。凡「V0/V2/V3」为同一 scene 的替代视图，**每个 epoch 使用同量视图**。真实:合成的初始建议抽样比例 2:1（候选非已冻结）；real-only 用重复真实采样补足相同优化步数，记录不同训练信息量的固有差异。

| 编号 | 训练数据 | 解答 |
|---|---|---|
| D-R | Real only | 原生真实基线 |
| D-U | Real + unconditioned physically plausible Blender raw（scene distribution different） | 仅随机化是否足够 |
| D-C0 | Real + real-conditioned Blender raw | 场景几何约束是否有用 |
| D-C2 | Real + **同 D-C0 几何**的 PASTA high-frequency amplitude augmentation | 任意高频扰动对照 |
| D-C3 | Real + **同 D-C0 几何**的 real-conditioned high-frequency/camera transfer | 真实纹理/统计的额外作用 |
| D-C1（按需） | 同 D-C0 几何 + FDA low freq | 低频风格 vs 高频区分 |
| D-C4（按需） | 同几何，UV 真纹理 + real-HF + sensor | 验证层级复合方案 |

**重要：** D-C3 初次使用 real-HF + camera 是复合对照；若优于 D-C2，必须随后分别做 **C3a real-HF only**、**C3b camera only** 和 **C3c both** 才能把贡献归给真实高频统计，不能在缺少这组三臂时声称 HF 独立。D-U 与 D-C0 也可能同时改变密集/困难比例，应报告每个 bin 的样本分布，必要时匹配 bin 的有效样本以判别「分布匹配」与「更多密集样本」。

**B pilot 决策（建议，不是 Run Gate）**：
- 看总体 Mask AP、接触子集 AP75/Recall、低对比、小 Mask、不同 camera 的真实 dev 表现；
- D-C3 相比 D-C2 若有跨 camera 同方向增益且无明显高置信负迁移，保留 V3；若只有一个 camera 改善，先测增强来源混合；若 D-C3≤D-C2 且证据精度足够，选择更简单的 D-C2；
- 没有视频/栏独立簇、收益低于方差时记 evidence_inconclusive，不因视觉质感好就扩大 3,000。
- B 选定的最佳**可复现配置**记 B*，在主实验中作为统一图像/初始化/预算基线。B* 可是 D-C0/D-C2/D-C3，不能必然假定 V3 胜出。

### Stage C：真实与合成原型空间构建（离线，不改 YOLO）
- 固定 B* 的 Backbone/Neck SHA，使用 train only（包括合法 synthetic）P3/P4 和可见 Mask；同输入预处理。
- 建 128D W3/W4 projector，冻结 Neck，合成 style pair + 真实 train augmented pair，训练一致性/方差与辅助 fg/bg 分辨；额外离线计算量要记录且提供对照。
- 按 real/syn、video、尺度、接触、fg/bg reservoir 采 token。动态发现 K，初始 memory capacity safeties 128/256 可作 dev 资源比较；lambda、merge/prune、seed 和真实 source coverage 全量留档。
- 对真实 **dev** 的 out-of-bank token，记录 fg recall@topk、real-vs-syn neighbour match、覆盖/低支持原型占比、cos distribution、domain-private cluster 比例。此指标只能判断检索可行，不能代替最终 AP。
- 训练独立的 fixed-K kmeans memory 对照，K 等于动态方法**实际** K 或在公平检索预算下对齐；对于 domain-specific 记忆，K、Top-k、计算量均须披露。
- C Gate：投影没有塌缩、memory retrievable、构库 feature SHA 和 query SHA 相同；失败时先修 projector/采样，不直接叠更多深度/attention。

### Stage D：注入 smoke、合法微调与机制对照
同一个 B* checkpoint、同一 B* 训练集、相同优化器 steps、同 head 可训练参数范围；所有 memory 分支 query tap 上游冻结。原有 PROTOCOL_DRAFT 的 C/D/E/F/G 命名映射如下：

| 臂 | 记忆/模型 | 训练公平性与要排除的解释 |
|---|---|---|
| B* continued | 无新模块；同 head 微调等步数 | 排除单纯多做一步 head 训练 |
| C Real-Mem | 仅 train-real prototypes；同 gating 层 | 验证真实记忆 |
| D Syn-Mem | 仅 train-synth prototypes；同 gating 层 | 验证渲染记忆泛化 |
| E Joint-Mem (uniform) | 同 combined memory，固定简化非条件融合 | 区分记忆内容与条件 Top-k 的贡献 |
| F SPM-Gated（目标） | real+syn 联合动态 memory、条件 Top-k + gate | 正式方法 |
| G Capacity-NoMem | 相近可训练参数/计算量、相同额外 projection 离线训练预算，但查询记忆被受限无检索信号替换 | 判别是不是参数/附加投影足够 |
| K Fixed-K-Mem（按需） | 同 K 同 query、固定 K-means | 检验动态数量/聚类策略是否值得 |

**一项公平改动**：如果 F 使用 P2 projector 在额外真实/合成增强视图上预训练，而 G 完全没有这一离线训练信息，不能声称 F 的增量完全归因于原型。要让 G 共享同一个冻结 projector/来自 B* 的输入 feature，并匹配尽可能近的 head/fusion 训练范围；如仍有 offline training 不等量，报告混杂。

**零残差/梯度 Gate**：alpha=0 检测/分割数值与 B* 精确对照，容差预先冻结；alpha 初次更新、后续 gate/query (若 trainable)/Wo 梯度合法；prototype buffer SHA 不变；one-to-one/one-to-many 和 COCOeval 完全同口径；同批输入不同工具链读写后输出可复算。若不满足，停止 D 正式训练。

### Stage E：同 checkpoint 的定向干预
对 F 完整 checkpoint 固定所有 backbone/head 参数：
1. α=0、冻结 query/randomly permute prototype IDs（注意仅重排表的行不改变全表 Top-k 语义，**不能**将无意义行置乱当「破坏记忆」）；
2. 使用等范数、相似度分布近似但**错误 key↔value 对应**的 memory（或从其他无关类别/合成噪声库 replacement），在保持输出残差 norm 大致相当下测试内容作用；
3. 仅 real、仅 syn、仅 fg、仅 bg、统一 r 与条件 top-k；
4. 打乱检索的空间对应位置，保留相近残差范数，检验局部相关性；
5. 源 camera/style 扰动下原型激活与 AP 的稳定性，拒绝只展示最漂亮的热图；
6. 控制干预导致 out-of-distribution 分布变化，联合解读可重训 G 控制，而不依赖单次消融作为全部因果证据。

## 3. 哪些量一定要冻在 Run 前

- **版本**：YOLO26-Seg model size、Ultralytics SHA、head/Segment26 API、one-to-one 推理、mask decoder、NMS/conf、input 640、torch/CUDA、weights SHA；
- **数据**：train/dev/test group list SHA、真实帧清单和 frame mask 修订、每 render geometry scene id+view id、real texture donor patch 集合、全部相机 profile SHA；
- **训练**：B arm 的 real/syn sampler、每步实际样本数、seed、epoch/step、优化器、学习率 schedule、EMA、checkpoint 挑选；D stage 的 projector pretraining/info match；
- **原型**：d=128、投影策略、每域每视频 token cap、lambda、最大实际 K 计算预算、prune/merge、Top-k、α/gate 与 pre-trained freeze 名单；
- **评价**：Mask mAP50–95 normal output、bbox AP 独立评估、接触/遮挡/小目标 bin 分母与确定规则、推理资源；簇 bootstrap、置信区间单位；
- **决策**：最低有价值 AP 差、可容忍正常样本/Box 损伤、FPS/显存护栏、positive/negative/inconclusive 对应唯一动作；
- **Run**：真实命令、主机、GPU、stage 和输出产物，以运行环境 run.json/日志为事实来源，而不以本设计草案推定执行。

## 4. 频域实验的唯一优先执行顺序

1. P0 数据簇审计 + 可见 Mask + 当前 YOLO26-Seg 源码/权重确认，确定 train-only texture donor 范围；
2. 真实 profile + texture bank 和 patch 素材审核；若来源/纯度不成立，不能接着做 V3；
3. **64–128 个** Blender 几何 smoke，先修复 visible/amodal、Mask RLE、脚部接触与镜头 warp；
4. 对 **100–200 个 QA scene**做 V0/V2/V3 paired，可视化和频谱/边界标签不变审计；
5. 有训练许可且冻结预算之后做 **600 scene** B stage 的小样本短程比较（样本数是建议设计参数，非当前正在执行）；先决定高频迁移是否值得更大样本；
6. 若 B* 稳定，做 projector/原型离线可行性 → P3 zero-alpha smoke → 小规模 F vs G；再考虑正式 3000 synthetic 的多 seed 实验和 P4 注入。

**唯一当前研究下一步：** 做第 1 项的只读审计和 train-only real-domain profile 数据合同；不直接渲染 3000、训练 7 个模型或开启远程 GPU。

## 5. 机制的阳性/阴性/信息不足与论文范围

- H-GEO：在真实 camera/pen split 的 Mask AP 与接触子集上，D-C0 > D-U 且对应样本条件差异、控制匹配已解释；若两者等效，数据条件建模只是工程选项。
- H-HF：同 geometry/labels D-C3 > D-C2；若 D-C3a/b/c 再证明真实高频自身有增量，方可在题目/贡献强调真实频谱；否则按复合相机与频率处理报告。
- H-MEM：在 B*、G 下，有用 AP 增量与来源覆盖、记忆破坏时的定向衰减一起支持；若只比 B* 好但与 G 无差，记忆独立机制证据不足。
- 信息不足：源/簇/Mask 质量不足、临时更换 eval 口径、有效样本太少或梯度不合法，写明证据不足而非伪阴性/伪阳性。
- **论文首选目标**：密集群养猪可见实例分割的场景校准合成与跨域原型条件检索。实际结果可能支持其中 1 项、2 项或 3 项，论文内容据正式结果调整，不预设三区或录用。

**本文件均为待冻结研究设计，所有阈值、样本数、算法对照、成本都需要由 P0/Pilot 证据确定，未产生科研结果。**
