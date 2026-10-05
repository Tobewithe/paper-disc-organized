# 三区域冻结表征诊断：实验完整性审计

**日期**：2026-09-11  
**审计对象**：`experiments/coco_clean_20260911/three_region_probe.py`、冻结协议、主输出、汇总器、结果说明、确定性复核，以及事后 `common/difference` 分解  
**审计员**：新鲜 Codex 审计代理 `/root/audit_three_region`  
**审查类别**：same-family  
**接受状态**：provisional  
**总体结论**：**WARN**  
**致命计算错误**：**未发现**

本结论只表示：在本次有界源代码、收据、结果表和说明文本审查中，没有发现伪造 GT、自归一化分数、幽灵结果、未执行读出器、主表算术错配或把 oracle 直接写成模型性能的情况。它不证明模型改进，不证明 CCL 或系数头的因果作用，也不把同图 GT 辅助读出提升为可部署证据。

## 结论摘要

- 主运行收据为 `COMPLETE`：160 张 train、200 张 val、11,300 条逐折读出；收据中 5 个文件的 SHA-256 全部与下载文件一致。
- 汇总收据为 `COMPLETE`：`target_readouts.csv`、`summary.csv`、`status_summary.csv`、`PAIRED_ANALYSIS.json` 的 SHA-256 全部一致。
- 565 个合格的“目标方向 × 支持域”记录，每个恰有 2 个空间折和 10 个读出，得到 `565 × 2 × 10 = 11,300` 行；不存在重复主键、缺折或缺读出。
- GT 直接来自 COCO dataset annotation，经 `pycocotools.COCO` 和 `annToMask` 栅格化；缓存只提供冻结原型、系数、框、检测和官方 bbox50 匹配映射。未发现以模型输出生成或替代 GT。
- 三类像素在每个折内等量抽样，范围为每类 12–128；所有 565 个成功记录均带 `no_shared_proto_stencil=True`，且运行时断言两折采样像素的双线性原型格点集合无交集。
- ridge 只用一个折拟合均值、标准差和权重，另一折评价；阈值只从拟合折的 own 正样本第 10 百分位取得。未发现使用持出折拟合阈值或标准化。
- `actual_own`、`signed_difference`、`foreground_safe` 是实际冻结响应的确定性函数；`oracle_two_logits`、`oracle_proto32`、坐标读出和标签打乱控制均清楚命名为 oracle/control。结果说明没有把它们写成无需 GT 的方法。
- 主结果可支持条件样本上的表征诊断。它不能支持总体 COCO、盲测泛化、Mask AP/IoU、因果机制、模型训练收益或“系数头是唯一根因”等结论。

## A. Ground Truth Provenance：PASS

`three_region_probe.py:149` 依据 split 直接加载 `data/annotations/instances_{train,val}2017.json`；`three_region_probe.py:54-71` 对 COCO annotation 调用 `annToMask`，并分别处理普通实例与 crowd。`three_region_probe.py:88-90` 将 own 和 paired neighbor 限制为全局 `occupancy == 1` 的独占 GT 像素，将 background 定义为所有普通 GT 之外的像素；crowd、GT 重叠和第三实例像素不进入三类样本。

预测到 GT 的映射来自 `frozen_mechanism_probe.py:93-109` 中 COCO bbox 评价路径的 IoU=0.5、同类别一对一匹配。train 缓存在 `relative_ownership_experiment.py:190-201` 由官方冻结推理输出生成映射；val 缓存在 `relative_ownership_experiment.py:203-205` 读取。选定同类相邻 GT 对的逻辑位于 `three_region_probe.py:54-65`，在检查匹配和像素支持前按固定哈希确定。

主协议记录的 annotation 哈希为：

- train2017：`610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`
- val2017：`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`

它们与 `experiments/coco_clean_20260911/audits/data_source_audit.json:5-20` 的既有 COCO 数据源记录一致。没有发现 self-generated GT、baseline-as-GT 或未标注的 proxy GT。

限定：本地下载包不含完整 annotation JSON 和 360 个输入 `.npz` 的源字节，因此本审计没有在本地再次对每个源输入独立算 SHA-256；审计核对的是运行内嵌哈希、事后完整重放断言和既有数据源审计链。此限制不改变“代码没有从模型输出构造 GT”的判断。

## B. Score Normalization：PASS，附协议措辞警告

`three_region_probe.py:25-26` 直接计算带 0.5 tie credit 的经验 AUC；没有除以模型自身最大值、均值或其他预测统计量。`three_region_probe.py:29-33` 的 ridge 标准化只使用拟合折均值和标准差，持出折复用这些统计量。`three_region_probe.py:121-130` 的阈值只来自拟合折 own 样本；原始零阈值指标和训练阈值指标均保留在结果表。

需要修正未来协议中的两处表述：

1. `three_region_protocol.json:13` 的 “Balanced own(+1) vs neighbor/background(-1)” 若按二元标签理解并不准确。实现是 own、neighbor、background 三类各取相同像素数，但 ridge 标签比例为 own 正类 1/3、合并负类 2/3（`three_region_probe.py:43`），不是二元 1:1 平衡。该实现对所有 oracle 一致，结果没有因此造假；未来应写成“三类等量抽样，二元标签比例 1:2”。
2. `three_region_protocol.json:14` 的“90% own training coverage”是目标值。实现用第 10 百分位和严格 `>`（`three_region_probe.py:122,127`）；受离散样本和并列值影响，逐折不保证恰好 90%。主数据各读出的平均训练覆盖约 89.4%–89.6%，极端记录可更低。结果表完整保留实际 `train_coverage`，所以应称“10th-percentile train threshold，并报告实测覆盖”，不应称精确 90% 门槛。

`sampled_precision` 只是在三类等量抽样像素上的诊断精度，不是自然类先验下的 mask precision；协议和结果说明已作此限定。

## C. Result Existence and Arithmetic：PASS（主结果）；smoke 局部未覆盖

主结果文件、字段和收据均存在。确定性核对得到：

- `statuses.csv` 共 993 行。train：4 图无 GT 对、14 图双匹配失败、383 个成功目标域、185 个像素不足目标域；val：124 图无 GT 对、7 图双匹配失败、182 个成功目标域、94 个像素不足目标域。
- 成功目标域为 `383 + 182 = 565`。`readouts.csv` 有 5,650 个“目标域 × 读出”键，每键恰有 fold 0/1；每个“目标域 × fold”恰有 10 个读出。
- `target_readouts.csv` 有 6,215 行，等于 5,650 个双折均值加 565 个三次 shuffle 的均值行；无重复键。
- `summary.csv` 有 132 行且无重复键；从目标表重算每一行的目标数、图像数和 11 个指标均匹配。
- `PAIRED_ANALYSIS.json` 有 1,188 个配对对比、22 个支持域点估计和 8 个资格计数；对比点估计与逐目标差值匹配。
- `LOCAL_VERIFICATION.json` 报告重算全部汇总与配对均值，并重算 40 个 val/high 关键 bootstrap 区间；最大算术误差 `3.33e-16`，最大区间误差 `7.11e-15` 个百分点。审计员另以独立 PowerShell 聚合核对了行数、键唯一性、折完整性、指标范围、混合 AUC 恒等式、收据哈希和全体 val 对比点估计。

`THREE_REGION_RESULTS_20260911.md` 的关键数字和边界与结果表一致，包括高 ICI 两域的 AUC 取舍、oracle 覆盖下降、未校正点态区间、条件分母和非盲测说明。没有发现引用不存在字段或与结果表冲突的数字。

未完整覆盖 `diagnostics/three_region_smoke_v2_20260911`：审计早期只确认远端存在 `COMPLETE.json` 及结果文件并记录其文件哈希，但 smoke v2 未下载到本地，因最终范围明确禁止继续网络读取，本审计没有逐行审查该 smoke 的协议和 CSV。主结果不依赖 smoke 数值作结论，因此主结果完整性仍为 PASS；smoke 子项状态为 **NOT FULLY REVIEWED**。

## D. Active Code and Dead Code：PASS，附 schema 注意事项

`scores()` 在 `three_region_probe.py:36-49` 定义全部 10 个实际读出；`three_region_probe.py:117-131` 对两个交叉折逐一调用并写行。输出中每个成功目标域都出现全部读出，说明没有“定义但未执行”的主指标。

`summarize_three_region.py:14-16` 在汇总前验证主收据；`summarize_three_region.py:20-27` 强制每键恰有两折；`summarize_three_region.py:30-36` 构造三次 shuffle 的均值；`summarize_three_region.py:43-70` 生成按图像聚类的 2,000 次配对 bootstrap；`summarize_three_region.py:80-84` 写出表和汇总收据。这些路径均有对应产物。

schema 注意事项：`three_region_probe.py:129-130` 为所有 oracle/shuffle 也写了 `zero_*` 字段，而预先协议只明确把零阈值 mask 指标用于 `actual_own` 和 `foreground_safe`。这些额外字段不是死代码，但 oracle 分数的零点不是部署掩码阈值；不得把 oracle/shuffle 的 `zero_*` 当作实际 mask 结果。当前结果说明只对相关实际读出解释零阈值集合关系，未越界。

## E. Scope Assessment：WARN

实际范围远窄于“200 张 val”这个表面分母：

- val 正常支持域只有 75 个目标、48 张图；外扩域只有 107 个目标、57 张图。
- val 高 ICI 子组分别只有 30 个目标/26 张图和 46 个目标/34 张图。
- 200 张 val 中 124 张没有协议要求的同类 GT 邻接对，7 张选定对未双匹配；另有 42 张至少出现像素不足记录。
- 两域只有 74 个共同合格目标、47 张图；两域绝对均值不是同一队列的外扩曲线。`PAIRED_ANALYSIS.json` 对共同目标只给域差点估计，没有 bootstrap 区间。
- val 图来自已反复探索的 COCO val 子池，不是盲测集；“train/val”不代表 oracle 跨图训练/测试，因为每个 oracle 都在同一图的一折拟合、另一折评价。
- 没有网络训练、独立随机种子实验或官方 Mask AP/IoU。三次 shuffle 是标签打乱控制，不是方法的三次训练种子。
- 所有置信区间是逐项 2,000 次图像簇 bootstrap，未做多重比较校正；高/低 ICI 对比是未调整探索性关联。

空间泄漏控制与其边界：`three_region_probe.py:85-87` 按相对目标框的 4×4 棋盘为原型格点赋折，`three_region_probe.py:102-113` 只保留四个双线性 stencil 格点同折的像素并断言跨折格点无交集。该控制足以支持“没有共享插值格点”，但同一原型图的深层感受野、全局网络计算、相同实例系数和同图上下文仍共享；协议与结果说明已经披露，不能称为完全像素独立或跨图泛化。

## F. Evaluation Type：PASS（分类完成）

主评价类型为 **real_gt**，因为标签和区域来自 dataset-provided COCO GT。应同时保留以下子类型：

- `actual_own`、`signed_difference`、`foreground_safe`：冻结模型响应在真实 GT 像素上的阈值无关排序诊断；阈值版本为 train-fold-only 的同图空间交叉拟合诊断。
- `oracle_two_logits`、`oracle_proto32`、`oracle_coordinate*`：**real-GT-assisted same-image oracle**。它们使用评价图自身 GT 标签拟合，不能归类为盲测模型表现。
- shuffle 读出：real-GT 样本上的 negative control。
- `common_difference_20260911`：**post-hoc real-GT descriptive decomposition**，不是独立确认。

## 平衡采样、折划分与阈值专项核查

每个折先分别建立 own、neighbor、background 候选池，取三者最小值并截到 128；小于 12 则整个目标域失败（`three_region_probe.py:102-109`）。565 个成功记录的 fold0/fold1 计数均在 12–128，主表 `eval_pixels_per_class` 同样在该范围，无越界或非有限指标。

随机种子由 image、target、domain、fold 和 shuffle index 构成。对 565 个成功目标域、2 折、3 个 shuffle 得到的 3,390 个 seed 值无重复。train 与 val 图像 ID 集合无交集。

每折阈值只看相反评价折之外的 own 训练像素。持出覆盖漂移被完整写入表，并且是解释 oracle 时的实质限制。全体 val 中：

- 正常域 `oracle_proto32` 的平均训练覆盖约 89.38%，持出覆盖 79.74%，漂移约 -9.64 个百分点；相对 `actual_own` 的持出覆盖差为 -7.54 pp，95% 点态区间 [-10.14, -5.16]。
- 外扩域 `oracle_proto32` 的平均训练覆盖约 89.60%，持出覆盖 78.32%，漂移约 -11.28 pp；相对 `actual_own` 的持出覆盖差为 -8.19 pp，区间 [-11.34, -5.47]。
- `oracle_coordinate32` 的持出覆盖更不稳定，正常/外扩全体 val 平均约 56.25%/48.88%；因此坐标与原型的阈值指标不能当作同覆盖公平对比。

AUC 本身是排序指标，不依赖阈值；所以 `oracle_proto32` 的 AUC 提升可以作为“该条件样本中的图内 GT 线性可读性”证据，但其低 FPR 不能写成相同覆盖下的泄漏改善。

## Ridge 比较的公平性边界

所有 ridge 读出共享相同像素、折、train-only 标准化、固定 `lambda=0.1` 和 centered-label intercept，这是有效的受控比较。仍需保留以下限制：

- `proto32` 与 `coordinate32` 同为 32 维，但特征族、频谱、归纳偏置和有效容量不同；“同维”不等于同函数容量。
- `two_logits` 只有 2 维，且是 `proto32` 的两个固定线性投影。`proto32` 比它高维且包含它的信息；优势不能单独定位为“系数头唯一失败”，也可能反映额外自由度。
- 固定 ridge 强度在维度不同的特征空间中不等价于相同有效复杂度；本实验没有独立数据选择正则化或做容量匹配曲线。
- 空间 checkerboard 对高频坐标 Fourier 特征的外推难度与对原型特征不必相同。坐标控制未复制原型结果，只能排除这一个具体坐标基线，不能彻底排除几何或组成混杂。

## 可支持与不可支持的结论

### 有条件支持

1. **差分改善 own-vs-neighbor 排序时会损伤 own-vs-background 排序。** 在合格 val 目标上，两域全体队列的 `signed_difference - actual_own` 邻居 AUC 分别约 +2.98/+3.01 pp，而背景 AUC约 -11.29/-10.15 pp，混合 AUC约 -4.16/-3.57 pp；结果说明对高 ICI 小队列使用了相应配对区间并正确披露正常域邻居区间跨 0。
2. **冻结原型在同图 GT 辅助线性读出下仍含额外的三类排序信息。** 全体 val 中 `oracle_proto32 - actual_own` 混合 AUC为正常域 +2.81 pp [0.73, 5.14]、外扩域 +2.39 pp [0.73, 4.25]。该结论必须同时写明图内 GT oracle 和覆盖下降。
3. **两响应 oracle 提供有限的图内可读性线索。** 全体 val 中混合 AUC相对 actual 为正常域 +1.47 pp [0.01, 3.32]、外扩域 +2.11 pp [0.77, 3.80]；这不是跨图规则或方法收益。
4. **`foreground_safe=min(actual,difference)` 的零阈值正像素是 actual 的子集。** 这是代数/确定性集合关系；结果也显示它会删除 own 像素，不能仅凭“不新增前景”称为改进。

### 不支持

1. 任何官方 Mask AP、Mask IoU、全图泄漏率或 end-to-end 模型改进。
2. 无需测试 GT 的可部署读出器，或跨图泛化能力。
3. “系数头是唯一根因”“CCL 已被证实”“原型 oracle 是 AP 上界”或纯因果机制。
4. 收益专属于高 ICI/拥挤实例；当前低组也出现相近或更大的部分变化，且没有类别、尺度、距离组成控制。
5. 代表性 COCO 总体结论、稳健/广泛/全面评价或独立盲测确认。
6. 相同覆盖下 `oracle_proto32` 的 FPR 改善。

## 事后 common/difference 分解：WARN / descriptive only

`common_difference_probe.py:23-34` 明确标记该分析在主结果后提出，并继承完全相同的图、匹配、样本、折和缓存。`common_difference_probe.py:36-48` 只计算 `m=(z_i+z_j)/2`、`d=(z_i-z_j)/2` 等无参数代数组合；`common_difference_probe.py:50-76` 重放主程序，要求 11,300 条原指标逐项零差、输入哈希完全一致后才写出 4,520 条补充读出。

当前收据显示 `replay_rows_checked=11300`、`replay_max_abs=0.0`、`input_hashes_match=true`；补充读出最大代数残差为 `3.55e-15`，所有比例/AUC 位于 [0,1]。补充收据和汇总收据哈希均匹配。

它可以描述：在相同条件样本上，common response 对 own/neighbor 合并相对背景仍有排序信息，而 half-difference 更偏向 own-vs-neighbor；它不能证明 `m` 是语义 objectness latent，不能独立确认主假说，不能作为新方法或因果证据。`THREE_REGION_RESULTS_20260911.md:65-75` 对这一边界表述正确。

## 结果说明审查

`refine-logs/coco-structure/THREE_REGION_RESULTS_20260911.md` 的主张边界与原始证据一致：

- 开头明确称“机制线索”，拒绝宣布有效方法或 CCL 因果机制（第 5 行）。
- 明确说明 val 已被探索、条件分母、空间折限制、非官方解码、非 AP/IoU/全图泄漏率（第 9–27 行）。
- 将 actual、difference、foreground-safe 与 GT 辅助 oracle 分开，并同时报告邻居/背景取舍与点态未校正区间（第 29–53 行）。
- 将 common/difference 标成事后解释并拒绝语义过度解释（第 55–75 行）。
- 对 proto oracle 同时报告覆盖下降，并拒绝“相同覆盖泄漏改善”和“唯一根因”（第 77–99 行）。
- 结尾只提出下一步需要新的 train-only、跨图、同覆盖和官方任务指标验证，继续暂停正式训练（第 101–109 行）。

因此，当前结果说明可作为项目内部、artifact-only 的探索报告。若未来用于论文，仍必须逐条经过原始结果到 claim 的独立审计，并补上跨图、无评价 GT 的预声明实验。

## 未覆盖范围

本审计明确没有覆盖：

- 本地逐字节重哈希完整 COCO annotation JSON 和 360 个远端输入缓存；这些源字节未包含在下载审计包中。
- `three_region_smoke_v2_20260911` 的本地逐行内容审查；只在审计早期确认远端完成文件存在，最终按指令不再访问网络。
- 通过独立环境重新执行 CUDA 主程序。运行真实性由脚本哈希、不可覆盖式输出目录、完成收据、两次重放和结果一致性支持，但本审计不是现场执行见证。
- 独立运行 `verify_and_present_three_region.py`；审计读取了其源代码和回执，并另行重算关键关系，但本地 shell 未暴露 `conda` 命令。
- 原始图像语义的人工抽查、COCO 官方 Mask API 任务评价、跨图训练、随机种子训练、超参数稳健性、多重比较校正及外部复现。
- 跨家族 reviewer。当前结论是 same-family/provisional，不可标记为 cross-family accepted。

## Action Items

1. 冻结本轮结果，不因 oracle AUC 启动或宣传模型训练收益。
2. 后续协议将 ridge 标签写成“三类等量、二元 1:2”，将门槛写成“第 10 百分位并报告实测 train coverage”。
3. 只把 oracle AUC用于形成下一次假说；下一次方法实验须在独立 train 开发图拟合统一规则，在新评价图中完全不接触 GT，并同时报告同覆盖的 own coverage、neighbor/background FPR 和官方 Mask AP/拥挤子指标。
4. 若论文引用本轮，限定为“matched-pair、three-region-eligible、same-image spatial cross-fit diagnostic”，并保留所有分母、覆盖漂移和未校正区间。

## 审计文件哈希锚点

- `three_region_protocol.json`：`0c96bd0e206835c12dbcb658c78ee563b92f937df1bf944535f160d59b43866e`
- `three_region_probe.py`：`05d797ffb4d0c0d65682295f7e804d6a915f686612fc86db760cdb35ba7b0ff8`
- `summarize_three_region.py`：`74466f957af34606c261a7b4380dcc5febf326f5aab036c45528e36abf32b9b4`
- main `readouts.csv`：`ebe8c7203c245d71dff2583510f3832f971cb7cc215ea5532ad4ae8ac683d9ab`
- main `statuses.csv`：`8f070cc8cb17755dcb89bb951a7242548e55b9080e71e55c3db026f4d3b9f5a0`
- main `PAIRED_ANALYSIS.json`：`95966ae2ee87234db19b0879da1f5b3b9338449a86fc0e6f72ea76499cc0275f`
- result report：`754b7cf5856f0e5be2dfedc6389deee7e3772dad58a522596114e90aa119555b`
- common/difference script：`713278ea492bae49648526896ad3f65eb95282c14e96854cf2ade368aa5377c7`
- local verification receipt：`e78b50b6aff1c94693d2f92a89baf79b06d912777e35acbef3199c0ced5c51f3`

审计轨迹：`.aris/traces/experiment-audit/2026-09-11_run02/`。
