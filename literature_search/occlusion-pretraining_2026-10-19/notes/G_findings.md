# G：官方网页与代码证据（工程选型）

检索范围：Kubric、BlenderProc、Objaverse、TCOW、TAPIR/TAPNext、CoTracker3、SAM 2、SAMEO、SyDog-Video 和 NVIDIA Cosmos 官方网页或官方代码。镜像、博客和无关仓库没有作为独立证据计数。原始规范化记录见 `raw/G_official_web_code.json`；状态和未解决缺口见 `raw/G_official_web_code.status.json`。

## 可直接支撑的事实

- [Kubric](https://github.com/google-research/kubric) 的 README 明确给出：Blender+PyBullet、实例分割、depth、optical flow、物理交互和可控组合泛化拆分；实现位于 `kubric/renderer/blender.py`，默认渲染层列有 `segmentation`、`depth`、`forward_flow`、`backward_flow`、`normal` 与 `object_coordinates`。代码仓库为 Apache-2.0，但输入资产仍各自受许可约束。
- [BlenderProc](https://github.com/DLR-RM/BlenderProc) 已封装姿态采样、物理碰撞、材质/光照/相机随机化、RGB/depth/normal/segmentation/flow、COCO/BOP 写出。官方 API 路径为 `blenderproc.renderer.render_optical_flow()`、`blenderproc.renderer.enable_segmentation_output()` 和 `blenderproc.writer`。它适合加速新数据工程，不自动生成“去掉遮挡物后的同一目标”真值。
- 当前项目已有经几何验证的原生 Blender/Cycles 脚本、资产、amodal overscan 和可见标签。就当前目标而言，**不建议为了换生成器迁移到 Kubric 或 BlenderProc**：它们是标注能力的证据和可选实现参考，当前最小风险方案是扩展现有 Blender 代码，增加标准化输出和场景随机化。
- [Objaverse 1.0](https://objaverse.allenai.org/objaverse-1.0/) 有 800K+ 对象、44K+动画对象与 rigged characters，但页面同时说明对象来自多源；使用时必须读取每个 UID 的许可元数据。项目当前已有场景与资产，因此不应把检索、导入或人工筛选公开资产列为训练前置；只有未来主动扩展外部资产时，才需要逐项核验许可、rig 与动作可用性。
- [SyDog-Video](https://cvssp.org/data/SyDogVideo/) 是可复用的合成动物视频参照：每版本 500×175 帧，提供框、33关键点、分割和若干视觉条件版本。官方说明没有 amodal、深度、前后遮挡关系或 reappearance 标签；可用于外部定性/pose-style 预训练基线，不能充当本项目遮挡预训练的核心监督。

## 视频模型：哪些能力可借，哪些不该直接套用

- [TCOW](https://github.com/basilevh/tcow) 最接近“目标进入遮挡—持续不可见—再出现”的训练语义：`gen_kubric/` 使用 Kubric 衍生的 X-ray 标签，训练命令为 30帧、320×240、causal attention。代价也高：官方 README 记录 2×RTX A6000、每卡约38GB VRAM、batch size=1，并要求适配自定义数据加载器。因此它适合借用**事件标签、因果窗口和评估定义**，不适合当前作为可直接部署到 YOLO 的轻量模块。
- [TAPIR/TAPNext](https://github.com/google-deepmind/tapnet) 和 [CoTracker3](https://github.com/facebookresearch/co-tracker) 都输出点轨迹/可见性，适合做“持续可观测区域”或合成数据 sanity check；它们并不解决实例级 amodal mask 或密集同类身份竞争。TAPIR 官方代码有 causal online checkpoint 和 Kubric 训练说明；CoTracker3 同时发布 online/offline checkpoint、pseudo-label pipeline 和 Kubric-only checkpoint。把它们直接当作 YOLO 目标 ID 恢复器会把点级可见性错误地提升为个体身份结论。
- [SAM 2](https://github.com/facebookresearch/sam2) 的 `sam2/sam2_video_predictor.py` / `training/model/sam2.py` 体现“image encoder + memory attention + memory encoder”的视频记忆接口。SA-V 是大规模 modal masklet 数据（51K视频、643K masklet，CC BY 4.0），不是几何 amodal 真值。它更适合作为强 modal-video encoder/teacher 候选，而非本项目遮挡标签的替代品。
- [SAMEO](https://weient.github.io/sameo.github.io/) 已经覆盖“不同前端检测器输出框 + SAM式 amodal decoder”的模块化接口，也以约300K合成 Amodal-LVIS 扩充训练。它是最直接的**创新边界**：本项目不能把“用YOLO框接 amodal decoder”单独写成创新。可区别的可检验点只能是：几何可控的同一实例遮挡配对、可见/遮挡/不确定性分解，以及是否给 YOLO 检测分割产生独立可复核的迁移增益。

## 生成式外观增强的严格边界

[NVIDIA Cosmos Transfer 的官方 X-Mobility 例子](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer1/inference-x-mobility/inference.html) 用 RGB+segmentation video 作为控制，segmentation control weight 设为1.0，并直接复制原始 semantic labels。官方**主张**这会精确保留几何与运动；该网页的原始/增强1:1、共520K帧、两个100 epoch阶段、8×H100，是后续 X-Mobility 世界模型/动作策略训练配置，**不是** Transfer 生成器本身的已记录成本。

这不足以让我们把生成帧的原始实例/遮挡/flow 标签当作已证实的像素级真值：该网页没有发布面向动物实例的 per-pixel IoU、ID 一致性、depth order、flow 误差或遮挡事件保持审计。因此 Cosmos 只能列为第二阶段的**视觉域适配候选**。一旦试用，必须为每帧复验：控制 mask 与生成画面边缘一致性、实例框/ID 是否仍可定位、前后遮挡事件是否保持；未通过者不能继承几何标签。当前数据主源仍应是可渲染、可重算真值的三维场景。

## 推荐的可实现数据与训练栈（设计建议，尚未冻结为方法）

### 数据：保持几何真值为源头

1. **不重建现有数据。**当前样本已有 `instance_id`、`amodal_i`、可见比例、`truncation` 与 `occluder_pixels`；应直接自动导出 visible/amodal 对、遮挡强度、遮挡者像素归属，以及由连续 `instance_id` 得到的重现监督。它们是三维几何真值，不需要人工补标合成样本。
2. 先将这些既有字段写入版本化 manifest，并按现有资产家族、场景模板和事件参数组切分，避免随机帧泄漏。只有后续诊断显示某种遮挡事件或视觉域覆盖不足时，才有理由在当前 Blender/Cycles 生成器中补充该事件；不因本方案而重建整库或改用另一生成器。
3. 对既有与后续样本做自动审计：visible mask 与前景 `instance_id` 一致；`visible ≤ amodal_i`；`occluder_pixels` 能解释被挡区域；无外部遮挡时 visible=amodal；同一 `instance_id` 的流/pose 连续。当前72段的像素检查脚本可复用，并把检查扩到以上既有字段。

### 模型：先空间遮挡表征，再可选因果时间头

最小可迁移骨干应是对象条件化单帧模型：`图像 + YOLO框/可见mask → FPN特征 → visible-mask、occluder/unknown-mask、visible ratio、confidence/uncertainty、对象embedding`。它的核心输出应是**可信可见区域与不确定性**，而不是强行猜出被挡住区域的纹理。

视频版只在上述空间模型后接小型 causal memory（GRU或小型 causal Transformer），输入过去 K 帧的 ROI/box/可见性/运动特征，输出 `partial/full/out/uncertain` 与 reappearance candidate score。训练时不得给它未来帧或 GT 框；完全遮挡时允许“证据不足/不确定”，避免把不存在或出画一概叫遮挡。

训练目标可分开：空间分支监督可见mask、遮挡比例、in-frame amodal（仅作辅助）；配对对比/匹配损失让同一实例在遮挡变化后保留区分其他同类的表征；视频头监督当前可见状态、重现候选的正负对应和拒绝相似竞争者。这样才与现有FULL“只调节历史embedding权重”的失败形成可检验差异。

### 迁移到 YOLO：先做可证伪的接口

第一阶段应冻结 YOLO 检测器，用其框/可见mask作为输入，验证遮挡模型在真实图像上的状态、比例、空间定位和置信度校准。第二阶段才比较同初始化/同数据/同预算的三臂：普通YOLO训练、普通遮挡增强、遮挡预训练的特征蒸馏或轻量辅助头。主结果必须按遮挡等级报告检测/可见分割收益和无遮挡损伤；不把成功的“分析器”说成 YOLO 改善。

## 当前优先判断

我建议研究主张聚焦为：**几何可验证的可见性—遮挡—重现配对监督，能否训练出在真实视频上改善遮挡条件下检测/分割或身份对应的对象条件化表征。** 这比训练一个独立遮挡四分类器更可能迁移，也避开 SAMEO 已覆盖的“检测器框接 amodal decoder”表述。

唯一下一步文献/设计动作：对已有数据自动导出上述监督字段及按资产/场景/事件组切分的 manifest，并记录真实评价所缺的标签；之后才有条件冻结模型与训练消融。当前没有启动新训练。

