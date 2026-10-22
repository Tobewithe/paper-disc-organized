# 2026-10-23 SPM-YOLO：数据、原型监督、融合训练与验收决策

**研究线：** spm_yoloseg  
**状态：** 讨论记录 / candidate_protocol / not_started；本记录不是训练授权，也不是已验证结论。  
**主协议：** [V4 数据—监督—训练—评价](../../paper/manuscripts/spm_yoloseg/DATA_SUPERVISION_TRAINING_EVALUATION_PROTOCOL_V4_20261023.md)  
**上游：** [V3 优先级和科学决策树](../../paper/manuscripts/spm_yoloseg/PRIORITIES_RISKS_AND_DECISION_TREE_V3.md) · [V2 实施蓝图](../../paper/manuscripts/spm_yoloseg/IMPLEMENTATION_BLUEPRINT_V2.md)

## 本轮已明确的方案决定

1. **先查同类接触线索，后投入大训练。** P0 仍是唯一立即推荐的只读工作；通过后优先申请 frozen P3/P4 C/B/I boundary probe，而非直接投入 10,000 张合成图。
2. **数据三层：** 真实实例 tight/context/contact crop；真实 cutout + 真实背景的受控合成；Blender 几何/可见掩码及真实高频/相机链配对数据。最终评价只用独立真实视频。
3. **监督目标分离：** 前景存在、同类猪-猪接触边界、增强稳定性、邻近实例局部可分辨性以及原型使用平衡。预训练后才通过样本 exemplar 与来源统计解释原型语义；不预定义“头/腿/背”，不要求相邻猪使用互异原型编号。
4. **训练不从头大联合：** 无 memory 数据基线 B* → 冻结上游特征训练/适配 prototype → 冻结 bank/projector → P3 零残差注入、训练检索/门控和原生头 → 在独立增益成立时才做低学习率 Neck 联合微调；监测 query-bank 漂移。
5. **不改 YOLO26-Seg 原生 Proto26/掩码系数链路。** 本主线中的 knowledge prototype 与逐图 mask prototype 是两个不同模块，不能用旧系数反解实验当作本线科学结果。
6. **K=32、d=128 仅固定 K pilot 候选。** 保留 V2/V3 的动态 K/合并与剔除方案，后续同检索成本与固定 K 比较；不要把 32 当成容量已知上限。
7. **首先排除竞争解释。** 用同图同标签的 V0/V2/V3 外观对照、相同训练预算的 B* continued、参数与额外监督匹配的普通 G Adapter、同 checkpoint 的 memory 干预，再判断 SPM 原型有无独立贡献。
8. **评价必须分层。** 主指标为独立真实测试 Mask mAP50–95；辅指标包含 Dense/Contact Mask AP75、small AP/AR、Contact Boundary F1、merge/split/miss、微扰稳定性与延迟/显存。Box/Mask AP 分开匹配；单条诊断固定预测实例身份。
9. **预注册建议门槛**（尚未冻结）：Dense Mask AP75 有意义提升（候选 +3 pp）、总体 Mask AP 不明显损伤（候选 -0.5 pp）、大多数 seed/独立视频同方向、且 F 同时优于 B* continued/G 并具有针对性 memory 干预证据。CI 太宽则标 evidence_inconclusive。
10. **现实规模从小到大。** 先数据审计、标签 QA、probe、64–128 Blender 几何 smoke / 小 cutout smoke，再以已授权的真实 dev pilot 决定是否进入 600+ 几何与更大原型库。无明确信息增益不升级复杂度。

## 第一批真正要做的事（均未执行）

- P0：建立真实 split/data/Mask/code/checkpoint/eval manifest 与 GT 接触难例账本。
- P1（须审批训练）：真实 frozen P3/P4 C/B/I 与 RGB 边缘对照 probe。
- P2（经 Gate 后）：建立 crop & train-only patch donor、copy-paste 合成 QA 与 Blender 小规模 geometry smoke。
- P3（须审批训练）：无 memory 同几何外观数据实验，固定 B*。
- P4（须审批训练）：K=32 fixed pilot / 按证据动态 K、真实域适配、冻结可解释 memory。
- P5（须审批训练）：P3 zero-alpha smoke → B* continued / G / Real-Mem / SPM-Gated 的控制实验。

**没有改变** study.json 的 not_started 状态、任何已有旧线 Run/总控或当前 GPU 排期。本文件用于人和 Agent 对齐研究讨论与下一步实现。
