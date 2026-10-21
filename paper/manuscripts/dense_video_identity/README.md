# 密集视频个体跟踪与遮挡恢复

**2026-10-19 后续研究路由：**用户已将毕业论文核心方法推进为独立的[通用遮挡表征预训练主线](../occlusion_representation_pretraining/README.md)。本线保留全部冻结实验、阴性结果、模型和展示，作为前期证据及下游验证来源。后续任务只按[当前总控](../../../project/ACTIVE_PLAN.md)与新线 Study 执行；下文累计记录中的旧“下一重点”不构成新的启动安排。

本线面向毕业论文的小样本原型与通用检测分割跟踪系统，猪和马作为应用/验证类别。
用户 2026-10-18 指定在 `28358lan` 离线笔记本从数据到实验完整实现。

- 已结案实验协议：[可见性学习与真实视频身份恢复迁移](../../../experiments/dense_video_occlusion_transfer_20261019/PROTOCOL.md)
- 真实状态：[study.json](../../../experiments/dense_video_occlusion_transfer_20261019/study.json)
- 工具选择：[场景生成与标签采集选型](../../../experiments/dense_video_synthetic_occlusion_20261019/TOOL_SELECTION.md)
- 方法设计：[毕业论文模型方案](../../毕业论文模型方案_密集视频身份保持与遮挡恢复_20261018.md)
- 实现：[shared/dense_video_identity](../../../shared/dense_video_identity/README.md)

第一轮实验和系统原型已完成。五个完整测试视频共 2,655 帧、41,875 个 GT 实例观测；两个无 GT 现场片段仅作演示。

2026-10-19用户明确授权在服务器继续生成、训练与真实基线/消融。本轮独立Study采用8种带骨骼动物资产，72段场景按资产家族划分，三臂同容量可见性模型各2000步；真实23段固定视频不用于本轮训练，11段历史留出为主要评价。本轮已完成72段生成、三臂各2000步训练及23段×8组真实评价（训练145、评价149、补充裁决150）。11段主要评价FULL IDF1为67.807643，较C +0.262926（CI下界0），较BoT-SORT −4.171077；只有duck_3改善0.647179，其余10段不变。打乱权重不更差，当前机制及稳定优越性未成立，停止本配置扩训。1660项回传SHA、本机指标/区间与历史基线复核通过；图表和两段视频已完成，服务器无剩余作业。详见[完整报告](../../../experiments/dense_video_occlusion_transfer_20261019/REPORT.md)与[离线展示](../../presentations/可见性学习与真实视频身份恢复_20261019/index.html)。见[实现合同](../../../experiments/dense_video_occlusion_transfer_20261019/IMPLEMENTATION_CONTRACT.md)和[标签修复](../../../experiments/dense_video_occlusion_transfer_20261019/GEOMETRY_CORRECTION.md)。

2026-10-19已在算力服务器完成Blender原生脚本/Cycles及逐目标几何标签验证。Slurm115—121中，116缺运行库的安装失败保留，其余完成；119单张A6000生成6段×24帧，用时3分48秒。288条个体观测含可见167、部分遮挡85、完全遮挡28、出视野8；6,912点几何核验、399文件回传及532条官方COCO掩码还原通过。4段完全遮挡区间中2段含重新出现。当前保留生成路线，下一重点是多资产/多动作且独立拆分的遮挡—重现训练集。模型训练0步，真实视频效用尚未验证；此前第四轮阴性与冻结协议保留。

- [生成与标签完整报告](../../../experiments/dense_video_synthetic_occlusion_20261019/REPORT.md)
- [生成场景与标签展示](../../presentations/三维遮挡场景与自动标签_20261019/index.html)

2026-10-19第四轮已完成：S0错接AUROC由单体0.5868提升至竞争0.6612、时序竞争0.6905，信息准入通过；随后在笔记本训练CETD-v1及SELF/NOW三组同结构模型，每组12,993参数、800步。4段新视频IDF1为67.403，相对ByteTrack+2.679，但与C完全相同、低于BoT-SORT4.369；校准后全部撤销为0，新增模型效用和机制未成立。15段评价×12组完整输出、3个checkpoint和可视化已回传校验。当前配置关闭，下一重点是事件监督与轨迹级动作效用的对齐；2个错接事件/4个触发时刻的撤销均无正收益，支持补动作结果标签，尚不足估计总体错位比例。未启动新一轮训练。

- [第四轮完整实验报告](../../../experiments/dense_video_evidence_decision_20261019/REPORT.md)
- [实际模型与新颖性边界](../../../experiments/dense_video_evidence_decision_20261019/MODEL_AND_NOVELTY.md)
- [第四轮离线可视化](../../presentations/CETD_竞争反证时序决策实验_20261019/index.html)
完整候选 IDF1 57.54，ByteTrack 63.69，Deep OC-SORT 60.17。候选未过效用门槛；不宣称已经解决遮挡或实现通用动物泛化。

- [完整实验报告](../../../experiments/dense_video_identity_20261018/REPORT.md)
- [展示与运行说明](../../presentations/密集视频个体跟踪_实验与系统_20261018/使用说明.md)
- [工程验收与证据索引](../../../experiments/dense_video_identity_20261018/DELIVERY.md)

第二轮已在同一笔记本完成：外观阶段删除对照、143条可用现场AI复核身份标签、8个GT间断原因核对、CGS/CGS-T/PEA及同容量与已有方法对照。CGS-T在预先指定的斑马1/2上IDF1 43.410→50.628（+7.218），但历史五视频只提高0.046，且低于同容量正确性对照；PEA在斑马3/4确认退化1.461。局部效用保留，稳定优越性与机制独立贡献未建立；当前固定配置不再增训。

第三轮已在笔记本完成19段视频×7组固定对照，新增训练零步。新确认zebra_5、duck_1、duck_2上，完整“确认+回退”IDF1为66.640，相对ByteTrack +1.130、相对BoT-SORT −1.456；仅确认为68.633。回退减少错接持续量，同时拆断正确连接，稳定优越性及回退独立贡献未建立，当前配置关闭。64个GT间断初筛、16例逐帧深入复核；11例物理遮挡补充层、165条沿用并复核官方框的标签，可靠完整遮挡事件为0，完整遮挡恢复目标尚未达到。下一研究重点是区分错误身份与姿态变化的撤销反证，新的训练/阈值搜索尚未启动。第二轮及其它研究线保持原状态。

- [第三轮完整实验报告](../../../experiments/dense_video_reversible_identity_20261019/REPORT.md)
- [第三轮离线可视化](../../presentations/密集视频身份保持_确认与回退实验_20261019/index.html)

- [第二轮展示](../../presentations/密集视频身份保持_第二轮实验_20261019/index.html)
- [第二轮使用说明](../../presentations/密集视频身份保持_第二轮实验_20261019/使用说明.md)
- [单帧收益协议](../../../experiments/dense_video_association_v2_20261018/METHOD_PROTOCOL.md)、[轨迹收益协议](../../../experiments/dense_video_association_v2_20261018/TRAJECTORY_PROTOCOL.md)、[保护式补接协议](../../../experiments/dense_video_association_v2_20261018/PROTECTED_EXTENSION_PROTOCOL.md)
