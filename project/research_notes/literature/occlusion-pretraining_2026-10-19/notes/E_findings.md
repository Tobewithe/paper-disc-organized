# E：开放仓储与数据集标签语义

本桶检索了公开数据仓储和官方下载入口，重点不是“是否有视频”，而是“是否有可用于当前主张的遮挡真值”。规范化记录在 `raw/E_open_repositories.json`，查询和失败记录在 `raw/E_open_repositories.status.json`。

| 资源 | 能可靠拿到的标签 | 不能据此声称的能力 | 当前用途 |
|---|---|---|---|
| SAIL-VOS | 每帧 RGB、modal instance-ID、同 ID 的 per-object amodal mask | 真实动物域泛化；无限制再分发 | 合成 amodal 视频格式和损失的最强参照 |
| MOVi（Kubric challenge 资源关系，**不计为独立论文**） | modal instance-ID、depth、flow、normal、object coordinate | amodal instance mask、动物重现 | flow/几何输出 schema 参照 |
| PointOdyssey | 点轨迹、point visibility、modal mask、depth、相机 | 个体 amodal mask 或身份恢复 | 运动/可见性配对参考 |
| TAP-Vid | 人工或模拟 point trajectory+occlusion flag；first/strided query | 实例身份、amodal mask、物理遮挡/出画因果拆分 | 点级可见性评估参考 |
| OVIS | 真实视频 modal instance mask、粗粒度遮挡等级 | full-amodal mask 或 fully-hidden ground truth | 检测+分割+跟踪遮挡测试 |
| MOSE | 真实 modal mask、复杂场景、disappearance/reappearance 统计 | 完整物理遮挡和出画的金标准拆分 | 真实时序鲁棒性/半监督VOS参考 |
| TAO-Amodal | amodal box、visibility、uncertainty，覆盖 fully invisible/out-of-frame/occluded | amodal pixel mask | 真实视频的 visibility/box/out-of-view 评估候选 |
| TABE-51 | 作者主张为 amodal-video benchmark，含完整遮挡 mask | 未审全文前，不把它当成物理遮挡金标准 | 未来 zero-shot amodal 参照，非当前训练来源 |

## 对当前方案的结论

已有公开资源没有同时给出“密集同类动物、逐实例 amodal 像素、遮挡者归属、真实重现身份、出画区别、可自由复用”的完整监督。项目现有 Blender 数据已具备 `instance_id`、`amodal_i`、可见比例、`truncation` 与 `occluder_pixels`，可自动导出 paired visible/amodal、遮挡强度、遮挡者归属与连续 ID 重现监督；不需要人工标注合成样本，也不需要为本研究重建资产或整库。SAIL-VOS 可用来对齐数据格式和证明这类 supervision 有先例，PointOdyssey/MOVi 用来对齐 flow/visibility 输出，TAO-Amodal 用来定义真实视频的 box/visibility 评估。

真实验证不应把 MOSE/OVIS 的普通连续 modal mask直接当作“完整遮挡后恢复”金标准。OVIS 有粗遮挡级别；MOSE 有消失-重现语境；两者都尚不足以区分物理遮挡、漏标、出画、镜头切换或标注缺失。若使用 TAO-Amodal，评估应分别报告 visible bbox、amodal bbox、visibility 和 `amodal_is_uncertain`，不把不确定标签并入强监督。

当前最佳可检验主张仍是：三维几何产生的 paired visible/amodal/occluder-depth/reappearance 标注预训练，是否能在严格的真实 modal 遮挡切片以及可获得的 TAO-Amodal box/visibility切片上提高表现。公开数据适合作为外部评价或预训练对照，不能替代项目自身的严格真值。

