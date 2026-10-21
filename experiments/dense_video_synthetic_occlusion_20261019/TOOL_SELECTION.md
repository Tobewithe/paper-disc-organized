# 场景生成与标签采集选型

2026-10-19。本次选择 **Blender原生Python + Cycles/OptiX + 定制几何标签导出**。真实执行版本为4.2.1，单卡小规模预检；未进行以下工具的速度对比。

| 路线 | 数据/标签能力 | 本项目判断 |
|---|---|---|
| Blender原生脚本 | 可控制模型、骨骼/部件、轨迹、相机和遮挡物；稳定身份与完整投影标签由本项目实现 | 当前首选。便于在既有驱动与Slurm上单独部署，不需要改变系统环境 |
| BlenderProc 2.8.0 | 官方支持图像/序列、深度/分割与COCO/BOP；提供随机化和资产加载接口 | 值得在资产扩展阶段使用。当前需要的逐目标时序遮挡关系仍需定制，故未把依赖它作为预检前提 |
| Kubric / TCOW | 专门面向多目标时序、物理模拟、丰富监督；TCOW有遮挡者标注与真实视频评价 | 采用其标签思路并列为后续研究参照。官方Kubric入门仍列Blender2.93和容器路径，当前不先迁入整套环境 |
| Isaac Sim / Replicator | 面向复杂场景与传感器数据生成，生态完整 | 暂不作为首选。当前latest文档列Ubuntu/Windows及更新驱动；服务器实际为openEuler和550.163.01，兼容成本高于当前目标 |
| 生成式视频模型 | 可扩展外观、场景与运动素材 | 后续可评估。身份、遮挡起止、几何掩码等标签必须逐项验证，不能由提示词自动认定 |

## 标签设计

每个scene/track/frame记录稳定ID、RGB、可见实例图、独立目标完整投影、相机参数与物体位姿。用原画幅和同像素尺度扩展画幅分离外部遮挡与视野截断。将目标投影中被其它实体占据的像素，分配到具体遮挡者ID，形成可审计关系。

后续训练可直接使用可见/部分遮挡/完全遮挡/出视野状态和事件边界；完整投影、真深度、真ID只能作监督。生成标签不能证明真实视频身份恢复有效。

本轮两个程序化四足体仅验证管线。正式数据阶段需要增加具有合理姿态变化的资产、相似个体、相机与材质变化，并按资产家族/场景种子拆分，防止相邻帧或同一资产变体泄漏。

## 主要依据

- [BlenderProc官方功能与安装](https://github.com/DLR-RM/BlenderProc)：模型/材质/相机、图像与序列输出。
- [BlenderProc渲染文档](https://dlr-rm.github.io/BlenderProc/docs/tutorials/renderer.html)：深度、实例/类别属性映射。
- [Kubric官方仓库](https://github.com/google-research/kubric)：多目标视频及丰富标注，入门环境。
- [TCOW项目与失败案例](https://tcow.cs.columbia.edu/)：合成监督和真实场景评价；相似动物遮挡仍可能混认身份。
- [Isaac Sim当前系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)：系统、驱动、GPU及资源要求。
- [Blender官方4.2.1散列](https://download.blender.org/release/Blender4.2/blender-4.2.1.sha256)：本次安装包完整性。
