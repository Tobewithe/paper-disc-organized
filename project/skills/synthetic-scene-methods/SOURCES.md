# 来源与阅读范围

2026-11-05从已有阅读、理论综合和实证复盘整理。本轮没有新增外部检索、原论文全文审读、渲染或模型实验。具体项目路径、SHA、主题范围与原来源台账见[整理来源](CURATION_SOURCES.json)。

主要综合来源是`SPATIAL_OBJECT_THEORY_SUPPORT_20261025.md`中D01–D14、D38–D69的原理与已得结果；只提炼知识，原研究方案、实施顺序和预算留在原位。合成泛化阅读审计、14项工具文档与27项专用接口来源分别保留自己的读取层级，不把它们当41篇全文论文。

主要原始入口：

- [Domain Randomization](https://arxiv.org/abs/1703.06907)、[Structured Domain Randomization](https://arxiv.org/abs/1810.10093)：成像变化与结构条件采样，原任务范围保留。
- [Meta-Sim](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kar_Meta-Sim_Learning_to_Generate_Synthetic_Datasets_ICCV_2019_paper.pdf)、[Meta-Sim2](https://arxiv.org/abs/2008.09092)、[Task2Sim](https://arxiv.org/abs/2112.00054)：已有指定方法段的生成分布与任务反馈阅读。
- [Scenic](https://arxiv.org/abs/1809.09310)、[场景生成文档](https://docs.scenic-lang.org/en/latest/reference/scene_generation.html)：关系约束、条件分布和拒绝采样，未读完全部论文或部署框架。
- [MAMMAL](https://www.nature.com/articles/s41467-023-43483-w)、[固定版本PIG_model](https://github.com/anl13/PIG_model/tree/ade41a0e389ef698ff3d3255bd1f9705ad5e7bd3)：猪体资产、rig、UV和关节回归来源。
- [BlenderProc renderer](https://dlr-rm.github.io/BlenderProc/docs/tutorials/renderer.html)、[Kubric](https://github.com/google-research/kubric)、[Blender 4.2.1](https://github.com/blender/blender/tree/v4.2.1)：指定API和版本源码，不是本轮集成验收。
- [ARAP](https://igl.ethz.ch/projects/ARAP/arap_web.pdf)、[libigl](https://libigl.github.io/tutorial/)、[Pose Space Deformation](https://www.scribblethink.org/Work/PSD/PSD.pdf)、[IPC Toolkit](https://github.com/ipc-sim/ipc-toolkit)：形状编辑、蒙皮和连续接触的理论/组件依据；PSD原记录只读摘要与引言。
- [Open3D射线](https://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html)、[SciPy DE](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.differential_evolution.html)、[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)：查询、搜索和纹理/UV接口边界。
- [WGAN](https://proceedings.mlr.press/v70/arjovsky17a.html)、[SimGAN](https://openaccess.thecvf.com/content_cvpr_2017/html/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.html)、[nvdiffrast](https://nvlabs.github.io/nvdiffrast/)：保留相关文献知识，原项目对抗及自动反馈分支的撤销继续有效。

几何核验数字、射线一致、接触与外观反例均是原实验记录，非本轮重算；自然性和真实任务泛化只按原证据解释。
