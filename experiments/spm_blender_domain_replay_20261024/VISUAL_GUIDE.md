# Blender诊断的场景、结果与技术图

最终图来自`RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d`，18组全部实际检查，见[QA](VISUAL_QA.json)。旧图Run保留；修正版只改善第一图坐标标签排版，不改模型、预测或统计。图上展示置信度≥.25的检测，正式成绩使用冻结的.001；图示少框不能当作评价没有输出。

先看[真实与全部32张渲染布局](ALL_RENDERED_LAYOUTS_01.png)及[后四布局](ALL_RENDERED_LAYOUTS_02.png)：图像在模型评价前检查，资产风格化、拥挤重建不足均保留。再看[栏位母猪的真实/四外观/两臂对照](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_02_native_fusion.png)：第一行原生、第二行融合；白线为GT，黄色为正常检测框，彩色为掩码。颜色编号不表示跨外观稳定实例匹配，比较以GT身份及数值账本为准。

| 科学图 | 用途 |
|---|---|
| [各域AP和正常检测覆盖](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/01_domain_mask_AP_and_coverage.png) | 同域两臂效用；旧单猪0/24 Box75覆盖，限制读出判断。 |
| [栏杆目标语义](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/02_rail_label_semantics.png) | 同RGB/同预测，仅GT变化，不能当作重新训练收益。 |
| [共同41实例的外观响应](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/03_common_instance_response.png) | 每布局等权响应，单独标出共同GT分母。 |
| [三维拟合保真度](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/04_scene_fitting_fidelity.png) | 渲染轮廓对真实GT的贴合程度，不是模型准确率。 |

四幅科学图均有同名SVG用于导出。全部案例入口：

| 布局／真实来源 | 场景对象 | 两臂五域对照 | 便携Blender文件 |
|---|---|---|---|
| 00／real540 | IR圈舍11猪 | [结果](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_00_native_fusion.png) | [scene.blend](runs/RUN_SPM_BLENDER_RENDER_FORMAL_339c3c42bf4f/layout_00/scene.blend) |
| 01／real1514 | IR圈舍11猪 | [结果](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_01_native_fusion.png) | [scene.blend](runs/RUN_SPM_BLENDER_RENDER_FORMAL_339c3c42bf4f/layout_01/scene.blend) |
| 02／real1088 | 栏位5母猪 | [结果](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_02_native_fusion.png) | [scene.blend](runs/RUN_SPM_BLENDER_RENDER_FORMAL_339c3c42bf4f/layout_02/scene.blend) |
| 03／real1558 | 栏位5母猪 | [结果](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_03_native_fusion.png) | [scene.blend](runs/RUN_SPM_BLENDER_RENDER_FORMAL_339c3c42bf4f/layout_03/scene.blend) |
| 04／real545 | 红地板2母猪 | [结果](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_04_native_fusion.png) | [scene.blend](runs/RUN_SPM_BLENDER_RENDER_FORMAL_339c3c42bf4f/layout_04/scene.blend) |
| 05／real546 | 红地板2母猪 | [结果](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_05_native_fusion.png) | [scene.blend](runs/RUN_SPM_BLENDER_RENDER_FORMAL_339c3c42bf4f/layout_05/scene.blend) |
| 06／real445 | 母猪/仔猪19，拥挤拟合不足 | [结果](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_06_native_fusion.png) | [scene.blend](runs/RUN_SPM_BLENDER_RENDER_FORMAL_339c3c42bf4f/layout_06/scene.blend) |
| 07／real928 | 母猪/仔猪18，拥挤拟合不足 | [结果](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/layout_07_native_fusion.png) | [scene.blend](runs/RUN_SPM_BLENDER_RENDER_FORMAL_339c3c42bf4f/layout_07/scene.blend) |

每个`layout_XX`目录含四外观PNG、两种标签的`labels.npz`和相机/实例/材质/光照配方`SCENE.json`。`.blend`保留可编辑实际三维场景，四外观图以PNG/配方交付，不声称四个独立资产。GT来自实际评估三角形射线，模型从未读取GT。

全部旧单猪案例按六姿态展示：[站立](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/legacy_standing_native_fusion.png)、[行走](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/legacy_walking_native_fusion.png)、[进食](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/legacy_feeding_native_fusion.png)、[趴卧](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/legacy_prone_native_fusion.png)、[左侧卧](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/legacy_lateral_left_native_fusion.png)、[右侧卧](runs/RUN_SPM_BLENDER_DOMAIN_VISUALS_132154dc0c9d/legacy_lateral_right_native_fusion.png)。这些图只说明已选资产的正常检测域差，不以零Mask AP单独否定记忆能力。

[真实GT/栏杆叠图](reference_GT_and_rails.png)、[18图视觉核验](VISUAL_QA.json)、[完整研究判断](REPORT.md)和[数值来源](DIAGNOSIS_NUMBERS.json)共同解释结果，不从少数示例推总体收益。
